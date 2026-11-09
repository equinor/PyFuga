"""_fuga.py
==========
Main API entry point for PyFuga — generate wake look-up tables.

This module orchestrates the full three-stage PyFuga pipeline:

1. **Preliminary LUTs (preLUTs)** — via :mod:`pyfuga.preluts_generator`
   Integrates atmospheric boundary layer equations between vertical stations.

2. **Fourier LUTs** — via :mod:`pyfuga.flut`
   Solves for spectral wake response to all forcing directions.

3. **Physical-space LUTs** — via :mod:`pyfuga.trafalgar`
   Inverse Fourier transform to obtain wake fields in physical x-y-z space.

The entry point is :func:`get_luts`, which orchestrates all three stages,
handling file caching and data I/O.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from pyfuga import utils
from pyfuga.constants import ACCGOAL, DS, UVW_LT
from pyfuga.flut import FourierLUTGenerator, filter_kz0
from pyfuga.paths import get_fluts_path, get_level_range, get_luts_path, get_preluts_path
from pyfuga.preluts import PreLUTs
from pyfuga.trafalgar import Trafalgar
from pyfuga.utils import ComplexXRDataset, NCpuStageConfig, get_beta_lst, get_kz0_lst


def get_luts(
    folder: str | Path,
    zeta0: float,
    nkz0: int,
    nbeta: int,
    diameter: float,
    zhub: float,
    z0: float,
    zi: float,
    zlow: float,
    zhigh: float,
    lut_vars: list[str] = UVW_LT,
    nx: int = 2048,
    ny: int = 512,
    dx: float | None = None,
    dy: float | None = None,
    jit: bool = True,
    n_cpu: int | None = None,
    n_cpu_stage: NCpuStageConfig | None = None,
    verbose: bool = True,
    *,
    cache_intermediate: bool = False,
    single_precision: bool = False,
    accgoal: float = ACCGOAL,
) -> xr.Dataset:  # pragma: no cover
    """Generate physical-space wake look-up tables via the full PyFuga pipeline.

    This function orchestrates the three-stage PyFuga pipeline:
        1. Preliminary LUTs (preLUTs) via pyfuga.preluts_generator
        2. Fourier LUTs via pyfuga.flut
        3. Physical-space LUTs via pyfuga.trafalgar

    Reuses compatible cached files and saves the final LUT. Intermediate files
    are written only when ``cache_intermediate=True``.

    Parameters
    ----------
    folder : str or Path
        Output folder for the final LUT and optional intermediate caches.
    zeta0 : float
        Monin-Obukhov stability parameter (z0/L). Positive=stable, negative=unstable,
        zero=neutral.
    nkz0 : int
        Spectral density: number of wavenumbers per log decade.
        Total count = 9*nkz0 - (nkz0-1); typically 8--16.
    nbeta : int
        Angular resolution for azimuthal averaging. Total angle count is nbeta + 1.
    diameter : float
        Wind turbine rotor diameter (m).
    zhub : float
        Wind turbine hub height (m).
    z0 : float
        Aerodynamic roughness length (m); defines vertical grid via s = log(z/z0).
    zi : float
        Atmospheric inversion height (m); upper domain boundary.
    zlow : float
        Lowest output height (m). If zlow == zhigh == zhub, only hub-height solution.
    zhigh : float
        Highest output height (m). If zlow == zhigh == zhub, only hub-height solution.
    lut_vars : sequence of str, optional
        Variable names to compute. Default is all 8: 'UL', 'UT', 'VL', 'VT', 'WL',
        'WT', 'PL', 'PT' (L=longitudinal, T=transverse forcing).
    nx : int, optional
        Streamwise grid points, default 2048. Turbine at x = nx/4 * dx.
    ny : int, optional
        Cross-stream half-grid size, default 512.
    dx : float or None, optional
        Streamwise grid spacing (m). If None (default), set to diameter / 4.
    dy : float or None, optional
        Cross-stream grid spacing (m). If None (default), set to diameter / 16.
    jit : bool, optional
        Enable JIT compilation (Numba). Default is True.
    n_cpu : int or None, optional
        Optional global worker-count override. If ``None`` (default), stage-aware
        defaults are used and all stages run on all CPUs. If an integer is provided, that
        worker count is applied to all stages unless overridden by
        ``n_cpu_stage``. Large ``nx``/``ny`` and multiple workers can increase
        memory use substantially.
    n_cpu_stage : dict, optional
        Optional per-stage overrides for worker counts. Supported keys are
        ``"preluts"``, ``"fluts"``, and ``"trafalgar"``. Provided keys override
        the resolved default/global setting for that stage. Precedence is:
        defaults -> global ``n_cpu`` (if provided) -> ``n_cpu_stage`` keys.
        A stage value may be ``None`` meaning all CPUs for that stage only.
    verbose : bool, optional
        Show progress bars for stages that are generated. Default is True.
    cache_intermediate : bool, optional
        Save preLUTs and Fourier LUTs for reuse across configurations. Default
        is False: keep intermediate results in memory and generate only the
        preLUT wavenumbers needed by this turbine. Existing caches are still reused.
    single_precision : bool, optional
        Run the final physical-space transform in single precision for speed.
        Default is False. The LUT is still stored as float64, under a separate
        ``_sp`` file name; values differ from the default by roughly 1e-7
        relative to the field maximum.
    accgoal : float, optional
        Accuracy goal of the adaptive preLUT integration. Default is 1e-4.
        A looser goal (e.g. 1e-3) makes the preLUT stage faster at the cost of
        accuracy. Non-default values are cached under separate ``_acc`` file names.

    Returns
    -------
    xarray.Dataset
        Physical-space wake fields with dimensions (z, x, y).

    See Also
    --------
    pyfuga.preluts_generator.PreLUTGenerator.make_prelut : First pipeline stage.
    pyfuga.flut.FourierLUTGenerator.make_lut : Second pipeline stage.
    pyfuga.trafalgar.Trafalgar.make_luts : Third pipeline stage.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resolved_n_cpu = utils.resolve_n_cpu_stage(n_cpu, n_cpu_stage)
    dx = dx or diameter / 4
    dy = dy or diameter / 16

    luts_path = get_luts_path(
        folder,
        zeta0,
        nkz0,
        nbeta,
        diameter,
        zhub,
        z0,
        zi,
        zlow,
        zhigh,
        lut_vars,
        nx,
        ny,
        dx,
        dy,
        single_precision=single_precision,
        accgoal=accgoal,
    )
    if luts_path.exists():
        luts = xr.load_dataset(luts_path)
        if utils.load_lut_if_current(luts, luts_path):
            luts.attrs["name"] = luts_path.stem
            return luts

    fluts = get_fluts(
        folder,
        zeta0,
        nkz0,
        nbeta,
        diameter,
        zhub,
        z0,
        zi,
        zlow,
        zhigh,
        lut_vars,
        jit,
        resolved_n_cpu["fluts"],
        n_cpu_stage={"preluts": resolved_n_cpu["preluts"]},
        verbose=verbose,
        cache_intermediate=cache_intermediate,
        accgoal=accgoal,
    )
    luts = Trafalgar(fluts, nx, ny, dx, dy, verbose=verbose).make_luts(
        interpolation_order=3, n_cpu=resolved_n_cpu["trafalgar"], single_precision=single_precision
    )
    if single_precision:
        luts.attrs["single_precision"] = 1
    utils.stamp_lut_metadata(luts)
    luts.to_netcdf(luts_path)
    luts.attrs["name"] = luts_path.stem
    return luts


def get_fluts(
    folder: str | Path,
    zeta0: float,
    nkz0: int,
    nbeta: int,
    diameter: float,
    zhub: float,
    z0: float,
    zi: float,
    zlow: float,
    zhigh: float,
    lut_vars: list[str] = UVW_LT,
    jit: bool = True,
    n_cpu: int | None = None,
    n_cpu_stage: NCpuStageConfig | None = None,
    verbose: bool = True,
    *,
    cache_intermediate: bool = True,
    accgoal: float = ACCGOAL,
) -> ComplexXRDataset:  # pragma: no cover
    """Reuse or generate Fourier LUTs; optionally persist intermediate results."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resolved_n_cpu = utils.resolve_n_cpu_stage(n_cpu, n_cpu_stage)

    fluts_path = get_fluts_path(
        folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh, lut_vars, accgoal=accgoal
    )
    if fluts_path.exists():
        fluts = ComplexXRDataset.from_netcdf(fluts_path)
        if utils.load_lut_if_current(fluts, fluts_path):
            return fluts

    utils.compile(jit)
    if cache_intermediate or get_preluts_path(folder, zeta0, nkz0, nbeta, accgoal=accgoal).exists():
        preluts = get_preluts(
            folder,
            zeta0,
            nkz0,
            nbeta,
            jit,
            resolved_n_cpu["preluts"],
            None,
            verbose=verbose,
            save=cache_intermediate,
            accgoal=accgoal,
        )
    else:
        kz0_lst = get_kz0_lst(nkz0, 1e-9, 1e-1)
        preluts = _make_preluts(
            zeta0,
            nkz0,
            nbeta,
            jit,
            resolved_n_cpu["preluts"],
            verbose,
            filter_kz0(kz0_lst, z0, diameter / 2),
            accgoal=accgoal,
        )
        # Preserve the full-grid provenance in the final LUT's attributes.
        preluts.attrs["kz0_lst"] = kz0_lst

    low_level_out, high_level_out = get_level_range(zlow, zhigh, zhub, z0)

    flut_generator = FourierLUTGenerator(preluts, zhub, diameter, zi, verbose=verbose)
    if low_level_out == high_level_out == 9999:
        fluts = flut_generator.make_hubheight_luts(z0, lut_vars, n_cpu=resolved_n_cpu["fluts"])
    else:
        fluts = flut_generator.make_lut(z0, low_level_out, high_level_out, lut_vars, n_cpu=resolved_n_cpu["fluts"])

    utils.stamp_lut_metadata(fluts)
    if cache_intermediate:
        fluts.save(fluts_path)
    return fluts


def get_preluts(
    folder: str | Path,
    zeta0: float,
    nkz0: int,
    nbeta: int,
    jit: bool = True,
    n_cpu: int | None = None,
    n_cpu_stage: NCpuStageConfig | None = None,
    verbose: bool = True,
    *,
    save: bool = True,
    accgoal: float = ACCGOAL,
) -> PreLUTs:  # pragma: no cover
    """Reuse or generate preLUTs, saving newly generated data unless ``save=False``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    resolved_n_cpu = utils.resolve_n_cpu_stage(n_cpu, n_cpu_stage)

    preluts_path = get_preluts_path(folder, zeta0, nkz0, nbeta, accgoal=accgoal)
    if preluts_path.exists():
        preluts = PreLUTs.from_netcdf(preluts_path)
        if utils.load_lut_if_current(preluts, preluts_path):
            return preluts

    preluts = _make_preluts(zeta0, nkz0, nbeta, jit, resolved_n_cpu["preluts"], verbose, accgoal=accgoal)
    if save:
        preluts.save(preluts_path)
    return preluts


def _make_preluts(
    zeta0: float,
    nkz0: int,
    nbeta: int,
    jit: bool,
    n_cpu: int,
    verbose: bool,
    kz0_lst: npt.NDArray[np.float64] | None = None,
    accgoal: float = ACCGOAL,
) -> PreLUTs:
    utils.compile(jit)
    preluts = PreLUTs.make_preluts(
        zeta0=zeta0,
        kz0_lst=get_kz0_lst(nkz0, 1e-9, 1e-1) if kz0_lst is None else kz0_lst,
        beta_lst=get_beta_lst(nbeta),
        kzmax=300,
        ds=DS,
        accgoal=accgoal,
        jit=jit,
        n_cpu=n_cpu,
        verbose=verbose,
    )
    preluts.attrs["nkz0"] = nkz0
    preluts.attrs["nbeta"] = nbeta
    utils.stamp_lut_metadata(preluts)
    return preluts


def main() -> None:
    if __name__ == "__main__":  # pragma: no cover
        luts = get_luts(
            folder=Path(__file__).parent.parent,
            zeta0=0,
            nkz0=16,
            nbeta=32,
            diameter=80,
            zhub=70,
            z0=0.00001,
            zi=400,
            zlow=70,
            zhigh=70,
            lut_vars=["UL"],
        )
        import matplotlib.pyplot as plt

        luts.UL.sel(z=70)[502:542, :40].plot(x="x")
        plt.axis("equal")
        plt.figure()
        luts = get_luts(
            folder=Path(__file__).parent.parent,
            zeta0=0,
            nkz0=16,
            nbeta=32,
            diameter=80,
            zhub=70,
            z0=0.00001,
            zi=400,
            zlow=30,
            zhigh=110,
            lut_vars=["UL"],
        )

        luts.UL.sel(x=500)[:, :50].plot(x="y")
        plt.axis("equal")
        plt.show()


main()
