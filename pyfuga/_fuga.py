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

import os
from pathlib import Path

import numpy as np
import xarray as xr

from pyfuga import utils
from pyfuga.constants import UVW_LT
from pyfuga.flut import FourierLUTGenerator
from pyfuga.preluts import PreLUTs
from pyfuga.trafalgar import Trafalgar
from pyfuga.utils import ComplexXRDataset, get_beta_lst, get_kz0_lst


def get_luts(
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
    lut_vars=UVW_LT,
    nx=2048,
    ny=512,
    dx=None,
    dy=None,
    jit=True,
    n_cpu=1,
):  # pragma: no cover
    """Generate physical-space wake look-up tables via the full PyFuga pipeline.

    This function orchestrates the three-stage PyFuga pipeline:
        1. Preliminary LUTs (preLUTs) via pyfuga.preluts_generator
        2. Fourier LUTs via pyfuga.flut
        3. Physical-space LUTs via pyfuga.trafalgar

    Caches intermediate results (preLUTs and Fourier LUTs) as NetCDF files in
    the specified folder, skipping recomputation if files exist.

    Parameters
    ----------
    folder : str or Path
        Output folder for all generated files (preLUTs, Fourier LUTs, LUTs).
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
        CPU cores for parallel computation. None uses all; 1 disables parallelisation.

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
    utils.compile(jit)
    folder = Path(folder)
    os.makedirs(folder, exist_ok=True)
    dx = dx or diameter / 4
    dy = dy or diameter / 16

    preluts_id = f"Zeta0={zeta0:3.2e}_{nkz0}_{nbeta}"

    L_vars = [v[0] for v in lut_vars if v[1] == "L"]
    T_vars = [v[0] for v in lut_vars if v[1] == "T"]
    lut_vars_id = ""
    if L_vars:
        lut_vars_id += f"_{''.join(L_vars)}L"
    if T_vars:
        lut_vars_id += f"_{''.join(T_vars)}T"

    ds = 0.05
    if zlow == zhigh == zhub:
        low_level_out = high_level_out = 9999
        z_id = f"z{zhub:.1f}"
    else:
        low_level_out = int(np.floor(np.log(zlow / z0) / ds))
        high_level_out = int(np.ceil(np.log(zhigh / z0) / ds))
        zlow = z0 * np.exp(low_level_out * ds)
        zhigh = z0 * np.exp(high_level_out * ds)
        z_id = f"z{zlow:.1f}-{zhigh:.1f}"

    fluts_id = preluts_id + f"_D{diameter}_zhub{zhub}_zi{zi}_z0={z0:.8f}_{z_id}{lut_vars_id}"

    luts_id = fluts_id + f"_nx{nx}_ny{ny}_dx{dx}_dy{dy}"

    luts_path = folder / f"LUTs_{luts_id}.nc"
    if not luts_path.exists():
        # LUTs are missing (run Trafalgar)
        fluts_path = folder / f"fLUTs_{fluts_id}.nc"
        if not fluts_path.exists():
            # FourierLUTs are missing (run lut)

            preluts_path = folder / f"preLUTs_{preluts_id}.nc"
            if not preluts_path.exists():
                # Preluts are missing (run prelut)
                preluts = PreLUTs.make_preluts(
                    zeta0=zeta0,
                    kz0_lst=get_kz0_lst(nkz0, 1e-9, 1e-1),
                    beta_lst=get_beta_lst(nbeta),
                    kzmax=300,
                    ds=ds,
                    accgoal=0.0001,
                    jit=jit,
                    n_cpu=n_cpu,
                )
                preluts.attrs["nkz0"] = nkz0
                preluts.attrs["nbeta"] = nbeta
                preluts.save(preluts_path)
            else:
                preluts = PreLUTs.from_netcdf(preluts_path)

            # preluts loaded make fourier luts
            flut_generator = FourierLUTGenerator(preluts, zhub, diameter, zi)
            if low_level_out == high_level_out == 9999:
                fluts = flut_generator.make_hubheight_luts(z0, lut_vars, n_cpu=n_cpu)
            else:
                fluts = flut_generator.make_lut(z0, low_level_out, high_level_out, lut_vars, n_cpu=n_cpu)

            fluts.save(fluts_path)
        else:
            fluts = ComplexXRDataset.from_netcdf(fluts_path)

        # fourier luts loaded make luts
        luts = Trafalgar(fluts, nx, ny, dx, dy).make_luts(3)
        luts.to_netcdf(luts_path)
    else:
        luts = xr.load_dataset(luts_path)
    luts.attrs["name"] = luts_path.stem
    return luts


def main():
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
