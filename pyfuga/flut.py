"""flut.py
========
Generate Fourier-space look-up tables from atmospheric boundary layer solutions.

This module implements the second stage of the PyFuga pipeline: the
Fourier-space perturbation solution derived from the adjoint-problem
solution stored in the preliminary LUTs.

**Input**
    Preliminary LUTs (:class:`pyfuga.preluts.PreLUT`) from
    :mod:`pyfuga.preluts_generator`. These encode the state profile
    (velocity, pressure) as a function of height and horizontal wavenumber.

**Output**
    Fourier LUTs (:class:`xarray.Dataset`) with variables ``UL``, ``VL``, ``WL``,
    ``PL`` (L = longitudinal forcing) and ``UT``, ``VT``, ``WT``, ``PT``
    (T = transverse forcing), each with dimensions ``(beta, kz0, level)``.
    These are passed to :mod:`pyfuga.trafalgar` for inverse Fourier transform.

**Method**
    Solves the superposition integral equation for each ``(beta, kz0)`` pair
    (polar angle and wavenumber), ranging over all levels in the rotor plane
    or at hub height. Uses parallel processing where applicable.
"""

import warnings
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from numba import njit
from numpy import linalg
from numpy import newaxis as na
from tqdm import tqdm  # pyright: ignore[reportMissingImports]

from pyfuga.constants import FIRST_Z_OUTPUT, KAPPA, UVW_LT
from pyfuga.preluts import PreLUTs
from pyfuga.utils import ComplexXRDataset, jit, multiprocessing_map, normalise_n_cpu_workers

# Index into the 6-component state vector for each LUT variable prefix (U, V, W, P).
_LUT_COMPONENT_INDEX: dict[str, int] = {"U": 0, "V": 2, "W": 4, "P": 5}


class SolveLayersJob(NamedTuple):
    """Arguments for :func:`solve_layers` (one ``(beta, kz0, forcing)`` task)."""

    prelut: Any
    beta: Any
    kz0: Any
    z0: float
    zhub: float
    radius: float
    forcing: Literal["L", "T"]
    lowerjf: int
    upperjf: int
    minlevel: int
    maxlevel: int
    low_level_out: int
    high_level_out: int


class _PreLUTArrays:
    """Picklable subset of a single-node preLUT: plain arrays and ``ds`` only."""

    __slots__ = ("ds", "arrays")

    def __init__(self, ds: float, arrays: dict[str, Any]) -> None:
        self.ds = ds
        self.arrays = arrays

    def __reduce__(self) -> tuple[Any, ...]:
        return (_PreLUTArrays, (self.ds, self.arrays))

    def __getattr__(self, name: str) -> Any:
        try:
            return self.arrays[name]
        except KeyError:
            raise AttributeError(name) from None

    __getitem__ = __getattr__


def filter_kz0(kz0_lst: npt.NDArray[np.float64], z0: float, radius: float) -> npt.NDArray[np.float64]:
    """Keep the spectral grid up to the rotor's existing wavelength cutoff."""
    nkz0 = int(np.round(1 / np.diff(np.log10(kz0_lst))[0])) if len(kz0_lst) > 1 else 1
    aux = z0 * np.pi * 8.0 / radius
    kz0limit = 10.0 ** (np.ceil(np.log10(aux) * nkz0) / nkz0)
    return cast(npt.NDArray[np.float64], kz0_lst[kz0_lst <= kz0limit])


def _select_prelut_arrays(
    preluts: PreLUTs, beta_kz0_lst: list[tuple[Any, Any]], names: list[str]
) -> Iterator[_PreLUTArrays]:
    """Select compact nodes by contiguous views rather than repeated full-table masks."""
    if "kz0_beta_i" not in preluts.dims:
        for beta, kz0 in beta_kz0_lst:
            prelut = preluts.sel(beta=beta, kz0=kz0)
            yield _PreLUTArrays(prelut.ds, {n: np.asarray(prelut[n]) for n in names})
        return

    beta, kz0 = preluts.beta.values, preluts.kz0.values
    boundaries = np.r_[0, np.flatnonzero((beta[1:] != beta[:-1]) | (kz0[1:] != kz0[:-1])) + 1, len(beta)]
    slices = {
        (beta[start], kz0[start]): slice(start, end) for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    }
    arrays = {n: np.asarray(preluts[n]) for n in names}
    for b, k in beta_kz0_lst:
        sl = slices[b, k] if len(slices) == len(boundaries) - 1 else (beta == b) & (kz0 == k)
        yield _PreLUTArrays(preluts.ds, {n: a[sl] for n, a in arrays.items()})


def _iter_solve_layer_jobs(
    preluts: PreLUTs,
    beta_kz0_lst: list[tuple[Any, Any]],
    z0: float,
    zhub: float,
    radius: float,
    forcing: Literal["L", "T"],
    lowerjf: int,
    upperjf: int,
    minlevel: int,
    maxlevel: int,
    low_level_out: int,
    high_level_out: int,
    prelut_lst: list[Any] | None = None,
) -> Iterator[SolveLayersJob]:
    axis = "x" if forcing == "L" else "y"
    names = ["level", "Y_lower", "R_upper", "R_lower", f"db{axis}_const", f"db{axis}_lin"]
    prelut_iter = _select_prelut_arrays(preluts, beta_kz0_lst, names) if prelut_lst is None else iter(prelut_lst)
    for (beta, kz0), prelut in zip(beta_kz0_lst, prelut_iter, strict=True):
        yield SolveLayersJob(
            prelut=_PreLUTArrays(prelut.ds, {n: np.asarray(prelut[n]) for n in names}),
            beta=beta,
            kz0=kz0,
            z0=z0,
            zhub=zhub,
            radius=radius,
            forcing=forcing,
            lowerjf=lowerjf,
            upperjf=upperjf,
            minlevel=minlevel,
            maxlevel=maxlevel,
            low_level_out=low_level_out,
            high_level_out=high_level_out,
        )


class FourierLUTGenerator:
    def __init__(
        self,
        preluts: PreLUTs,
        zhub: float,
        diameter: float,
        zi: float,
        verbose: bool = True,
    ) -> None:
        self.preluts = preluts
        self.zhub = zhub
        self.radius = diameter / 2
        self.zi = zi  # Domain height
        self.verbose = verbose

    def make_rotor_luts(
        self,
        z0: float,
        luts: list[str] = UVW_LT,
        n_cpu: int | None = None,
    ) -> ComplexXRDataset:  # for wake solution at all levels within rotor plane
        # note: can change z0 --> modify T.I. through z=z0*exp(1/T.I.)
        ds = self.preluts.ds
        # Calculate j levels within rotor plane, including points at ceiling (zh+R) and floor (zh-R).
        low_level_out = int(np.floor(np.log((self.zhub - self.radius) / z0) / ds))
        high_level_out = int(np.ceil(np.log((self.zhub + self.radius) / z0) / ds))
        return self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)

    def make_hubheight_luts(
        self,
        z0: float,
        luts: list[str] = UVW_LT,
        n_cpu: int | None = None,
    ) -> ComplexXRDataset:  # for wake solution only at hub height
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log(self.zhub / z0) / ds))
        high_level_out = int(np.ceil(np.log(self.zhub / z0) / ds))

        flut_ds = self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)
        if low_level_out == high_level_out:
            lut_vars = flut_ds.isel(level=0)
        else:
            a = np.log(self.zhub / z0) / ds - low_level_out
            lut_vars = flut_ds.isel(level=0) * (1 - a) + flut_ds.isel(level=1) * (a)
        lut_vars["level"] = 9999

        return ComplexXRDataset(
            data_vars={
                "z": (("level"), [self.zhub]),
                "k": flut_ds.k,
                "diameter": flut_ds.diameter,
                "hubheight": self.zhub,
                "z0": z0,
                # fourier luts, UL, UT, ...
                **{k: lut_vars[k].expand_dims("level", 2) for k in lut_vars if lut_vars[k].dims == ("beta", "kz0")},
            },
            coords={"beta": flut_ds.beta, "kz0": flut_ds.kz0, "level": [9999]},
            attrs=flut_ds.attrs,
        )

    def make_lut(
        self,
        z0: float,
        low_level_out: int,
        high_level_out: int,
        luts: list[str] = UVW_LT,
        n_cpu: int | None = None,
    ) -> ComplexXRDataset:
        """
        Generate Fourier-space wake look-up tables for specified height range.

        This is the core method of the second stage in the PyFuga pipeline. It
        takes preliminary LUTs containing the adjoint-problem solution and solves
        for spectral wake response across all ``(beta, kz0)`` wavenumber pairs
        for the specified problem geometry and turbine characteristics. Output is
        suitable for inverse Fourier transformation by
        :class:`pyfuga.trafalgar.Trafalgar`.

        Parameters
        ----------
        z0 : float
            Roughness length (m). Defines the vertical grid spacing via logarithmic
            height increments (s = log(z/z0)).
        low_level_out : int
            Lowest output level index. Defines minimum height on the vertical grid.
        high_level_out : int
            Highest output level index. Defines maximum height on the vertical grid.
        luts : sequence of str, optional
            Variable names to generate. Default is all 8 variables: 'UL', 'UT', 'VL',
            'VT', 'WL', 'WT', 'PL', 'PT' (L = longitudinal forcing, T = transverse
            forcing). Variables are u, v, w (velocity components) and p (pressure).
        n_cpu : int or None, optional
            Number of CPU cores for parallel (beta, kz0) computation. Default is
            ``None`` (all available CPUs, at least ``1``). Pass ``1`` for serial
            execution. Values greater than ``1`` enable a process pool.

        Returns
        -------
        xarray.Dataset
            Fourier-space wake fields with variables (as requested), each with
            dimensions (beta, kz0, level). Dimensions correspond to:
            - beta: azimuthal angles (radians)
            - kz0: wavenumber magnitude scaled to domain height
            - level: vertical index on the log-height grid
            Also includes scalar coordinates: diameter, hubheight, z0.

        See Also
        --------
        pyfuga.trafalgar.Trafalgar.make_luts : Next pipeline stage; inverse Fourier transform.
        pyfuga.preluts_generator.PreLUTGenerator.make_prelut : Previous stage; generates input.

        Notes
        -----
        The method validates that output levels fit within the precomputed preLUT range.
        Wavenumbers larger than those with wavelengths < R/4 (rotor radius/4) are filtered
        out as physically irrelevant. The computation is embarrassingly parallel over
        (beta, kz0) pairs and uses multiprocessing when n_cpu > 1.
        """
        invalid = [lut for lut in luts if lut not in UVW_LT]
        if invalid:
            raise ValueError(f"luts must be a subset of {list(UVW_LT)!r}, invalid entries: {invalid!r}")
        workers = normalise_n_cpu_workers(n_cpu)
        zh = self.zhub
        R = self.radius
        ds, _, zeta0 = [self.preluts.attrs[k] for k in ["ds", "kzmax", "zeta0"]]

        # Get all unique kz0 values from the prelut, sorted in ascending order
        kz0_lst = np.sort(np.unique(self.preluts.kz0.values))

        # Filter out wavenumbers that are too large (wavelengths smaller than R/4)
        kz0_lst = filter_kz0(kz0_lst, z0, R)

        upperjf = int(np.floor(np.log((zh + R) / z0) / ds))  # top of rotor, excluding point at ceiling (zh+R).
        lowerjf = int(np.ceil(np.log((zh - R) / z0) / ds))  # bottom of rotor, excluding point at floor (zh-R).
        minlevel = int(np.floor(np.log(np.maximum(FIRST_Z_OUTPUT / z0, 1)) / ds))  # ground
        maxlevel = int(np.ceil(np.log(self.zi / z0) / ds))  # inversion height

        # Validate that output levels are within the range of prelut levels
        if low_level_out < minlevel + 1:
            warnings.warn(
                f"LoLevelOut ({low_level_out}) raised to MinLevel ({minlevel + 1}).",
                UserWarning,
                stacklevel=2,
            )
            low_level_out = minlevel + 1

        if high_level_out > maxlevel - 1:
            warnings.warn(
                f"HiLevelOut ({high_level_out}) lowered to MaxLevel ({maxlevel - 1}).",
                UserWarning,
                stacklevel=2,
            )
            high_level_out = maxlevel - 1

        log_zi_z0 = np.log(self.zi / z0)

        # Calculate maximum meaningful vertical wake extent
        # assert smaxx>self.preluts.ds * upperjf # TODO: smaxx not available here
        smax = np.minimum(np.log((16 * np.pi * (1 + zh / R) + 120.0) / kz0_lst), log_zi_z0)  # Neutral
        if np.abs(zeta0) > 0:
            smax = np.minimum(smax, np.log(np.abs(15 / zeta0)))  # Stable and unstable

        # Create list of (beta, kz0) combinations for which to solve the layers
        beta_lst = np.sort(np.unique(self.preluts.beta.values))
        ktab = kz0_lst / z0
        beta_kz0_lst = [(beta, kz0) for beta in beta_lst for kz0 in kz0_lst]

        xL: list[npt.NDArray[np.complex128]] = []  # longitudinal forcing
        xT: list[npt.NDArray[np.complex128]] = []  # transversal forcing

        do_l = any(lut[1] == "L" for lut in luts)
        do_t = any(lut[1] == "T" for lut in luts)
        prelut_lst = (
            list(
                _select_prelut_arrays(
                    self.preluts,
                    beta_kz0_lst,
                    ["level", "Y_lower", "R_upper", "R_lower", "dbx_const", "dbx_lin", "dby_const", "dby_lin"],
                )
            )
            if do_l and do_t
            else None
        )  # select once, share between forcings

        # With the loops compiled a job takes well under a millisecond, so running them in-process beats both
        # worker processes (start-up, copying preLUTs) and threads (contention for the GIL between short kernels).
        pool = nullcontext(map) if _up_sweep is _up_sweep_nb else multiprocessing_map(workers)
        with pool as map_func:
            if any(lut[1] == "L" for lut in luts):
                jobs_l = _iter_solve_layer_jobs(
                    self.preluts,
                    beta_kz0_lst,
                    z0,
                    self.zhub,
                    self.radius,
                    "L",
                    lowerjf,
                    upperjf,
                    minlevel,
                    maxlevel,
                    low_level_out,
                    high_level_out,
                    prelut_lst,
                )
                xL = list(
                    tqdm(
                        map_func(solve_layers, jobs_l),
                        total=len(beta_kz0_lst),
                        disable=(not self.verbose),
                        desc="Fourier LUTS: Solving for longitudinal forcing",
                    )
                )

            if any(lut[1] == "T" for lut in luts):
                jobs_t = _iter_solve_layer_jobs(
                    self.preluts,
                    beta_kz0_lst,
                    z0,
                    self.zhub,
                    self.radius,
                    "T",
                    lowerjf,
                    upperjf,
                    minlevel,
                    maxlevel,
                    low_level_out,
                    high_level_out,
                    prelut_lst,
                )
                xT = list(
                    tqdm(
                        map_func(solve_layers, jobs_t),
                        total=len(beta_kz0_lst),
                        disable=(not self.verbose),
                        desc="Fourier LUTS: Solving for transversal forcing",
                    )
                )

        xL_array, xT_array = np.asarray(xL), np.asarray(xT)

        def get_var(s: str) -> npt.NDArray[np.complex128]:
            """Extract specific component and forcing direction from results."""
            x = xL_array if s[1] == "L" else xT_array
            i = _LUT_COMPONENT_INDEX[s[0]]
            # Extract: (n_beta, n_kz0, n_levels)
            return np.reshape(x, (len(beta_lst), len(kz0_lst)) + np.shape(x)[1:])[..., i]

        level = np.arange(low_level_out, high_level_out + 1)
        z = z0 * np.exp(ds * level)  # defining j levels / stations

        return ComplexXRDataset(
            data_vars={
                "z": (("level"), z),
                "k": (("kz0"), ktab),
                "diameter": R * 2,
                "hubheight": zh,
                "z0": z0,
                **{k: (("beta", "kz0", "level"), get_var(k)) for k in luts},
            },
            coords={"beta": beta_lst, "kz0": kz0_lst, "level": level},
            attrs={"zi": self.zi, **self.preluts.attrs},
        )


def solve_layers(job: SolveLayersJob | tuple[Any, ...]) -> npt.NDArray[np.complex128]:
    """
    Solve for a single (beta, kz0, forcing direction) combination.

    Discards pre-calculated levels outside of min/max level range for memory optimisation.
    Returns ``X(z)`` at all output levels for that (beta, kz0, forcing) pair.
    See eq. (5) in Forcing decomposition for the definition of triangular forcing.

    Parameters
    ----------
    job
        :class:`SolveLayersJob` or a 14-element tuple in the same field order (for tests
        and backward compatibility).
    """
    if not isinstance(job, SolveLayersJob):
        job = SolveLayersJob._make(job)

    if job.forcing not in ("L", "T"):
        raise ValueError(f"forcing must be 'L' or 'T', got {job.forcing!r}")

    axis = job.forcing.replace("L", "x").replace("T", "y")
    ds = job.prelut.ds
    jf_l = np.arange(job.lowerjf + 1, job.upperjf)  # layers within rotor

    z = job.z0 * np.exp(ds * (jf_l + np.array([-1, 0, 1])[:, na]))  # height blow, at and above current layer
    k = job.kz0 / job.z0

    max_table_level = np.nanmax(job.prelut.level).item()

    # Memory optimisation: discard irrelevant prelut data
    if max_table_level < job.minlevel:
        return np.zeros(
            (job.high_level_out - job.low_level_out + 1, 6),
            dtype=np.complex128,
        )  # Discard if below min level

    level_all = np.asarray(job.prelut.level)
    imin, imax = np.searchsorted(level_all, [job.minlevel, min(job.maxlevel, max_table_level)])
    sl = slice(imin, imax + 1)  # discard unused levels (outside min/max range); i is the positional index

    zero_pad_levels = int(max(0, job.maxlevel - max_table_level))

    levels = level_all[sl].astype(int)
    ijf0_l, ijf1_l, ijf2_l = np.searchsorted(levels, [jf_l - 1, jf_l, jf_l + 1])  # the three levels per layer
    # e.g. ijf0_l[0], ijf1_l[0], ijf2_l[0] makes up the first layer

    Q = np.asarray(job.prelut.Y_lower)[sl]
    R_upper = np.asarray(job.prelut.R_upper)[sl]
    R_lower = np.asarray(job.prelut.R_lower)[sl]
    db_const = np.asarray(job.prelut[f"db{axis}_const"])[sl]
    db_lin = np.asarray(job.prelut[f"db{axis}_lin"])[sl]

    # Triangular/Chapeau function weights
    phi_const_low_lst = z[0] / (KAPPA * k * (z[1] - z[0]))  # constant part of Chapeau function for lowest part of layer
    phi_lin_low_lst = 1 / (KAPPA * (z[1] - z[0]) * k**2)  # prop. part of Chapeau function for lowest part of layer
    phi_const_high_lst = z[2] / (KAPPA * k * (z[1] - z[2]))  # const. -- for upper part of layer
    phi_lin_high_lst = 1 / (KAPPA * (z[1] - z[2]) * k**2)

    # Apply geometric weighting (Fourier transform of circular rotor)
    zf = job.z0 * np.exp(ds * np.arange(job.lowerjf, job.upperjf + 1))
    layer_halfwidth = np.sqrt(job.radius**2 - (zf - job.zhub) ** 2)
    ky = k * np.sin(job.beta)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", r"invalid value encountered in divide")
        fac = np.where(ky == 0, layer_halfwidth, np.sin(ky * layer_halfwidth) / ky)

    area_err_fac = (
        job.radius**2 * np.pi / np.sum(np.sqrt(job.radius**2 - (zf - job.zhub) ** 2) * zf * (np.exp(ds) - np.exp(-ds)))
    )
    weights = fac[1:-1] * area_err_fac

    # The solution is linear in the forcing, so the weighted sum over layers is done on the forcing
    # and a single up/down sweep is performed.
    n = len(Q)
    f_up = np.zeros((n, 3), dtype=np.complex128)
    f_down = np.zeros((n, 3), dtype=np.complex128)
    _accumulate_forcing(
        f_up,
        f_down,
        db_const,
        db_lin,
        ijf0_l,
        ijf1_l,
        ijf2_l,
        weights,
        phi_const_low_lst,
        phi_lin_low_lst,
        phi_const_high_lst,
        phi_lin_high_lst,
    )

    full = _sweep(R_upper, R_lower, Q, f_up, f_down, int(ijf0_l.min()) if len(ijf0_l) else n)

    # Only keep the levels corresponding to the original prelut levels, excluding substations.
    new_level = np.concatenate((np.array([True]), levels[1:-1] > levels[:-2]))
    out = np.einsum("jkl,jk->jl", Q, full)[1:][new_level]
    out = np.concatenate([out, np.zeros((zero_pad_levels, 6), dtype=np.complex128)])

    # Extract output levels
    s = slice(job.low_level_out - job.minlevel - 1, job.high_level_out - job.minlevel)
    return cast(npt.NDArray[np.complex128], out[s])


def _sweep(R_upper, R_lower, Q, f_up, f_down, start):  # type: ignore[no-untyped-def]
    """Up/down sweep for given forcing. Returns the 6-component state at every prelut level."""
    n = len(Q)
    ru = np.ascontiguousarray(np.swapaxes(R_upper[:, :3, :3], 1, 2))
    rl = np.ascontiguousarray(np.swapaxes(R_lower[:, :, 3:], 1, 2))  # (n, 3, 6)
    b = np.zeros((n, 3), dtype=np.complex128)
    _up_sweep(ru, f_up, b, start)

    y_tilde = np.concatenate(
        (
            np.conj(Q[-1, :3]),
            np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=np.complex128),
        )
    )
    x_ih = linalg.solve(y_tilde, np.concatenate((b[-1], np.zeros(3, dtype=np.complex128))))

    c = np.einsum("jkl,jl->jk", rl[1:, :, :3], b[1:]) + f_down[:-1]  # forcing + lower part of state, for j-1
    m = rl[1:, :, 3:]
    u = np.zeros((n, 3), dtype=np.complex128)
    # C order, as in a pickled copy: ``@`` rounds differently for the strided view of the preLUT table, which
    # in-process jobs see.
    u[-1] = np.conj(np.ascontiguousarray(Q[-1, 3:])) @ x_ih
    _down_sweep(m, c, u)
    return np.concatenate((b, u), axis=1)


# The forcing accumulation and the sweeps are long loops of tiny 3-element operations: as NumPy they are dominated by
# Python overhead, so they are compiled (releasing the GIL, so the jobs can run in threads). The kernels repeat
# NumPy's arithmetic exactly (in-order complex dot products, complex-by-real products via a zero imaginary part).


def _accumulate_forcing_np(  # type: ignore[no-untyped-def]
    f_up, f_down, db_const, db_lin, ijf0_l, ijf1_l, ijf2_l, weights, pcl_l, pll_l, pch_l, plh_l
) -> None:
    for ijf0, ijf1, ijf2, w, pcl, pll, pch, plh in zip(
        ijf0_l, ijf1_l, ijf2_l, weights, pcl_l, pll_l, pch_l, plh_l, strict=True
    ):
        lo, mid = slice(ijf0, ijf1), slice(ijf1, ijf2)
        f_up[lo] += w * (-db_const[lo, :3] * pcl + db_lin[lo, :3] * pll)
        f_up[mid] += w * (-db_const[mid, :3] * pch + db_lin[mid, :3] * plh)
        f_down[lo] += w * (db_const[lo, 3:] * pcl - db_lin[lo, 3:] * pll)
        f_down[mid] += w * (db_const[mid, 3:] * pch - db_lin[mid, 3:] * plh)


def _up_sweep_np(ru, f_up, b, start) -> None:  # type: ignore[no-untyped-def]
    for j in range(start, len(b) - 1):
        b[j + 1] = ru[j] @ (b[j] + f_up[j])


def _down_sweep_np(m, c, u) -> None:  # type: ignore[no-untyped-def]
    for j in range(len(u) - 1, 0, -1):
        u[j - 1] = m[j - 1] @ u[j] + c[j - 1]


@njit(cache=True, nogil=True)
def _times_real(a, r):  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
    """``a * r`` as NumPy computes it: ``r`` promoted to a complex number with zero imaginary part."""
    return complex(a.real * r - a.imag * 0.0, a.real * 0.0 + a.imag * r)


@njit(cache=True, nogil=True)
def _forcing_rows(f, d_c, d_l, start, stop, col, sign, w, pc, pl):  # type: ignore[no-untyped-def]  # pragma: no cover
    for i in range(start, stop):
        for k in range(3):
            if sign < 0:  # w * (-d_c * pc + d_l * pl)
                v = _times_real(-d_c[i, col + k], pc) + _times_real(d_l[i, col + k], pl)
            else:  # w * (d_c * pc - d_l * pl)
                v = _times_real(d_c[i, col + k], pc) - _times_real(d_l[i, col + k], pl)
            f[i, k] = f[i, k] + _times_real(v, w)


@njit(cache=True, nogil=True)
def _accumulate_forcing_nb(  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
    f_up, f_down, db_const, db_lin, ijf0_l, ijf1_l, ijf2_l, weights, pcl_l, pll_l, pch_l, plh_l
) -> None:
    for m in range(len(ijf0_l)):
        w = weights[m]
        _forcing_rows(f_up, db_const, db_lin, ijf0_l[m], ijf1_l[m], 0, -1, w, pcl_l[m], pll_l[m])
        _forcing_rows(f_up, db_const, db_lin, ijf1_l[m], ijf2_l[m], 0, -1, w, pch_l[m], plh_l[m])
        _forcing_rows(f_down, db_const, db_lin, ijf0_l[m], ijf1_l[m], 3, 1, w, pcl_l[m], pll_l[m])
        _forcing_rows(f_down, db_const, db_lin, ijf1_l[m], ijf2_l[m], 3, 1, w, pch_l[m], plh_l[m])


@njit(cache=True, nogil=True)
def _up_sweep_nb(ru, f_up, b, start) -> None:  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
    v = np.empty(3, dtype=np.complex128)
    for j in range(start, len(b) - 1):
        for k in range(3):
            v[k] = b[j, k] + f_up[j, k]
        for i in range(3):
            s = 0j
            for k in range(3):
                s += ru[j, i, k] * v[k]
            b[j + 1, i] = s  # in-order sum, as NumPy's ``@`` for this size


@njit(cache=True, nogil=True)
def _down_sweep_nb(m, c, u) -> None:  # type: ignore[no-untyped-def]  # pragma: no cover - compiled
    for j in range(len(u) - 1, 0, -1):
        for i in range(3):
            s = 0j
            for k in range(3):
                s += m[j - 1, i, k] * u[j, k]
            u[j - 1, i] = s + c[j - 1, i]


def _kernels_match_numpy() -> bool:
    """True if the compiled kernels reproduce the NumPy code bit for bit on random data with many magnitudes and
    signed zeros (``@`` may use a BLAS with a different operation order on other builds)."""
    rng = np.random.default_rng(0)

    def rand(*shape: int) -> npt.NDArray[np.complex128]:
        a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 10.0 ** rng.uniform(-6, 6, shape)
        mask = rng.random(shape)
        a.real[mask < 0.1] = 0.0
        a.imag[mask > 0.9] = -0.0
        return np.asarray(a, dtype=np.complex128)

    n = 40
    for _ in range(20):
        R, f, c = rand(n, 3, 6), rand(n, 3), rand(n - 1, 3)
        ru = np.ascontiguousarray(R[:, :, :3])
        b0, u0, f_up0, f_down0 = rand(n, 3), rand(n, 3), rand(n, 3), rand(n, 3)
        b0[6:] = 0
        u0[:-1] = 0
        d_c, d_l = rand(n, 6), rand(n, 6)
        ijf = np.sort(rng.integers(0, n, (3, 12)), axis=0)
        w, pcl, pll, pch, plh = (rng.standard_normal(12) * 10.0 ** rng.uniform(-6, 6, 12) for _ in range(5))
        outs = []
        for up, down in ((_up_sweep_np, _down_sweep_np), (_up_sweep_nb, _down_sweep_nb)):
            b, u = b0.copy(), u0.copy()
            up(ru, f, b, 5)
            down(R[1:, :, 3:], c, u)
            outs.append(b.tobytes() + u.tobytes())
        for acc in (_accumulate_forcing_np, _accumulate_forcing_nb):
            f_up, f_down = f_up0.copy(), f_down0.copy()
            acc(f_up, f_down, d_c, d_l, ijf[0], ijf[1], ijf[2], w, pcl, pll, pch, plh)
            outs.append(f_up.tobytes() + f_down.tobytes())
        if outs[0] != outs[1] or outs[2] != outs[3]:
            return False
    return True


_accumulate_forcing, _up_sweep, _down_sweep = (
    (_accumulate_forcing_nb, _up_sweep_nb, _down_sweep_nb)
    if _kernels_match_numpy()
    else (_accumulate_forcing_np, _up_sweep_np, _down_sweep_np)
)


@jit  # (f'complex128[:,:]({c3}{c3}{c3}{i1}{c2}{c2}{d}{d}{d}{d}{i}{i}{i})')
def solve_layer(  # type: ignore[no-untyped-def]
    R_upper,
    R_lower,
    Q,
    levels,  # excluding substations
    db_const,
    db_lin,
    phi_const_low,  # 4 Triangular/Chapeau function weights
    phi_lin_low,
    phi_const_high,
    phi_lin_high,
    icl_m1,  # icl-1
    icl,  # i_current level
    icl_p1,  # icl+1
):
    """
    Loops over a single (beta, kz0, layer, forcing direction) combination.
    Returns: X(z) at all output levels for that specific layer and forcing. See eq. (3) and (12) in Numerical Solution.
    NB: Forcing contributions are calculated using all (sub-)stations, but the output only includes the main stations.
    """
    icl_m1, icl, icl_p1 = icl_m1.item(), icl.item(), icl_p1.item()  # icl-1/icl/icl+1 as int
    b_lower_3 = [np.zeros(3, dtype=np.complex128)] * (icl_m1 + 1)  # minlevel(z=1m) to cl-1 -- b for lower BC

    forcing_increments_up = np.concatenate(
        (
            (-db_const[icl_m1:icl, :3] * phi_const_low + db_lin[icl_m1:icl, :3] * phi_lin_low),  # cl-1 to cl
            (-db_const[icl:icl_p1, :3] * phi_const_high + db_lin[icl:icl_p1, :3] * phi_lin_high),  # cl to cl+1
        )
    )

    # cl-1 to cl+1
    R_upper_slice = R_upper[icl_m1:icl_p1, :3, :3]
    if len(forcing_increments_up) != len(R_upper_slice):
        raise ValueError("Length mismatch between forcing_increments_up and R_upper slice")
    for F_increment, RR in zip(forcing_increments_up, R_upper_slice):  # noqa: B905
        b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1] + F_increment))

    # cl+1 to max level
    for RR in R_upper[icl_p1:-1, :3, :3]:
        b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1]))

    # at max level
    Y_tilde = np.concatenate(
        (
            np.conj(Q[-1, :3]),
            np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=np.complex128),
        )
    )
    b_tilde = np.concatenate((b_lower_3[-1][:3], np.asarray([0 + 0j, 0, 0])))
    x_ih = linalg.solve(Y_tilde, b_tilde)
    Y_ih_last_3 = np.conj(Q[-1, 3:])
    b_full_6 = [np.concatenate((b_lower_3.pop(), Y_ih_last_3 @ x_ih))]

    # max level to cl+1
    for RL in R_lower[-1:icl_p1:-1, :, 3:]:
        b_full_6.append(np.concatenate((b_lower_3.pop(), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1]))))

    icl_m1, icl, icl_p1 = icl_m1 - 1, icl - 1, icl_p1 - 1
    forcing_increments_down = np.concatenate(
        (
            (+db_const[icl_p1:icl:-1, 3:] * phi_const_high - db_lin[icl_p1:icl:-1, 3:] * phi_lin_high),  # cl+1 to cl
            (+db_const[icl:icl_m1:-1, 3:] * phi_const_low - db_lin[icl:icl_m1:-1, 3:] * phi_lin_low),
        )
    )

    # cl+1 to cl-1
    R_lower_slice = R_lower[icl_p1 + 1 : icl_m1 + 1 : -1, :, 3:]
    if len(forcing_increments_down) != len(R_lower_slice):
        raise ValueError("Length mismatch between forcing_increments_down and R_lower slice")
    for F_increment, RL in zip(forcing_increments_down, R_lower_slice):  # noqa: B905
        b_full_6.append(
            np.concatenate((b_lower_3.pop(), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1]) + F_increment))
        )

    # cl-1 to min_level
    for RL in R_lower[icl_m1 + 1 : 0 : -1, :, 3:]:
        b_full_6.append(
            np.concatenate((np.zeros(3, dtype=np.complex128), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1])))
        )

    # Only keep the levels corresponding to the original prelut levels, excluding substations.
    new_level = np.concatenate((np.array([True]), levels[1:-1] > levels[:-2]))

    filtered_q_levels = [v for v, nl in zip(Q[1:], new_level) if nl]  # noqa: B905
    filtered_b_levels = [v for v, nl in zip(b_full_6[::-1][1:], new_level) if nl]  # noqa: B905
    if len(filtered_q_levels) != len(filtered_b_levels):
        raise ValueError("Length mismatch between filtered Q and b levels")

    paired_levels = zip(filtered_q_levels, filtered_b_levels)  # noqa: B905

    return [(np.ascontiguousarray(Q_level.T) @ b_level) for Q_level, b_level in paired_levels]
