"""
trafalgar.py
============
Inverse Fourier transform of the spectral wake solution into physical space.

This module implements the Fast Fourier Integral Transform (FFIT) described in
``docs/theory/FFIT.rst``.  It is the third and final stage of the PyFuga pipeline:

  preluts_generator  →  flut  →  **trafalgar**

**Input**
    ``fourier_luts`` (:class:`xarray.Dataset`) — Fourier-space wake fields produced
    by :mod:`pyfuga.flut`.  Variables: ``UL``, ``VL``, ``WL``, ``PL``, ``UT``, ``VT``,
    ``WT``, ``PT`` (L = longitudinal forcing, T = transverse forcing), each with
    dimensions ``(beta, kz0, level)``.

**Output**
    :class:`xarray.Dataset` with the same variable names but now in physical space,
    dimensions ``(z, x, y)``.

**Method**
    The 2D Fourier integral (eq. ``eq:fourier_integral`` in FFIT.rst) is approximated
    over a finite wavenumber domain partitioned into rectangular tiles covering an
    L-shaped region in :math:`(k_x, k_y)` space (see Section "Wavenumber Tiling" in
    FFIT.rst).  Each tile is evaluated with the Bluestein/FFIT algorithm
    (eq. ``eq:ffit_main``) first along :math:`x`, then along :math:`y`.
    Tile contributions are summed in physical space.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import xarray as xr
from numba import njit
from numpy import newaxis as na
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import sparse
from scipy.interpolate import BSpline, RectBivariateSpline, make_interp_spline
from tqdm import tqdm

from pyfuga.constants import KAPPA
from pyfuga.utils import fma as _fma
from pyfuga.utils import normalise_n_cpu_workers, threading_map

M_PI = 3.141592653589793

# +1 → take real part of FFIT output; -1 → take imaginary part.
_SIGNS = {"UL": 1, "VL": -1, "WL": 1, "PL": 1, "UT": -1, "VT": 1, "WT": -1, "PT": -1}


class Trafalgar:
    """
    Convert Fourier-space wake fields to physical-space wake fields via FFIT.

    See ``docs/theory/FFIT.rst`` for the mathematical background.

    Parameters
    ----------
    fourier_luts : xarray.Dataset
        Spectral wake solution from :mod:`pyfuga.flut`.
        Must contain variables ``UL``, ``VL``, ... and scalar coordinates
        ``diameter``, ``hubheight``, ``z0``.
    nx : int
        Number of output grid points in the streamwise direction :math:`x`.
    ny : int
        Number of output grid points in the cross-stream direction :math:`y`
        (internally halved; the wake is symmetric about :math:`y=0`).
    dx : float
        Streamwise grid spacing (m).  Determines spatial resolution in :math:`x`
        and, together with ``NNNs``, the spectral bandwidth per tile.
    dy : float
        Cross-stream grid spacing (m).
    sigmax : float
        Streamwise Gaussian smoothing width (m) applied in spectral space to
        represent the finite rotor extent.  Enters as
        :math:`e^{-\\frac{1}{2}(\\sigma_x k_x)^2}`.
    sigmay : float
        Cross-stream Gaussian smoothing width (m), analogous to ``sigmax``.
    verbose : bool
        Show progress bars.
    """

    def __init__(
        self,
        fourier_luts: xr.Dataset,
        nx: int = 2048,
        ny: int = 512,
        dx: float = 20,
        dy: float = 5,
        sigmax: float = 10,
        sigmay: float = 10,
        verbose: bool = True,
    ) -> None:

        self.fourier_luts = fourier_luts
        self.nx = nx
        self.ny = ny // 2  # exploit y-symmetry of the wake
        self.dx = dx
        self.dy = dy
        self.sigmax = sigmax
        self.sigmay = sigmay

        # Output grid: rotor at x=0, wake extends downstream (positive x)
        self.x_lst = np.arange(-self.nx // 4, self.nx * 3 / 4) * self.dx
        self.y_lst = np.arange(self.ny) * self.dy
        self.verbose = verbose

    def make_luts(
        self,
        interpolation_order: int = 3,
        NNNs: Sequence[int] = (2**16, 2**8, 2**2, 2**2),
        legacy: bool = False,
        n_cpu: int | None = None,
    ) -> xr.Dataset:
        """
        Evaluate the 2D FFIT over all wavenumber tiles and return the wake LUTs.

        Parameters
        ----------
        interpolation_order : int
            Spline order used to interpolate the Fourier LUTs onto the
            :math:`(k_x, k_y)` grid of each tile.
        NNNs : sequence of int
            Spectral density factors for successive tile levels.  Ignored when
            ``legacy=True`` (forced to ``(4,)``).
        legacy : bool
            If ``True``, reproduce the behaviour of the original single-tile
            implementation: NNN=4, single tile starting at k=0, no arms,
            sign applied to spectral data before interpolation, and
            DC component zeroed.
        n_cpu : int or None, optional
            Number of parallel worker threads for evaluating FFIT over
            wavenumber tiles. ``None`` (default) uses all available
            CPUs (at least ``1``). Values greater than ``1`` use a bounded thread pool.
            In :func:`pyfuga.get_luts`, default policy also resolves to all CPUs
            unless overridden via ``n_cpu``/``n_cpu_stage``.

        Returns
        -------
        xarray.Dataset
            Wake fields ``UL``, ``VL``, ... with dimensions ``(z, x, y)`` and
            coordinates ``x``, ``y``, ``z``.
        """
        workers = normalise_n_cpu_workers(n_cpu)
        kz0_lst = np.asarray(self.fourier_luts.kz0.values)
        self.ny0 = self.ny // 2

        if legacy:
            # Legacy mode: single tile at k=0, NNN=4, no arms
            kmin = 0.0
            NNNs = (2**2,)
        else:
            kmin = float(np.min(kz0_lst)) / float(self.fourier_luts.z0.item())

        kmax = float(np.max(kz0_lst)) / float(self.fourier_luts.z0.item())

        # Build tile list (L-shaped coverage of wavenumber space)
        tiles = doKvector_tiles(self.nx, self.ny, self.dx, self.dy, NNNs, kmin, kmax, use_arms=(not legacy))

        args = (
            (self.fourier_luts,)
            + t
            + (
                interpolation_order,
                self.nx,
                self.ny,
                self.dx,
                self.dy,
                self.sigmax,
                self.sigmay,
                # Per-tile bars only when serial; worker bars would interleave with the outer one
                self.verbose and workers == 1,
                legacy,
            )
            for t in tiles
        )

        # The y-transform is linear and depends only on the tile's y-axis, so x-transformed tiles
        # sharing a y-axis are summed (in tile order) and y-transformed once per group.
        # Parallelism is per wavenumber tile for the x-transform and per (variable, level) for the
        # y-transform; groups are added in a fixed order, so results do not depend on ``n_cpu``.
        groups: dict[tuple[Any, float], dict[str, NDArray[np.float64]]] = {}
        with threading_map(workers) as map_func:
            for tile, fields_x in zip(
                tiles,
                tqdm(
                    map_func(_x_transform_tile, args),
                    total=len(tiles),
                    disable=(not self.verbose),
                    desc="LUTS: adding meshes",
                ),
                strict=True,
            ):
                group = groups.setdefault((tile[2], tile[3]), fields_x)
                if group is not fields_x:
                    for v, field_x in fields_x.items():
                        group[v] += field_x

            if not groups:  # pragma: no cover
                raise ValueError("No wavenumber tiles were generated for the requested parameters")

            lut_vars = list(next(iter(groups.values())))
            n_level = len(self.fourier_luts.level)
            y_axes = [(NNNy, doKvectorFIT_double(self.ny, self.dy, k0=k0y, NNN=NNNy)) for NNNy, k0y in groups]

            def y_transform(item: tuple[str, int]) -> NDArray[np.float64]:
                var, level = item
                out = np.zeros((self.nx, self.ny))
                for (NNNy, ky), group in zip(y_axes, groups.values(), strict=True):
                    out += _fitfit_y(group[var][level], _SIGNS[var], ky, self.ny, self.dy, NNNy)
                return out

            items = [(v, i) for v in lut_vars for i in range(n_level)]
            fields = list(
                tqdm(
                    map_func(y_transform, items),
                    total=len(items),
                    disable=(not self.verbose),
                    desc="LUTS: y-transform",
                )
            )

        luts_dict = {
            v: (("z", "x", "y"), np.stack(fields[j * n_level : (j + 1) * n_level])) for j, v in enumerate(lut_vars)
        }
        fluts = self.fourier_luts
        return xr.Dataset(
            {**luts_dict, **{k: fluts[k] for k in ["diameter", "hubheight", "z0"]}},
            coords={"x": self.x_lst, "y": self.y_lst, "z": np.atleast_1d(fluts.z)},
            attrs=fluts.attrs,
        )


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------


class _SplineOperator(NamedTuple):
    """Factored B-spline evaluation operator (see :func:`_spline_operator`).

    Row ``p`` of the design matrix ``W`` has the non-zeros ``bx[p, a] * by[p, b]`` at column
    ``(ox[p] + a) * ny + oy[p] + b``, stored in the order ``a * (k + 1) + b``.
    """

    bx: NDArray[np.float64]
    ox: NDArray[np.int64]
    by: NDArray[np.float64]
    oy: NDArray[np.int64]
    nx: int
    ny: int
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]

    def matrix(self) -> sparse.csr_matrix:
        """The design matrix ``W`` as CSR."""
        n, m = self.bx.shape
        data = (self.bx.reshape(n, m, 1) * self.by.reshape(n, 1, m)).reshape(n, m * m)
        ix = self.ox[:, na] + np.arange(m)
        iy = self.oy[:, na] + np.arange(m)
        cols = (ix.reshape(n, m, 1) * self.ny + iy.reshape(n, 1, m)).reshape(n, m * m)
        indptr = np.arange(0, n * m * m + 1, m * m)
        return sparse.csr_matrix((data.ravel(), cols.ravel(), indptr), shape=(n, self.nx * self.ny))


@njit(cache=True, nogil=True)
def _bspline_basis(
    x: NDArray[np.float64], t: NDArray[np.float64], k: int, vals: NDArray[np.float64], offsets: NDArray[np.int64]
) -> None:  # pragma: no cover - compiled
    """Non-zero B-spline values at in-range ``x``, as SciPy's ``BSpline.design_matrix`` (FITPACK ``data_matrix``).

    Same interval search and de Boor recursion as SciPy's C++, so the values match bit for bit.
    """
    nt = t.shape[0] - k - 1
    h = np.empty(k + 1)
    hh = np.empty(k + 1)
    ind = k
    for i in range(x.shape[0]):
        xval = x[i]
        ell = ind if k < ind < nt else k
        while xval < t[ell] and ell != k:
            ell -= 1
        ell += 1
        while xval >= t[ell] and ell != nt:
            ell += 1
        ind = ell - 1
        offsets[i] = ind - k
        h[0] = 1.0
        for j in range(1, k + 1):
            hh[:j] = h[:j]
            h[0] = 0.0
            for n in range(1, j + 1):
                xb = t[ind + n]
                xa = t[ind + n - j]
                if xb == xa:
                    h[n] = 0.0
                    continue
                w = hh[n - 1] / (xb - xa)
                h[n - 1] += w * (xb - xval)
                h[n] = w * (xval - xa)
        vals[i, :] = h


def _basis(q: NDArray[np.float64], t: NDArray[np.float64], k: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if not np.isfinite(q).all():
        design = BSpline.design_matrix(q, t, k)  # raises like before
        return design.data.reshape(-1, k + 1), design.indices[:: k + 1].astype(np.int64)
    vals = np.empty((q.size, k + 1))
    offsets = np.empty(q.size, dtype=np.int64)
    _bspline_basis(
        np.ascontiguousarray(q, dtype=np.float64), np.ascontiguousarray(t, dtype=np.float64), k, vals, offsets
    )
    return vals, offsets


def _spline_operator(
    x: NDArray[np.float64], y: NDArray[np.float64], px: NDArray[np.float64], py: NDArray[np.float64], order: int
) -> _SplineOperator | None:
    """Precompute a reusable evaluation operator for ``RectBivariateSpline(x, y, z, kx=order, ky=order).ev``.

    The interpolating spline is linear in ``z``, so for fixed ``(x, y)`` and evaluation points the
    result is ``W @ (Cx @ z @ Cy.T)`` with ``W`` a sparse B-spline design matrix, kept factored as the
    per-axis basis values. Returns ``None`` for even orders, whose knot placement differs from FITPACK's.
    """
    if order % 2 == 0 or len(x) <= order or len(y) <= order:
        return None
    k = order
    sx = make_interp_spline(x, np.eye(len(x)), k=k)
    sy = make_interp_spline(y, np.eye(len(y)), k=k)
    # FITPACK clamps evaluation points to the data range
    qx = np.clip(px.ravel(), x[0], x[-1])
    qy = np.clip(py.ravel(), y[0], y[-1])
    bx, ox = _basis(qx, sx.t, k)
    by, oy = _basis(qy, sy.t, k)
    return _SplineOperator(bx, ox, by, oy, len(x), len(y), sx.c, sy.c)


def _spline_coefficients(operator: _SplineOperator, flut: NDArray[Any]) -> NDArray[Any]:
    """Spline coefficients of every level of ``flut`` (shape ``(nx, ny, nlevel)``), one column per level."""
    coef = np.einsum("ab,bcl,dc->adl", operator.cx, flut, operator.cy, optimize=True)
    return cast("NDArray[Any]", coef.reshape(-1, flut.shape[2]))


def _as_components(coef: NDArray[Any]) -> NDArray[np.float64] | None:
    """Finite complex coefficients as interleaved real/imaginary columns (real weights act on them
    independently); ``None`` if they need the generic path."""
    if coef.dtype != np.complex128 or not np.isfinite(coef).all():
        return None
    return np.ascontiguousarray(coef).view(np.float64)


def _interpolate_levels(operator: _SplineOperator, flut: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Evaluate the spline of every level of ``flut`` (shape ``(nx, ny, nlevel)``) at the operator's points."""
    w = operator.matrix()
    coef = _spline_coefficients(operator, flut)
    components = _as_components(coef)
    if components is not None:
        return cast("NDArray[np.complex128]", (w @ components).view(np.complex128))
    return cast("NDArray[np.complex128]", w @ coef)


@njit(cache=True, nogil=True)
def _eval_scaled_spline(
    bx: NDArray[np.float64],
    ox: NDArray[np.int64],
    by: NDArray[np.float64],
    oy: NDArray[np.int64],
    ny: int,
    components: NDArray[np.float64],
    scale: NDArray[np.float64],
    start: int,
    out: NDArray[np.float64],
) -> None:  # pragma: no cover - compiled
    """Fused ``(W @ components)`` and complex scaling for the columns ``start:start + width`` of ``scale``;
    ``out`` is the real view ``(nlevel, nrow, width, 2)``.

    Each weight is formed as in :meth:`_SplineOperator.matrix`. Each output is summed over the row's
    non-zeros in CSR order starting from zero, and scaled as ``(re + i im) * (s + 0i)``, matching SciPy's
    sparse product followed by NumPy's complex multiply bit for bit.
    """
    m = bx.shape[1]
    zero = 0.0
    width = out.shape[2]
    nq = components.shape[1]
    # One row of accumulators, so each level is then stored as a contiguous run
    rowacc = np.empty((width, nq))
    ws = np.empty(m)
    for r in range(out.shape[1]):
        for c in range(width):
            p = r * scale.shape[1] + start + c
            acc = rowacc[c]
            acc[:] = zero
            for a in range(m):
                for b in range(m):
                    ws[b] = bx[p, a] * by[p, b]
                base = (ox[p] + a) * ny + oy[p]
                if m == 4:  # cubic: the four terms of a row in one pass, same summation order
                    w0, w1, w2, w3 = ws[0], ws[1], ws[2], ws[3]
                    r0, r1, r2, r3 = components[base], components[base + 1], components[base + 2], components[base + 3]
                    for q in range(nq):
                        acc[q] = (((acc[q] + w0 * r0[q]) + w1 * r1[q]) + w2 * r2[q]) + w3 * r3[q]
                else:
                    for b in range(m):
                        w = ws[b]
                        row = components[base + b]
                        for q in range(nq):
                            acc[q] += w * row[q]
        for level in range(out.shape[0]):
            for c in range(width):
                s = scale[r, start + c]
                re = rowacc[c, 2 * level]
                im = rowacc[c, 2 * level + 1]
                out[level, r, c, 0] = re * s - im * zero
                out[level, r, c, 1] = re * zero + im * s


def _spline_components(operator: _SplineOperator, flut: NDArray[np.complex128]) -> NDArray[np.float64] | None:
    """Spline coefficients of every level of ``flut`` as interleaved real/imaginary columns; ``None`` if they
    need the generic (non-finite) path."""
    return _as_components(_spline_coefficients(operator, flut))


def _scaled_block(
    operator: _SplineOperator,
    components: NDArray[np.float64],
    scale: NDArray[np.float64],
    start: int,
    out: NDArray[np.complex128],
) -> None:
    """Interpolate and scale the columns ``start:start + out.shape[2]`` of the tile into ``out`` (level-major)."""
    _eval_scaled_spline(
        operator.bx,
        operator.ox,
        operator.by,
        operator.oy,
        operator.ny,
        components,
        scale,
        start,
        out.view(np.float64).reshape(*out.shape, 2),
    )


def _interpolate_scaled(
    operator: _SplineOperator, flut: NDArray[np.complex128], scale: NDArray[np.float64]
) -> NDArray[np.complex128] | None:
    """Return ``_interpolate_levels(operator, flut)`` times ``scale`` (2D), level-major with ``scale``'s shape
    prepended by the level count; ``None`` if the coefficients need the generic (non-finite) path."""
    components = _spline_components(operator, flut)
    if components is None:
        return None
    out = np.empty((flut.shape[2], *scale.shape), dtype=np.complex128)
    _scaled_block(operator, components, scale, 0, out)
    return out


def _x_transform_fused(
    operator: _SplineOperator,
    components: list[NDArray[np.float64]],
    scale: NDArray[np.float64],
    signs: list[int],
    kx: NDArray[np.float64],
    nx: int,
    dx: float,
    NNNx: float,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Interpolate, scale and x-transform all fields of a tile one column block at a time.

    ``components`` holds the spline coefficients of each variable (as :func:`_spline_components`) and
    ``signs`` the sign of each variable. Returns ``(n_variables * n_levels, nx, ny)``, bitwise equal to
    :func:`_interpolate_scaled` followed by :func:`_fitfit_x` per field, without materialising the
    interpolated tile.
    """
    phix, psix, hhatx = _fitfit_axis(nx, dx, NNNx, float(kx[0]), -nx / 4.0)
    n_level = components[0].shape[1] // 2
    n_fields = len(components) * n_level
    ny = scale.shape[1]
    field_x = np.empty((n_fields, nx, ny))
    for start in tqdm(range(0, ny, _FITFIT_BLOCK_X), desc="Trafalgar:", disable=(not verbose)):
        width = min(_FITFIT_BLOCK_X, ny - start)
        block = np.empty((n_level, nx, width), dtype=np.complex128)
        for v, comp in enumerate(components):
            _scaled_block(operator, comp, scale, start, block)
            for i in range(n_level):
                j = v * n_level + i
                _fitfit_x_block(block[i], 0, width, signs[v], phix, psix, hhatx, field_x[j, :, start : start + width])
    return field_x


def _x_transform_tile(args: tuple[Any, ...]) -> dict[str, NDArray[np.float64]]:
    """
    Evaluate the x-direction FFIT for a single wavenumber tile.

    A tile is defined by a rectangular region in :math:`(k_x, k_y)` space,
    parameterised by ``(NNNx, k0x, NNNy, k0y)``.  The spectral fields are
    interpolated from the Fourier LUTs onto the tile's :math:`k`-grid, weighted
    by the rotor Gaussian force spectrum (``force``) and the normalisation
    factor (``fuzz``), and then transformed along :math:`x` via ``_fitfit_x``.
    The y-direction transform is applied later, once per group of tiles sharing
    a y-axis (see :meth:`Trafalgar.make_luts`).

    Parameters
    ----------
    args : tuple
        Packed arguments from :meth:`Trafalgar.make_luts`.

    Returns
    -------
    dict of str to numpy.ndarray
        Per variable, the x-transformed real fields with shape ``(n_levels, nx, ny)``.
    """
    (
        fluts,
        NNNx,
        k0x,
        NNNy,
        k0y,
        interpolation_order,
        nx,
        ny,
        dx,
        dy,
        sigmax,
        sigmay,
        verbose,
        legacy,
    ) = args

    beta_lst = fluts.beta.values
    kz0_lst = fluts.kz0.values

    _, zhub, z0 = (fluts.diameter.item(), fluts.hubheight.item(), fluts.z0.item())

    # Wavenumber grid for this tile
    kx = doKvectorFIT_double(nx, dx, k0=k0x, NNN=NNNx)
    ky = doKvectorFIT_double(ny, dy, k0=k0y, NNN=NNNy)
    bx = M_PI / (dx * nx) / NNNx
    by = M_PI / (dy * ny) / NNNy

    # Normalisation
    U = np.log(zhub / z0) / KAPPA
    fuzz = U * bx * by * nx * ny / (M_PI * M_PI)

    # Rotor Gaussian force spectrum
    force = np.exp(-0.5 * (sigmay * ky[na]) ** 2) * np.exp(-0.5 * (sigmax * kx[:, na]) ** 2)

    # Polar coordinates in k-space for interpolating the LUTs
    klen = np.sqrt(kx[:, na] ** 2 + ky[na] ** 2)
    angleTab = np.arctan2(ky[na], kx[:, na])
    kz0Tab = np.log(z0 * klen)

    operator = _spline_operator(beta_lst, np.log(kz0_lst), angleTab, kz0Tab, interpolation_order)
    scale = fuzz * force

    fields_x = {}
    tile_vars = [var for var in _SIGNS if var in fluts]
    if operator is not None and not legacy:
        maybe = [_spline_components(operator, fluts[var].values) for var in tile_vars]
        components = [c for c in maybe if c is not None]
        if len(components) == len(maybe):
            n_level = len(fluts.level)
            signs = [_SIGNS[var] for var in tile_vars]
            stacked = _x_transform_fused(operator, components, scale, signs, kx, nx, dx, NNNx, verbose)
            return {var: stacked[j * n_level : (j + 1) * n_level] for j, var in enumerate(tile_vars)}

    for var, sign in tqdm(_SIGNS.items(), desc="Trafalgar:", disable=(not verbose)):
        if var not in fluts:
            continue

        flut = fluts[var].values  # shape: (n_beta, n_kz0, n_levels)

        # Difference #2: legacy multiplies spectral data by sign before interpolation
        if legacy:
            flut = flut * sign

        # Interpolation and the force/normalisation scaling, fused where possible
        scaled = None if operator is None else _interpolate_scaled(operator, flut, scale)
        if operator is not None and scaled is None:
            fields = _interpolate_levels(operator, flut).reshape(*klen.shape, -1)

        field_x = np.empty((len(fluts.level), nx, ny))
        for i in tqdm(range(len(fluts.level)), disable=(not verbose), leave=False):
            if scaled is not None:
                field = scaled[i]
            elif operator is not None:
                field = fields[:, :, i]
            else:
                re, im = flut[:, :, i].real, flut[:, :, i].imag
                field = RectBivariateSpline(
                    beta_lst, np.log(kz0_lst), re, kx=interpolation_order, ky=interpolation_order
                ).ev(angleTab, kz0Tab) + 1j * RectBivariateSpline(
                    beta_lst, np.log(kz0_lst), im, kx=interpolation_order, ky=interpolation_order
                ).ev(
                    angleTab, kz0Tab
                )

            if scaled is None:
                # Apply force spectrum and normalisation
                field = field * scale

            # Difference #3: legacy zeroes the DC component
            if legacy:
                field[0, 0] = 0.0

            # x-direction FFIT; the y-direction is applied per tile group
            field_x[i] = _fitfit_x(field, sign, kx, nx, dx, NNNx)

        fields_x[var] = field_x

    return fields_x


# ---------------------------------------------------------------------------
# Wavenumber grid helpers
# ---------------------------------------------------------------------------


def doKvectorFIT_double(
    n: int,
    delta: float,
    k0: float | None = None,
    NNN: int = 2**3,
) -> NDArray[np.float64]:
    """
    Return the wavenumber midpoints for one tile axis.

    The spectral spacing is :math:`b = \\pi / (n\\,\\Delta x) / NNN` and the
    midpoints are :math:`k_q = k_0 + b(q + 1/2)`, :math:`q = 0,\\ldots,n-1`
    (cf. FFIT.rst, Section "The FFIT Algorithm").

    Parameters
    ----------
    n : int
        Number of points.
    delta : float
        Spatial grid spacing (m).
    k0 : float or None
        Left edge of the tile in wavenumber space.  ``None`` → 0.
    NNN : int
        Spectral density factor; larger values give a narrower tile with finer
        :math:`k`-spacing.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    c = np.pi / (n * delta) / NNN
    k0_ = 0.0 if k0 is None else float(k0)
    return cast(NDArray[np.float64], k0_ + c * (np.arange(int(n)) + 0.5))


def doKvector_tiles(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    NNNs: Sequence[int],
    kmin: float,
    kmax: float,
    use_arms: bool = True,
) -> list[tuple[float, float, float, float]]:
    """
    Build the list of wavenumber tiles covering the L-shaped domain.

    Parameters
    ----------
    nx, ny : int
        Grid size (same as :class:`Trafalgar`).
    dx, dy : float
        Grid spacings (m).
    NNNs : sequence of int
        Spectral density levels, from finest to coarsest.
    kmin, kmax : float
        Physical wavenumber range (rad/m).
    use_arms : bool
        If ``True``, include left and bottom arm tiles.

    Returns
    -------
    list of tuple
        Each tuple is ``(NNNx, k0x, NNNy, k0y)`` defining one tile.
    """
    kmin = float(kmin)
    kmax = float(kmax)

    nnn_levels = np.sort(np.array(NNNs, dtype=int))[::-1]  # finest first
    if nnn_levels.size == 0:
        return []

    # Special case: kmin=0 (legacy mode) — skip the kmax>kmin>0 check
    if kmin == 0.0:
        # Single tile starting at k=0 for each NNN level
        legacy_tiles: list[tuple[float, float, float, float]] = []
        for NNN in nnn_levels:
            legacy_tiles.append((int(NNN), 0.0, int(NNN), 0.0))
        return legacy_tiles

    if not (kmax > kmin > 0):
        raise ValueError("Require kmax > kmin > 0")

    def tile_span(n: int, delta: float, NNN: int) -> tuple[float, float]:
        """Return (total span, spacing) for one axis."""
        c = np.pi / (n * delta) / NNN
        return n * c, c

    # Build 1D level lists starting at kmin
    def build_levels(n: int, delta: float) -> list[tuple[int, float]]:
        levels: list[tuple[int, float]] = []
        k0 = kmin
        for NNN in nnn_levels:
            span, _ = tile_span(n, delta, NNN)
            if k0 + span > kmax:
                break
            levels.append((int(NNN), float(k0)))
            k0 += span
        return levels

    levels_x = build_levels(nx, dx)
    levels_y = build_levels(ny, dy)

    # Keep equal number of levels on both axes (diagonal tiling)
    n_levels = min(len(levels_x), len(levels_y))
    levels_x = levels_x[:n_levels]
    levels_y = levels_y[:n_levels]
    if n_levels == 0:
        return []

    tiles: list[tuple[float, float, float, float]] = []

    # A) Main square: all (NNNx, k0x) × (NNNy, k0y) combinations
    for NNNx, k0x in levels_x:
        for NNNy, k0y in levels_y:
            tiles.append((NNNx, k0x, NNNy, k0y))

    if use_arms:
        # NNN for arm tiles: chosen so the tile exactly spans [0, kmin]
        NNNx_arm = np.pi / (dx * kmin)
        NNNy_arm = np.pi / (dy * kmin)

        # B) Left arm: kx in [0, kmin], ky >= kmin
        for NNNy, k0y in levels_y:
            tiles.append((NNNx_arm, 0.0, NNNy, k0y))

        # C) Bottom arm: ky in [0, kmin], kx >= kmin
        for NNNx, k0x in levels_x:
            tiles.append((NNNx, k0x, NNNy_arm, 0.0))

    return tiles


# ---------------------------------------------------------------------------
# 2D FFIT core
# ---------------------------------------------------------------------------


_FITFIT_BLOCK_X = 32  # columns per x-transform block
_FITFIT_BLOCK_Y = 128  # rows per y-transform block


@functools.lru_cache(maxsize=64)
def _fitfit_axis(
    n: int, delta: float, NNN: float, k0: float, i0: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Phase factors and chirp kernel for one axis of :func:`FITFIT` (cached; depends only on the grid)."""
    b = np.pi / (delta * n) / NNN
    ab = delta * b
    x0 = ab * i0 / b  # = delta * i0

    i = np.arange(n)
    phi = np.exp(1j * ((k0 + b * i) * x0 + ab / 2 * i**2))
    psi = np.exp(1j * ab * (k0 * i / b + i**2 / 2))

    i_ = np.arange(n + 1)
    aux = np.exp(-1j * ab / 2.0 * i_ * i_)
    hhat = sfft.ifft(np.r_[aux, aux[::-1][1:-1]])
    for arr in (phi, psi, hhat):
        arr.flags.writeable = False
    return cast("tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]", (phi, psi, hhat))


def FITFIT(
    field: NDArray[np.complexfloating],
    sign: int,
    kx: NDArray[np.float64],
    ky: NDArray[np.float64],
    nx: int,
    dx: float,
    ny: int,
    dy: float,
    NNN: tuple[int, int] = (4, 4),
) -> NDArray[np.float64]:
    """
    Apply the 2D Fast Fourier Integral Transform to ``field``.

    Implements eq. ``eq:ffit_main`` (FFIT.rst) separably:
    first a 1D FFIT along :math:`x`, then a 1D FFIT along :math:`y`.

    Parameters
    ----------
    field : numpy.ndarray, shape (nx, ny)
        Spectral field for one variable and one vertical level.
    sign : int
        +1 → output is real part of FFIT; -1 → output is imaginary part.
    kx, ky : numpy.ndarray
        Wavenumber midpoints from ``doKvectorFIT_double``.
    nx, ny : int
        Grid sizes.
    dx, dy : float
        Grid spacings (m).
    NNN : tuple of int
        ``(NNNx, NNNy)`` spectral density factors.

    Returns
    -------
    numpy.ndarray, shape (nx, ny)
        Physical-space field (real).
    """
    NNNx, NNNy = NNN
    return _fitfit_y(_fitfit_x(field, sign, kx, nx, dx, NNNx), sign, ky, ny, dy, NNNy)


# ``roll(ifft(a)[::-1], 1) == fft(a) / N``, so the reversal and roll are folded into the transform
# direction and the ``/ N`` cancels the ``* N`` that follows the second transform.
# The 1D transforms are independent, so they are processed in blocks small enough to stay in cache.


# NumPy's SIMD complex multiply ``a * b`` computes ``re = fma(ar, br, -(ai * bi))`` and
# ``im = fma(ar, bi, ai * br)``; the kernels below repeat that so their results are bitwise identical.


@njit(cache=True, nogil=True)
def _chirp_pad_x(
    phi: NDArray[np.complex128], field: NDArray[np.complexfloating], start: int, buf: NDArray[np.complex128]
) -> None:  # pragma: no cover - compiled
    """``buf = pad(phi[:, na] * field[:, start:start + w], 2n)`` along axis 0, as NumPy computes it."""
    n = field.shape[0]
    for i in range(n):
        ar = phi[i].real
        ai = phi[i].imag
        for c in range(buf.shape[1]):
            b = field[i, start + c]
            buf[i, c] = complex(_fma(ar, b.real, -(ai * b.imag)), _fma(ar, b.imag, ai * b.real))
    for i in range(n, buf.shape[0]):
        for c in range(buf.shape[1]):
            buf[i, c] = 0


@njit(cache=True, nogil=True)
def _chirp_pad_y(
    phi: NDArray[np.complex128], field: NDArray[np.float64], start: int, buf: NDArray[np.complex128]
) -> None:  # pragma: no cover - compiled
    """``buf = pad(phi[na, :] * field[start:start + h], 2n)`` along axis 1, as NumPy computes it (real ``field``
    promoted to complex with a zero imaginary part)."""
    n = field.shape[1]
    zero = 0.0
    for r in range(buf.shape[0]):
        for i in range(n):
            ar = phi[i].real
            ai = phi[i].imag
            b = field[start + r, i]
            buf[r, i] = complex(_fma(ar, b, -(ai * zero)), _fma(ar, zero, ai * b))
        for i in range(n, buf.shape[1]):
            buf[r, i] = 0


@njit(cache=True, nogil=True)
def _chirp_part(
    psi: NDArray[np.complex128],
    tmp: NDArray[np.complexfloating],
    axis: int,
    factor: float,
    imag: bool,
    out: NDArray[np.float64],
) -> None:  # pragma: no cover - compiled
    """``out = factor * (psi * tmp[:n]).real`` (or ``.imag``) with ``psi`` along ``axis`` of ``out``, as NumPy
    computes it."""
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            a = psi[r] if axis == 0 else psi[c]
            b = tmp[r, c]
            if imag:
                out[r, c] = factor * _fma(a.real, b.imag, a.imag * b.real)
            else:
                out[r, c] = factor * _fma(a.real, b.real, -(a.imag * b.imag))


def _fitfit_x_block(
    field: NDArray[np.complexfloating],
    start: int,
    width: int,
    sign: int,
    phix: NDArray[np.complex128],
    psix: NDArray[np.complex128],
    hhatx: NDArray[np.complex128],
    out: NDArray[np.float64],
) -> None:
    """x-direction pass of :func:`FITFIT` for the columns ``start:start + width`` of ``field`` into ``out``."""
    nx = field.shape[0]
    buf = np.empty((2 * nx, width), dtype=phix.dtype)
    _chirp_pad_x(phix, field, start, buf)
    ghatx = sfft.fft(buf, axis=0, overwrite_x=True)
    np.multiply(hhatx[:, na], ghatx, out=ghatx)
    tmpx = sfft.ifft(ghatx, axis=0, overwrite_x=True)
    _chirp_part(psix, tmpx, 0, 2.0 if sign == 1 else -2.0, sign != 1, out)


def _fitfit_x(
    field: NDArray[np.complexfloating], sign: int, kx: NDArray[np.float64], nx: int, dx: float, NNNx: float
) -> NDArray[np.float64]:
    """x-direction pass of :func:`FITFIT`; returns the real intermediate field."""
    phix, psix, hhatx = _fitfit_axis(nx, dx, NNNx, float(kx[0]), -nx / 4.0)
    field_x = np.empty(field.shape)
    for start in range(0, field.shape[1], _FITFIT_BLOCK_X):
        width = min(_FITFIT_BLOCK_X, field.shape[1] - start)
        _fitfit_x_block(field, start, width, sign, phix, psix, hhatx, field_x[:, start : start + width])
    return field_x


def _fitfit_y(
    field_x: NDArray[np.float64], sign: int, ky: NDArray[np.float64], ny: int, dy: float, NNNy: float
) -> NDArray[np.float64]:
    """y-direction pass of :func:`FITFIT`; linear in ``field_x``."""
    phiy, psiy, hhaty = _fitfit_axis(ny, dy, NNNy, float(ky[0]), 0.0)
    field_y = np.empty(field_x.shape)
    for start in range(0, field_x.shape[0], _FITFIT_BLOCK_Y):
        height = min(_FITFIT_BLOCK_Y, field_x.shape[0] - start)
        buf = np.empty((height, 2 * ny), dtype=phiy.dtype)
        _chirp_pad_y(phiy, field_x, start, buf)
        ghaty = sfft.fft(buf, axis=1, overwrite_x=True)
        np.multiply(hhaty[na, :], ghaty, out=ghaty)
        tmpy = sfft.ifft(ghaty, axis=1, overwrite_x=True)
        _chirp_part(psiy, tmpy, 1, 2.0, sign != 1, field_y[start : start + height])
    return field_y
