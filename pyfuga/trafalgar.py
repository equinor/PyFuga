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

import multiprocessing
from collections.abc import Sequence

import numpy as np
import xarray as xr
from numpy import newaxis as na
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from pyfuga.constants import KAPPA

M_PI = 3.141592653589793


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
        n_cpu : int or None
            Number of parallel worker processes.  ``None`` uses all available
            CPUs; ``1`` disables multiprocessing.

        Returns
        -------
        xarray.Dataset
            Wake fields ``UL``, ``VL``, ... with dimensions ``(z, x, y)`` and
            coordinates ``x``, ``y``, ``z``.
        """
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

        map_func = map if n_cpu == 1 else multiprocessing.Pool(n_cpu).imap

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
                self.x_lst,
                self.y_lst,
                self.verbose,
                legacy,
            )
            for t in tiles
        )

        # Accumulate tile contributions (linearity of Fourier integral)
        luts_accum = None
        luts = list(
            tqdm(map_func(make_luts_per_mesh, args), total=len(tiles), disable=self.verbose, desc="LUTS: adding meshes")
        )
        for lut in luts:
            if luts_accum is None:
                luts_accum = lut
            else:
                lut_vars = [v for v in lut.data_vars if lut[v].dims == ("z", "x", "y")]
                with xr.set_options(keep_attrs=True):
                    for v in lut_vars:
                        luts_accum[v] = luts_accum[v] + lut[v]

        return luts_accum


# ---------------------------------------------------------------------------
# Per-tile worker
# ---------------------------------------------------------------------------


def make_luts_per_mesh(args: tuple) -> xr.Dataset:
    """
    Evaluate the FFIT for a single wavenumber tile and return wake fields.

    A tile is defined by a rectangular region in :math:`(k_x, k_y)` space,
    parameterised by ``(NNNx, k0x, NNNy, k0y)``.  The spectral fields are
    interpolated from the Fourier LUTs onto the tile's :math:`k`-grid, weighted
    by the rotor Gaussian force spectrum (``force``) and the normalisation
    factor (``fuzz``), and then transformed to physical space via ``FITFIT``.

    Parameters
    ----------
    args : tuple
        Packed arguments from :meth:`Trafalgar.make_luts`.

    Returns
    -------
    xarray.Dataset
        Physical-space fields for this tile, same structure as the final output.
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
        x_lst,
        y_lst,
        verbose,
        legacy,
    ) = args

    beta_lst = fluts.beta.values
    kz0_lst = fluts.kz0.values

    # sign_dict: +1 → take real part of FFIT output; -1 → take imaginary part.
    sign_dict = {"UL": 1, "VL": -1, "WL": 1, "PL": 1, "UT": -1, "VT": 1, "WT": -1, "PT": -1}

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

    luts_dict = {}
    for var, sign in tqdm(sign_dict.items(), desc="Trafalgar:", disable=(not verbose)):
        if var not in fluts:
            continue

        flut = fluts[var].values  # shape: (n_beta, n_kz0, n_levels)

        # Difference #2: legacy multiplies spectral data by sign before interpolation
        if legacy:
            flut = flut * sign

        luts = []
        for i in tqdm(range(len(fluts.level)), disable=(not verbose), leave=False):
            re, im = flut[:, :, i].real, flut[:, :, i].imag

            # Bivariate spline interpolation of LUT onto (angleTab, kz0Tab)
            field = RectBivariateSpline(
                beta_lst, np.log(kz0_lst), re, kx=interpolation_order, ky=interpolation_order
            ).ev(angleTab, kz0Tab) + 1j * RectBivariateSpline(
                beta_lst, np.log(kz0_lst), im, kx=interpolation_order, ky=interpolation_order
            ).ev(
                angleTab, kz0Tab
            )

            # Apply force spectrum and normalisation
            field *= fuzz * force

            # Difference #3: legacy zeroes the DC component
            if legacy:
                field[0, 0] = 0.0

            # 2D FFIT: spectral → physical space
            luts.append(FITFIT(field, sign, kx, ky, nx, dx, ny, dy, NNN=(NNNx, NNNy)).real)

        luts_dict[var] = (("z", "x", "y"), luts)

    return xr.Dataset(
        {**luts_dict, **{k: fluts[k] for k in ["diameter", "hubheight", "z0"]}},
        coords={"x": x_lst, "y": y_lst, "z": np.atleast_1d(fluts.z)},
        attrs=fluts.attrs,
    )


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
    return k0_ + c * (np.arange(int(n)) + 0.5)


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

    NNNs = np.sort(np.array(NNNs, dtype=int))[::-1]  # finest first
    if NNNs.size == 0:
        return []

    # Special case: kmin=0 (legacy mode) — skip the kmax>kmin>0 check
    if kmin == 0.0:
        # Single tile starting at k=0 for each NNN level
        tiles: list[tuple[float, float, float, float]] = []
        for NNN in NNNs:
            tiles.append((int(NNN), 0.0, int(NNN), 0.0))
        return tiles

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
        for NNN in NNNs:
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

    # Spectral spacings
    bx = np.pi / (dx * nx) / NNNx
    by = np.pi / (dy * ny) / NNNy
    abx = dx * bx
    aby = dy * by

    # Starting positions in physical space
    i0 = -nx / 4.0
    x0 = abx * i0 / bx  # = dx * i0
    y0 = 0.0
    k0x = kx[0]
    k0y = ky[0]

    i = np.arange(nx)
    phix = np.exp(1j * ((k0x + bx * i) * x0 + abx / 2 * i**2))
    psix = np.exp(1j * abx * (k0x * i / bx + i**2 / 2))

    j = np.arange(ny)
    phiy = np.exp(1j * ((k0y + by * j) * y0 + aby / 2 * j**2))
    psiy = np.exp(1j * aby * (k0y * j / by + j**2 / 2))

    # Kernel h̃ along x
    i_ = np.arange(nx + 1)
    auxx = np.exp(-1j * abx / 2.0 * i_ * i_)
    hhatxList = np.fft.ifft(np.r_[auxx, auxx[::-1][1:-1]])

    j_ = np.arange(ny + 1)
    auxy = np.exp(-1j * aby / 2.0 * j_ * j_)
    hhatyList = np.fft.ifft(np.r_[auxy, auxy[::-1][1:-1]])

    # ---- Transform along x ----
    ghatx = np.roll(np.fft.ifft(phix[:, na] * field, n=2 * nx, axis=0)[::-1], 1, axis=0)
    tmpx = np.fft.ifft(hhatxList[:, na] * ghatx, axis=0) * 2 * nx
    field = 2.0 * (psix[:, na] * tmpx[:nx]).real if sign == 1 else -2.0 * (psix[:, na] * tmpx[:nx]).imag

    # ---- Transform along y ----
    ghaty = np.roll(np.fft.ifft(phiy[na, :] * field, n=2 * ny, axis=1)[:, ::-1], 1, 1)
    tmpy = np.fft.ifft(hhatyList[na, :] * ghaty, axis=1) * 2 * ny
    field = 2.0 * (psiy[na, :] * tmpy[:, :ny]).real if sign == 1 else 2.0 * (psiy[na, :] * tmpy[:, :ny]).imag

    return field
