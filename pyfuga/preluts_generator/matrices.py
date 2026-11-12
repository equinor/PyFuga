"""Adjoint system matrices for preLUT generation with stability effects."""

import numpy as np

from pyfuga.common import get_phi, get_phi_inverse, get_psi
from pyfuga.constants import COORD_S, COORD_T, KAPPA, KAPPA_SQUARED
from pyfuga.typing import ComplexArray
from pyfuga.utils import jit

from .transforms import t_to_s, u0


@jit("Tuple((double, double))(int32, double, double, double, double, double, double)")
def compute_kz_u(
    j: int, t: float, kz0: float, psi0: float, lastkz: float, zeta0: float, cdivkL: float
) -> tuple[float, float]:
    """Compute the current s (kz) coordinate and wind speed, u."""
    assert j in (COORD_T, COORD_S)
    if j == COORD_T:
        kz = t_to_s(t, zeta0, kz0, lastkz, psi0, cdivkL)
        u = t / KAPPA
    else:  # j == COORD_S
        kz = t
        u = u0(kz, kz0, get_psi(zeta0, kz, cdivkL), psi0)
    return kz, u


@jit("Tuple((double, double, double, double))(double, double, double, double, double)")
def compute_kstar_params(
    zeta0: float, kz: float, cdivkL: float, cosbeta: float, sinbeta: float
) -> tuple[float, float, float, float]:
    """Compute K_star, dKstar_dz, K_star_cos, and K_star_sin based on stability."""
    if zeta0 < 0:  # unstable
        K_star = KAPPA * kz * get_phi_inverse(kz, cdivkL)
        dKstar_dz = K_star * (1.0 / kz + 0.25 / (1.0 / cdivkL + kz))
    else:  # stable or neutral
        aux = get_phi(zeta0, kz, cdivkL)
        K_star = KAPPA * kz / aux
        dKstar_dz = KAPPA / (aux**2)
    K_star_cos = K_star * cosbeta
    K_star_sin = K_star * sinbeta
    return K_star, dKstar_dz, K_star_cos, K_star_sin


@jit
def _fill_M_t(m: ComplexArray, u: float, K_star: float, dKstar_dz: float, K_star_cos: float, K_star_sin: float) -> None:
    """Set the non-zero entries of ``create_M_t`` in the zeroed ``m`` (same values as a nested-list literal)."""
    m[0, 1] = complex(-(K_star**2), K_star_cos * u) / KAPPA_SQUARED
    m[0, 4] = complex(0, -K_star_cos / KAPPA)
    m[0, 5] = complex(0, -2 * dKstar_dz * K_star_cos / KAPPA)  # TODO: misplaced factor of K*/KAPPA?
    m[1, 0] = -1
    m[1, 5] = complex(0, -K_star_cos)
    m[2, 3] = complex(-(K_star**2), K_star_cos * u) / KAPPA_SQUARED
    m[2, 4] = complex(0, -K_star_sin / KAPPA)
    m[2, 5] = complex(0, -2 * dKstar_dz * K_star_sin / KAPPA)  # TODO: misplaced factor of K*/KAPPA?
    m[3, 2] = -1
    m[3, 5] = complex(0, -K_star_sin)
    m[4, 1] = complex(-1, -dKstar_dz * K_star_cos) / KAPPA_SQUARED  # TODO: misplaced factor of K*/KAPPA?
    m[4, 3] = complex(0, -dKstar_dz * K_star_sin / KAPPA_SQUARED)  # TODO: misplaced factor of K*/KAPPA?
    m[4, 5] = complex(K_star**2, -K_star_cos * u) / (KAPPA)
    m[5, 1] = complex(0, K_star_cos / KAPPA_SQUARED)
    m[5, 3] = complex(0, K_star_sin / KAPPA_SQUARED)


@jit
def _fill_M_s(
    m: ComplexArray,
    u: float,
    K_star: float,
    dKstar_dz: float,
    cosbeta: float,
    sinbeta: float,
    K_star_cos: float,
    K_star_sin: float,
) -> None:
    """Set the non-zero entries of ``create_M_s`` in the zeroed ``m`` (same values as a nested-list literal)."""
    m[0, 1] = complex(-1, cosbeta * u / K_star)
    m[0, 4] = complex(0, -cosbeta)
    m[0, 5] = complex(0, -2 * dKstar_dz * cosbeta)
    m[1, 0] = -1
    m[1, 1] = dKstar_dz / K_star
    m[1, 5] = complex(0, -K_star_cos)
    m[2, 3] = complex(-1, cosbeta * u / K_star)
    m[2, 4] = complex(0, -sinbeta)
    m[2, 5] = complex(0, -2 * dKstar_dz * sinbeta)
    m[3, 2] = -1
    m[3, 3] = dKstar_dz / K_star
    m[3, 5] = complex(0, -K_star_sin)
    m[4, 1] = complex(-1 / K_star**2, dKstar_dz / K_star * cosbeta)  # TODO: missing complex conjugation?
    m[4, 3] = complex(0, -dKstar_dz / K_star * sinbeta)
    m[4, 5] = complex(K_star, -cosbeta * u)
    m[5, 1] = complex(0, cosbeta / K_star)
    m[5, 3] = complex(0, sinbeta / K_star)


@jit("complex128[:,:](double, double, double, double, double)")
def create_M_t(u: float, K_star: float, dKstar_dz: float, K_star_cos: float, K_star_sin: float) -> ComplexArray:
    """Create the 6x6 adjoint system matrix M = -A* in t-space."""
    m = np.zeros((6, 6), dtype=np.complex128)
    _fill_M_t(m, u, K_star, dKstar_dz, K_star_cos, K_star_sin)
    return m


@jit("complex128[:,:](double, double, double, double, double, double, double)")
def create_M_s(
    u: float, K_star: float, dKstar_dz: float, cosbeta: float, sinbeta: float, K_star_cos: float, K_star_sin: float
) -> ComplexArray:
    """Create the 6x6 adjoint system matrix M = -A* in s-space."""
    m = np.zeros((6, 6), dtype=np.complex128)
    _fill_M_s(m, u, K_star, dKstar_dz, cosbeta, sinbeta, K_star_cos, K_star_sin)
    return m


@jit("complex128[:,:](int32,double,double, double, double,double,double,double,double)")
def getM(
    j: int,
    t: float,
    kz0: float,
    psi0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    cosbeta: float,
    sinbeta: float,
) -> ComplexArray:
    """
    Compute the adjoint system matrix M = -A* with stability effects included in the eddy diffusivity.
    Refactored to use helper functions for clarity.

    Args:
        j: Coordinate system indicator (COORD_T = 1: t = u0*kappa, COORD_S = 2: s = kz).
        t: The input coordinate (t or s depending on j).
        kz0: The base s-coordinate.
        psi0: The base stability correction value at kz0.
        lastkz: The s-coordinate from the last station or integration step.
        zeta0: The stability parameter (z0/L).
        cdivkL: Eddy diffusivity scaling parameter.
        cosbeta: The cosine of the phase angle.
        sinbeta: The sine of the phase angle.

    Returns:
        A 6x6 complex matrix M.
    """
    kz, u = compute_kz_u(j, t, kz0, psi0, lastkz, zeta0, cdivkL)
    K_star, dKstar_dz, K_star_cos, K_star_sin = compute_kstar_params(zeta0, kz, cdivkL, cosbeta, sinbeta)
    if j == COORD_T:
        return create_M_t(u, K_star, dKstar_dz, K_star_cos, K_star_sin)
    else:  # j == COORD_S:
        return create_M_s(u, K_star, dKstar_dz, cosbeta, sinbeta, K_star_cos, K_star_sin)


@jit
def getM_into(
    m: ComplexArray,
    j: int,
    t: float,
    kz: float,
    kz0: float,
    psi0: float,
    zeta0: float,
    cdivkL: float,
    cosbeta: float,
    sinbeta: float,
) -> None:
    """:func:`getM` written into ``m``, with ``kz`` (``= t_to_s(t, ...)`` for COORD_T, ``= t`` for COORD_S)
    already computed."""
    u = t / KAPPA if j == COORD_T else u0(kz, kz0, get_psi(zeta0, kz, cdivkL), psi0)
    K_star, dKstar_dz, K_star_cos, K_star_sin = compute_kstar_params(zeta0, kz, cdivkL, cosbeta, sinbeta)
    for r in range(6):
        for c in range(6):
            m[r, c] = 0
    if j == COORD_T:
        _fill_M_t(m, u, K_star, dKstar_dz, K_star_cos, K_star_sin)
    else:
        _fill_M_s(m, u, K_star, dKstar_dz, cosbeta, sinbeta, K_star_cos, K_star_sin)
