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
def compute_kK_params(
    zeta0: float, kz: float, cdivkL: float, cosbeta: float, sinbeta: float
) -> tuple[float, float, float, float]:
    """Compute kK, dKdz, kKcos, and kKsin based on stability."""
    if zeta0 < 0:  # unstable
        kK = KAPPA * kz * get_phi_inverse(kz, cdivkL)
        dKdz = kK * (1.0 / kz + 0.25 / (1.0 / cdivkL + kz))
    else:  # stable or neutral
        aux = get_phi(zeta0, kz, cdivkL)
        kK = KAPPA * kz / aux
        dKdz = KAPPA / (aux**2)
    kKcos = kK * cosbeta
    kKsin = kK * sinbeta
    return kK, dKdz, kKcos, kKsin


@jit("complex128[:,:](double, double, double, double, double)")
def create_M_t(u: float, kK: float, dKdz: float, kKcos: float, kKsin: float) -> ComplexArray:
    """Create the 6x6 adjoint system matrix M = -A* in t-space."""
    return np.array(
        [
            # M(1,2)=cmplx(-kK**2,kKcos*u)/kappa2
            # M(1,5)=cmplx(0.0E0,-kKcos/kappa)
            # M(1,6)=cmplx(0.0E0,-2.0E0*dKdz*kKcos/kappa/pscale)
            [
                0,
                complex(-(kK**2), kKcos * u) / KAPPA_SQUARED,
                0,
                0,
                complex(0, -kKcos / KAPPA),
                complex(0, -2 * dKdz * kKcos / KAPPA),  # ERROR - misplaced factor of kK*/KAPPA
            ],
            # M(2,1)=cmplx(-1.0E0,0.0E0)
            # M(2,6)=cmplx(0.0E0,-kKcos/pscale)
            [-1, 0, 0, 0, 0, complex(0, -kKcos)],
            # M(3,4)=M(1,2)
            # M(3,5)=cmplx(0.0E0,-kKsin/kappa)
            # M(3,6)=cmplx(0.0E0,-2.0E0*dKdz*kKsin/kappa/pscale)
            [
                0,
                0,
                0,
                complex(-(kK**2), kKcos * u) / KAPPA_SQUARED,
                complex(0, -kKsin / KAPPA),
                complex(0, -2 * dKdz * kKsin / KAPPA),  # ERROR - misplaced factor of kK*/KAPPA
            ],
            # M(4,3)=cmplx(-1.0E0,0.0E0)
            # M(4,6)=cmplx(0.0E0,-kKsin/pscale)
            [0, 0, -1, 0, 0, complex(0, -kKsin)],
            # M(5,2)=cmplx(-1.0E0,-dKdz*kKcos)/kappa2
            # M(5,4)=cmplx(0.0E0,-dKdz*kKsin/kappa2)
            # M(5,6)=cmplx(kK**2,-kKcos*u)/(pscale*kappa)
            [
                0,
                complex(-1, -dKdz * kKcos) / KAPPA_SQUARED,  # ERROR - misplaced factor of kK*/KAPPA
                0,
                complex(0, -dKdz * kKsin / KAPPA_SQUARED),  # ERROR - misplaced factor of kK*/KAPPA
                0,
                complex(kK**2, -kKcos * u) / (KAPPA),
            ],
            # M(6,2)=cmplx(0.0E0,kKcos/kappa2*pscale)
            # M(6,4)=cmplx(0.0E0,kKsin/kappa2*pscale)
            [
                0,
                complex(0, kKcos / KAPPA_SQUARED),
                0,
                complex(0, kKsin / KAPPA_SQUARED),
                0,
                0,
            ],
        ]
    )


@jit("complex128[:,:](double, double, double, double, double, double, double)")
def create_M_s(
    u: float, kK: float, dKdz: float, cosbeta: float, sinbeta: float, kKcos: float, kKsin: float
) -> ComplexArray:
    """Create the 6x6 adjoint system matrix M = -A* in s-space."""
    return np.array(
        [
            # M(1,2)=dcmplx(-1.0D0,cosbeta*u/kK)
            # M(1,5)=dcmplx(0.0D0,-cosbeta)
            # M(1,6)=dcmplx(0.0D0,-2.0D0*dKdz*cosbeta/pscale)
            [
                0,
                complex(-1, cosbeta * u / kK),
                0,
                0,
                complex(0, -cosbeta),
                complex(0, -2 * dKdz * cosbeta),
            ],
            # M(2,1)=-1.0D0
            # M(2,2)=dKdz/kK
            # M(2,6)=dcmplx(0.0D0,-kK*cosbeta/pscale)
            [
                -1,
                dKdz / kK,
                0,
                0,
                0,
                complex(0, -kKcos),
            ],
            # M(3,4)=dcmplx(-1.0D0,cosbeta*u/kK)
            # M(3,5)=dcmplx(0.0D0,-sinbeta)
            # M(3,6)=dcmplx(0.0D0,-2.0D0*dKdz*sinbeta/pscale)
            [
                0,
                0,
                0,
                complex(-1, cosbeta * u / kK),
                complex(0, -sinbeta),
                complex(0, -2 * dKdz * sinbeta),
            ],
            # M(4,3)=-1.0D0
            # M(4,4)=M(2,2)
            # M(4,6)=dcmplx(0.0D0,-kK*sinbeta/pscale)
            [
                0,
                0,
                -1,
                dKdz / kK,
                0,
                complex(0, -kKsin),
            ],
            # M(5,2)=dcmplx(-1.0D0/kK**2,dKdz/kK*cosbeta)
            # M(5,4)=dcmplx(0.0D0,-dKdz/kK*sinbeta)
            # M(5,6)=dcmplx(kK,-cosbeta*u)
            [
                0,
                complex(-1 / kK**2, dKdz / kK * cosbeta),  # ERROR - missing complex conjugation
                0,
                complex(0, -dKdz / kK * sinbeta),
                0,
                complex(kK, -cosbeta * u),
            ],
            # M(6,2)=dcmplx(0.0D0,cosbeta*pscale/kK)
            # M(6,4)=dcmplx(0.0D0,sinbeta*pscale/kK)
            [
                0,
                complex(0, cosbeta / kK),
                0,
                complex(0, sinbeta / kK),
                0,
                0,
            ],
        ]
    )


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
    kK, dKdz, kKcos, kKsin = compute_kK_params(zeta0, kz, cdivkL, cosbeta, sinbeta)
    if j == COORD_T:
        return create_M_t(u, kK, dKdz, kKcos, kKsin)
    else:  # j == COORD_S:
        return create_M_s(u, kK, dKdz, cosbeta, sinbeta, kKcos, kKsin)
