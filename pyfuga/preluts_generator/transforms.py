"""
Module for Monin-Obukhov Similarity Theory (MOST) transformations.
"""

import numpy as np

from pyfuga.common import get_phi_inverse, get_psi
from pyfuga.constants import CM_STABLE, CM_UNSTABLE, KAPPA
from pyfuga.utils import jit


@jit("double(double,double,double,double)")
def u0(kz: float, kz0: float, psi: float, psi0: float) -> float:
    """
    Compute the wind speed from Monin-Obukhov Similarity Theory (MOST) normalised by
    the friction velocity, ustar.

    Arg:
        kz: Current wavenumber
        kz0: Base wavenumber
        psi: Stability correction function at kz
        psi0: Stability correction function at kz0

    Returns:
        The normalised wind speed
    """
    return (np.log(kz / kz0) + psi - psi0) / KAPPA


@jit("double(double, double)")
def t_to_s_neutral(t: float, kz0: float) -> float:
    """
    Transform from t = u0 * kappa to s = kz for neutral conditions (zeta0 = 0).
    """
    return kz0 * np.exp(t)


@jit("double(double, double, double, double)")
def t_to_s_stable(t: float, zeta0: float, kz0: float, lastkz: float) -> float:
    """
    Transform from t = u0 * kappa to s = kz for stable conditions (zeta0 > 0).
    """
    a = CM_STABLE * zeta0
    b = t + a + np.log(a)
    if b < 1:
        # Use lastkz as a proxy for the current value.
        ax = np.exp(b) if lastkz < 0 else a * lastkz / kz0
        while True:
            dax = (np.exp(b - ax) - ax) / (1 + ax)
            ax = ax + dax
            if abs(dax / ax) < 1e-14:
                break
    else:
        ax = b if lastkz < 0 else a * lastkz / kz0
        while True:
            dax = (b - ax - np.log(ax)) / (1 + 1 / ax)
            ax = ax + dax
            if abs(dax / ax) < 1e-14:
                break
    return kz0 * ax / a


@jit("double(double, double, double, double, double)")
def t_to_s_unstable(t: float, zeta0: float, lastkz: float, psi0: float, cdivkL: float) -> float:
    """
    Unstable MOST mapping from t = u0 * kappa to kz (s-coordinate).

    This routine inverts the unstable-regime relation using a Newton iteration.
    It relies on `lastkz` as an initial guess and includes a safety fallback for
    pathological iterations.
    """
    b0 = psi0 + np.log(CM_UNSTABLE * zeta0 / 8)
    b = t + b0
    if lastkz < 0:
        x = np.exp(b)
    else:
        aux = get_phi_inverse(lastkz, cdivkL)
        x = (cdivkL * lastkz) / ((aux**2 + 1) * (1 + aux) ** 2)

        # Guard against negative x for arctans and logs
        if x <= 0:
            x = np.exp(b)
            dx = x
        else:
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1) ** 2

        while abs(dx / x) > 1e-14:
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1) ** 2
            x = x + dx
            if x < 0:  # pragma: no cover  (safety fallback for pathological iterations)
                x = np.exp(b)
                dx = x
    return 8 * x * (1 + x**2) / (cdivkL * (1 - x) ** 4)


@jit("double(double, double, double, double, double, double)")
def t_to_s(t: float, zeta0: float, kz0: float, lastkz: float, psi0: float, cdivkL: float) -> float:
    """
    Dispatcher for coordinate transform from t = u0 * kappa to s = kz.
    Calls a helper function based on the stability parameter (zeta0).

    Args:
        t: t coordinate
        zeta0: Stability parameter (z0/L)
        kz0: Base s-coordinate
        lastkz: Previous s-coordinate
        psi0: Base stability correction value
        cdivkL: Eddy diffusivity coefficient

    Returns:
        The wavenumber kz.
    """
    if np.abs(zeta0) < 1e-14:
        return t_to_s_neutral(t, kz0)
    if zeta0 > 0:
        return t_to_s_stable(t, zeta0, kz0, lastkz)
    return t_to_s_unstable(t, zeta0, lastkz, psi0, cdivkL)


@jit("double(double, double, double, double, double)")
def s_to_t(s: float, zeta0: float, kz0: float, psi0: float, cdivkL: float) -> float:
    """Forward MOST mapping from s = kz to t = u0 * kappa."""
    return KAPPA * u0(s, kz0, get_psi(zeta0, s, cdivkL), psi0)
