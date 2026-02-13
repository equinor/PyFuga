import numpy as np
from scipy.optimize import fsolve, root_scalar
from scipy.special import lambertw

from pyfuga.constants import CM_STABLE

CM_UNSTABLE = -19.3  # DIFFERENT FROM constants.py. See issue #30 in pyfuga repo.


def phi(zeta):
    zeta = np.asarray(zeta, dtype=float)
    zeta = np.atleast_1d(zeta)
    out = np.empty_like(zeta, dtype=float)

    m = zeta <= 0
    out[m] = (1 + CM_UNSTABLE * zeta[m]) ** -0.25
    out[~m] = 1 + CM_STABLE * zeta[~m]
    return out


def psi(zeta, unstable=""):
    zeta = np.atleast_1d(zeta)
    ind = zeta < 0
    psi_n = np.zeros(zeta.shape)
    aux = phi(zeta) ** -1
    if unstable == "Wilson":
        psi_n[ind] = 3 * np.log(1 + (1 + 3.6 * np.abs(zeta[ind]) ** (2 / 3)) ** 0.5)
    else:
        aux2 = (1.0 + aux[ind]) ** 2 * (1 + aux[ind] ** 2)
        psi_n[ind] = -np.log(8.0 / aux2) - 2.0 * np.arctan(CM_UNSTABLE * zeta[ind] / aux2)
    psi_n[~ind] = 1 - aux[~ind] ** -1
    psi_n[zeta == 0] = 0
    return psi_n


def _g_unstable(x, ti, zeta0):
    # x must be positive; if a solver probes x<=0, make g invalid/very negative
    if np.any(np.asarray(x) <= 0):
        return -np.inf
    return x / np.exp(psi(x * zeta0) - psi(zeta0)) - np.exp(1.0 / ti)


def _solve_x_unstable(ti, zeta0, zref, z0_limit):
    x_max = float(zref / z0_limit)

    x_lo = 1e-12
    f_lo = _g_unstable(x_lo, ti, zeta0)

    x_hi = min(x_max, max(10.0, zref / 1e-3))
    f_hi = _g_unstable(x_hi, ti, zeta0)

    # Track the last finite x_hi we saw
    x_hi_good = x_hi if np.isfinite(f_hi) else None
    f_hi_good = f_hi if np.isfinite(f_hi) else None

    for _ in range(80):
        if np.isfinite(f_lo) and np.isfinite(f_hi) and np.sign(f_lo) != np.sign(f_hi):
            break
        if x_hi >= x_max:
            break

        x_hi = min(x_max, x_hi * 1.5)
        f_hi = _g_unstable(x_hi, ti, zeta0)

        if np.isfinite(f_hi):
            x_hi_good = x_hi
            f_hi_good = f_hi

    # If we never bracketed, explicitly check x_max:
    if not (np.isfinite(f_lo) and np.isfinite(f_hi) and np.sign(f_lo) != np.sign(f_hi)):
        f_at_max = _g_unstable(x_max, ti, zeta0)

        if np.isfinite(f_at_max) and f_at_max < 0:
            return x_max

        # Fallback bracket search between last finite x_hi and x_max
        if (
            np.isfinite(f_at_max)
            and f_at_max > 0
            and x_hi_good is not None
            and f_hi_good is not None
            and x_hi_good < x_max
            and np.sign(f_hi_good) != np.sign(f_at_max)
        ):
            sol = root_scalar(_g_unstable, args=(ti, zeta0), bracket=(x_hi_good, x_max), method="brentq")
            if sol.converged and 0 < sol.root <= x_max:
                return float(sol.root)

    # Normal bracketing solve
    if np.isfinite(f_lo) and np.isfinite(f_hi) and np.sign(f_lo) != np.sign(f_hi):
        sol = root_scalar(_g_unstable, args=(ti, zeta0), bracket=(x_lo, x_hi), method="brentq")
        if sol.converged and 0 < sol.root <= x_max:
            return float(sol.root)

    # Fallback: fsolve, but if it doesn't converge, *still* allow clamp behaviour
    x0 = min(zref / 1e-3, x_max)
    x, _, ier, msg = fsolve(lambda xx: _g_unstable(xx, ti, zeta0), x0, full_output=True)

    if ier == 1 and np.isfinite(x[0]) and 0 < x[0] <= x_max:
        return float(x[0])

    # If we get here, last chance: check clamp behaviour again
    f_at_max = _g_unstable(x_max, ti, zeta0)
    if np.isfinite(f_at_max) and f_at_max < 0:
        return x_max

    raise RuntimeError(f"Unstable z0_from_TI solve failed: ti={ti}, zeta0={zeta0}, x_max={x_max}, ier={ier}, msg={msg}")


def z0_from_TI(TI, zref, zeta0, z0_limit=1e-5):
    TI = np.atleast_1d(TI).astype(float)

    if zeta0 == 0:
        z0 = zref * np.exp(-1.0 / TI)

    elif zeta0 > 0:
        a = CM_STABLE * zeta0
        b = 1 / TI
        x = np.real(lambertw(a * np.exp(a + b))) / a
        z0 = zref / x

    else:
        x = np.array([_solve_x_unstable(ti, zeta0, zref, z0_limit) for ti in TI])
        z0 = zref / x

    z0 = np.maximum(z0, z0_limit)
    return z0
