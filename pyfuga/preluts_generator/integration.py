"""
Integration routines for preLUT generation.
"""

import numpy as np

import pyfuga.utils as utils
from pyfuga.common import get_phi, get_phi_inverse
from pyfuga.constants import COORD_T, KAPPA, Y_NORM_THRESHOLD
from pyfuga.typing import ComplexArray
from pyfuga.utils import jit

from .matrices import getM
from .transforms import s_to_t, t_to_s

if utils.preludium_equivalent:
    from pyfuga.preludium_eq_routines import get_new_h2_Preludium as get_new_h2
else:
    from pyfuga.common import get_new_h2

# 2nd order Runge-Kutta coefficients
B1 = 4.0 / 3.0
B2 = -1.0 / 3.0

# Simpson's rule coefficients
C1 = 1.0 / 6.0
C2 = -1.0 / 6.0
C3 = 4.0 / 6.0
C4 = 2.0 / 6.0

# Signature helper strings for numba JIT compilation
c2 = "complex128[:,:],"
c1 = "complex128[:],"
d = "double,"
db = c1 * 6


@jit("complex128[:,:](int32,double,double, complex128[:,:],complex128[:,:], \
        double,double,double,double,double,double,double)")
def rk2_integration_step(
    j: int,
    x: float,
    h: float,
    my: ComplexArray,
    y: ComplexArray,
    kz0: float,
    psi0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    cosbeta: float,
    sinbeta: float,
) -> ComplexArray:
    r"""
    Performs a single integration step of the 2nd-order Runge-Kutta (RK2) method:
    Computes a "trial" step to the midpoint of the interval. Then uses both the values of vector Y and the matrix

    M (= -A^\dagger) at that midpoint to calculate the value of Y at the end of the interval.
    Refer to "Numerical Recipes" section 17.1 Runge-Kutta Method for more details.

    Args:
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        x: Current coordinate (s or t depending on j)
        h: Integration step size.
        my: Product of the matrix M and the vector Y.
        y: Current adjoint state vector.
        kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta:
            Parameters required to compute the matrix M via getM.

    Returns:
        The updated adjoint state vector after the RK2 intermediate step.
    """

    # Compute the matrix M at the midpoint
    m = getM(j, x + h * 0.5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute the vector Y at the midpoint using Euler's method ("trial" step)
    y_midpoint = y + h * my / 2

    # Compute the final value of Y using the midpoint values
    y2 = y + h * np.dot(np.ascontiguousarray(m), y_midpoint)
    return y2


@jit("Tuple((complex128[:,:],complex128[:,:]))(complex128[:,:], double,double,int64,\
    double,double,double,double,double,double,double,\
    complex128[:],complex128[:],complex128[:],complex128[:],complex128[:],complex128[:])")
def modified_midpoint_integration_step(
    y: ComplexArray,
    x: float,
    h: float,
    j: int,
    kz0: float,
    psi0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    cosbeta: float,
    sinbeta: float,
    dbx_const: ComplexArray,
    dby_const: ComplexArray,
    dbz_const: ComplexArray,
    dbx_lin: ComplexArray,
    dby_lin: ComplexArray,
    dbz_lin: ComplexArray,
) -> tuple[ComplexArray, ComplexArray]:
    """
    Perform a integration step using the Modified Midpoint method, as described in Numerical Recipes, §17.3.1, with two
    substeps (n=2). The estimate is fourth-order accurate, the same as the fourth-order Runge-Kutta method, but
    requires fewer function evaluations.

    Also updates the differential forcing accumulators based on the integrated state.

    Args:
        y: Current state matrix.
        x: Current state variable.
        h: Integration step size.
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        kz0: Base s-coordinate.
        psi0: Base stability correction.
        lastkz: Previous s-coordinate.
        zeta0: Stability parameter.
        cdivkL: Eddy diffusivity coefficient.
        cosbeta: Cosine of phase angle.
        sinbeta: Sine of phase angle.
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin:
            Differential forcing accumulators.

    Returns:
        A tuple where the first element is the updated state matrix (y_upper)
        and the second element is an error estimate (y_err).
    """
    # Compute matrix M and the product MY at y1, the initial point
    m = getM(j, x, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    my = np.dot(np.ascontiguousarray(m), np.ascontiguousarray(y))

    # Compute an estimate of y at x + h using RK2
    y2 = rk2_integration_step(j, x, h, my, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute an estimate of y at x + h/2 using RK2
    y3 = rk2_integration_step(j, x, h * 0.5, my, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute matrix M and the product MY at y3, the midpoint
    m = getM(j, x + h * 0.5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    my = np.dot(np.ascontiguousarray(m), np.ascontiguousarray(y3))

    # Compute an estimate of y at x + h using RK2 from the midpoint
    y4 = rk2_integration_step(j, x + h * 0.5, h * 0.5, my, y3, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Combine the two estimates to get the final output and error estimate
    y_upper = B1 * y4 + B2 * y2  # eq. (17.3.4) in Numerical Recipes
    y_err = y_upper - y4

    if j == COORD_T:
        # kz1, kzm, kz2 = self.get_kz(x + np.array([0, .5, 1]) * h)

        kz1, kzm, kz2 = [t_to_s(x + s * h, zeta0, kz0, lastkz, psi0, cdivkL) for s in [0.0, 0.5, 1.0]]

        if zeta0 < 0:
            # Unstable - the constant part of the forcing matrix F_i
            a1 = kz1 * get_phi_inverse(kz1, cdivkL)
            am = kzm * get_phi_inverse(kzm, cdivkL)
            a2 = kz2 * get_phi_inverse(kz2, cdivkL)
        else:
            # Stable and neutral - the constant part of the forcing matrix F_i
            a1 = 1 / (1 / kz1 + cdivkL)
            am = 1 / (1 / kzm + cdivkL)
            a2 = 1 / (1 / kz2 + cdivkL)

        # Update the differential forcing accumulators using Simpson's rule
        dbx_const += h * (np.conj(a1 * C1 * y[1, :] + am * C3 * y3[1, :] + a2 * (C4 * y4[1, :] + C2 * y2[1, :])))
        dby_const += h * (np.conj(a1 * C1 * y[3, :] + am * C3 * y3[3, :] + a2 * (C4 * y4[3, :] + C2 * y2[3, :])))
        dbz_const += (
            h * (np.conj(a1 * C1 * y[5, :] + am * C3 * y3[5, :] + a2 * (C4 * y4[5, :] + C2 * y2[5, :]))) * KAPPA
        )
        dbx_lin += h * (
            np.conj(kz1 * a1 * C1 * y[1, :] + kzm * am * C3 * y3[1, :] + kz2 * a2 * (C4 * y4[1, :] + C2 * y2[1, :]))
        )
        dby_lin += h * (
            np.conj(kz1 * a1 * C1 * y[3, :] + kzm * am * C3 * y3[3, :] + kz2 * a2 * (C4 * y4[3, :] + C2 * y2[3, :]))
        )
        dbz_lin += (
            h
            * (np.conj(kz1 * a1 * C1 * y[5, :] + kzm * am * C3 * y3[5, :] + kz2 * a2 * (C4 * y4[5, :] + C2 * y2[5, :])))
            * KAPPA
        )
    else:  # j == COORD_S

        xm = x + 0.5 * h
        x2 = x + h
        a1 = get_phi(zeta0, x, cdivkL)
        am = get_phi(zeta0, xm, cdivkL)
        a2 = get_phi(zeta0, x2, cdivkL)

        # Update the differential forcing accumulators using Simpson's rule
        dbx_const += h * np.conj(
            a1 * C1 * y[1, :] / x + am * C3 * y3[1, :] / xm + a2 * (C4 * y4[1, :] + C2 * y2[1, :]) / x2
        )
        dby_const += h * np.conj(
            a1 * C1 * y[3, :] / x + am * C3 * y3[3, :] / xm + a2 * (C4 * y4[3, :] + C2 * y2[3, :]) / x2
        )
        dbz_const += KAPPA * h * np.conj(C1 * y[5, :] + C3 * y3[5, :] + C4 * y4[5, :] + C2 * y2[5, :]) * KAPPA
        dbx_lin += h * np.conj((a1 * C1 * y[1, :] + am * C3 * y3[1, :]) + a2 * (C4 * y4[1, :] + C2 * y2[1, :]))
        dby_lin += h * np.conj((a1 * C1 * y[3, :] + am * C3 * y3[3, :]) + a2 * (C4 * y4[3, :] + C2 * y2[3, :]))
        dbz_lin += (
            KAPPA
            * h
            * np.conj((x * C1 * y[5, :] + xm * C3 * y3[5, :]) + x2 * (C4 * y4[5, :] + a2 * C2 * y2[5, :]))
            * KAPPA
        )
    return y_upper, y_err


@jit(f"""Tuple(({c2}{d}{d}{d}))({c2}{d}{d}{db}{d}{c2}{d}int32,{d}{d}{d}{d}{d}{d}{d})""")
def integrate_between_stations(
    y_lower: ComplexArray,
    log_s_lower: float,
    log_s_upper: float,
    dbx_const: ComplexArray,
    dby_const: ComplexArray,
    dbz_const: ComplexArray,
    dbx_lin: ComplexArray,
    dby_lin: ComplexArray,
    dbz_lin: ComplexArray,
    h: float,
    y_err: ComplexArray,
    acc: float,
    j: int,
    kz0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    psi0: float,
    cosbeta: float,
    sinbeta: float,
):
    """
    Integrates the adjoint system between two stations using the Modified Midpoint Method with adaptive step size
    control.

    Measures the error for each step, (d)y_err, and adjusts the following step size (h) accordingly to meet the accuracy
    requirement (acc).

    If it is the last step to reach the next station, the step size is adjusted to exactly reach that point.

    Measures the norm of the state vector after each step; if the norm exceeds a threshold, a substation is created.

    Args:
        Y_lower: State matrix at the lower bound of the segment.
        log_s_lower: Logarithm of the s-coordinate of the lower bound of the current segment.
        log_s_upper: Logarithm of the s-coordinate of the upper bound of the current segment.
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin:
            Forcing differential accumulators.
        h: Initial integration step size.
        y_err: Current accumulated error.
        acc: Desired accuracy level.
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        kz0: Base s-coordinate.
        lastkz: Previous s-coordinate.
        zeta0: Stability parameter (z0/L).
        cdivkL: Eddy diffusivity coefficient.
        psi0: Stability correction parameter.
        cosbeta: Cosine of phase angle.
        sinbeta: Sine of phase angle.

    Returns:
        The updated state matrix after integration and the new step size.
    """
    s_lower = kz0 * np.exp(log_s_lower)
    s_upper = kz0 * np.exp(log_s_upper)

    if j == COORD_T:
        t1 = s_to_t(s_lower, zeta0, kz0, psi0, cdivkL)
        t2 = s_to_t(s_upper, zeta0, kz0, psi0, cdivkL)
        # Generic integration variable
        x1 = t1
        x2 = t2
    else:  # j == COORD_S
        x1 = s_lower
        x2 = s_upper

    x = x1

    y_norm1 = np.linalg.norm(y_lower[:, 0])
    norm_lst = []
    # print(f'{p.logs_lower:.3f}, {h:0.5f}')
    # if p.logs_lower >= 18.15:
    #     print()
    counter = 0
    while True:
        counter += 1
        if x + 1.1 * h > x2:  # not (h + x) > x2 because this could give you numerical issues
            # step, h, big enough to reach x2 -> take the final step
            h = x2 - x
            # dy_err = self.rk2(p, Y1, x, h, j)
            y_upper, dy_err = modified_midpoint_integration_step(
                y_lower,
                x,
                h,
                j,
                kz0,
                psi0,
                lastkz,
                zeta0,
                cdivkL,
                cosbeta,
                sinbeta,
                dbx_const,
                dby_const,
                dbz_const,
                dbx_lin,
                dby_lin,
                dbz_lin,
            )

            y_err += dy_err
            h = get_new_h2(h, acc, dy_err, y_upper)

            return y_upper, h, log_s_upper, lastkz
        else:
            y_upper, dy_err = modified_midpoint_integration_step(
                y_lower,
                x,
                h,
                j,
                kz0,
                psi0,
                lastkz,
                zeta0,
                cdivkL,
                cosbeta,
                sinbeta,
                dbx_const,
                dby_const,
                dbz_const,
                dbx_lin,
                dby_lin,
                dbz_lin,
            )

            y_err += dy_err
            y_lower = y_upper.copy()
            x = x + h
            h = get_new_h2(h, acc, dy_err, y_upper)
            y_norm = np.linalg.norm(y_upper[:, 0]) / y_norm1
            norm_lst.append(y_norm)
            # If y_norm is too large, then we make a "sublevel" or "substation"
            # One can have multiple sublevels between two levels if necessary.
            if y_norm > Y_NORM_THRESHOLD:
                if j == COORD_T:  # pragma: no cover
                    s_lower = t_to_s(x, zeta0, kz0, lastkz, psi0, cdivkL)
                    lastkz = s_lower
                    log_s_lower = np.log(s_lower / kz0)
                else:
                    # This might be a mistake according to sqot, but it is used.
                    log_s_lower = np.log(x / kz0)

                return y_upper, h, log_s_lower, lastkz
