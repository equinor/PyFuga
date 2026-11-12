"""
Integration routines for preLUT generation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

import pyfuga.utils as utils
from pyfuga.common import get_phi, get_phi_inverse
from pyfuga.constants import COORD_S, COORD_T, KAPPA, Y_NORM_THRESHOLD
from pyfuga.typing import ComplexArray
from pyfuga.utils import fma, jit

from .matrices import getM, getM_into
from .nodes import qr_decomposition
from .transforms import s_to_t, t_to_s

__all__ = [
    "get_new_h2",
    "rk2_integration_step",
    "modified_midpoint_integration_step",
    "integrate_between_stations",
    "integrate_all_stations",
]


def _get_new_h2_impl() -> Callable[[float, float, ComplexArray, ComplexArray], float]:
    """Resolve step-size control at import time without a cyclic mypy graph."""
    if utils.preludium_equivalent:
        from pyfuga.preludium_eq_routines import get_new_h2_Preludium

        return get_new_h2_Preludium
    from pyfuga.common import get_new_h2

    return get_new_h2


get_new_h2 = _get_new_h2_impl()


@njit(cache=True, nogil=True)
def _zmatmul6(m: ComplexArray, y: ComplexArray, out: ComplexArray) -> None:  # pragma: no cover - compiled
    """``out = m @ y`` for 6x6 complex matrices with the operation order of OpenBLAS's ``zgemm`` (columns 0-3:
    one fma chain over k; columns 4-5: separate re/im product sums), skipping the exact zeros of the sparse ``m``
    (they add nothing to an fma chain), so the result is bitwise equal to ``np.dot`` but much faster."""
    for i in range(6):
        # 16 independent accumulators (columns unrolled) so the fma chains overlap
        re0 = re1 = re2 = re3 = im0 = im1 = im2 = im3 = 0.0
        rr4 = ii4 = ri4 = ir4 = rr5 = ii5 = ri5 = ir5 = 0.0
        for k in range(6):
            xr = m[i, k].real
            xi = m[i, k].imag
            if xr == 0.0 and xi == 0.0:
                continue
            nxi = -xi
            b0, b1, b2, b3, b4, b5 = y[k, 0], y[k, 1], y[k, 2], y[k, 3], y[k, 4], y[k, 5]
            re0 = fma(nxi, b0.imag, fma(xr, b0.real, re0))
            im0 = fma(xi, b0.real, fma(xr, b0.imag, im0))
            re1 = fma(nxi, b1.imag, fma(xr, b1.real, re1))
            im1 = fma(xi, b1.real, fma(xr, b1.imag, im1))
            re2 = fma(nxi, b2.imag, fma(xr, b2.real, re2))
            im2 = fma(xi, b2.real, fma(xr, b2.imag, im2))
            re3 = fma(nxi, b3.imag, fma(xr, b3.real, re3))
            im3 = fma(xi, b3.real, fma(xr, b3.imag, im3))
            rr4 = fma(xr, b4.real, rr4)
            ii4 = fma(xi, b4.imag, ii4)
            ri4 = fma(xr, b4.imag, ri4)
            ir4 = fma(xi, b4.real, ir4)
            rr5 = fma(xr, b5.real, rr5)
            ii5 = fma(xi, b5.imag, ii5)
            ri5 = fma(xr, b5.imag, ri5)
            ir5 = fma(xi, b5.real, ir5)
        out[i, 0] = complex(re0, im0)
        out[i, 1] = complex(re1, im1)
        out[i, 2] = complex(re2, im2)
        out[i, 3] = complex(re3, im3)
        out[i, 4] = complex(rr4 - ii4, ri4 + ir4)
        out[i, 5] = complex(rr5 - ii5, ri5 + ir5)


def _np_matmul(m: ComplexArray, y: ComplexArray, out: ComplexArray) -> None:
    np.dot(m, y, out)


_blas_matmul = njit(cache=True, nogil=True)(_np_matmul)


def _zmatmul6_matches_blas() -> bool:
    """True if :func:`_zmatmul6` reproduces this machine's BLAS bit for bit (the operation order is
    BLAS-kernel specific), checked on random dense and sparse matrices spanning many magnitudes."""
    rng = np.random.default_rng(0)
    out_blas = np.empty((6, 6), dtype=np.complex128)
    out_kernel = np.empty_like(out_blas)
    for t in range(400):
        m, y = (
            np.asarray(
                (rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))) * 10.0 ** rng.uniform(-6, 6, (6, 6)),
                dtype=np.complex128,
            )
            for _ in range(2)
        )
        if t % 2:
            m[rng.random((6, 6)) < 0.6] = 0
        _blas_matmul(m, y, out_blas)
        _zmatmul6(m, y, out_kernel)
        if out_blas.tobytes() != out_kernel.tobytes():
            return False
    return True


# Product of the 6x6 system matrix with the state. The fast kernel is only used when compiled and when it matches
# the BLAS that np.dot would call (otherwise np.dot itself is used), so results never change.
_matmul = _zmatmul6 if utils.numba_jit and _zmatmul6_matches_blas() else _blas_matmul if utils.numba_jit else _np_matmul

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


@jit
def _update_forcing(
    y: ComplexArray,
    y2: ComplexArray,
    y3: ComplexArray,
    y4: ComplexArray,
    x: float,
    h: float,
    j: int,
    kz0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    psi0: float,
    dbx_const: ComplexArray,
    dby_const: ComplexArray,
    dbz_const: ComplexArray,
    dbx_lin: ComplexArray,
    dby_lin: ComplexArray,
    dbz_lin: ComplexArray,
    kz1: float,
    kzm: float,
    kz2: float,
) -> None:
    """Update the differential forcing accumulators using Simpson's rule (eq. (7)-(8) in Numerical Solution),
    element by element. ``kz1``/``kzm``/``kz2`` are ``t_to_s`` at ``x``, ``x + 0.5 * h``, ``x + h`` (COORD_T only)."""
    if j == COORD_T:
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

        for c in range(y.shape[1]):
            dbx_const[c] += h * (np.conj(a1 * C1 * y[1, c] + am * C3 * y3[1, c] + a2 * (C4 * y4[1, c] + C2 * y2[1, c])))
            dby_const[c] += h * (np.conj(a1 * C1 * y[3, c] + am * C3 * y3[3, c] + a2 * (C4 * y4[3, c] + C2 * y2[3, c])))
            dbz_const[c] += (
                h * (np.conj(a1 * C1 * y[5, c] + am * C3 * y3[5, c] + a2 * (C4 * y4[5, c] + C2 * y2[5, c]))) * KAPPA
            )
            dbx_lin[c] += h * (
                np.conj(kz1 * a1 * C1 * y[1, c] + kzm * am * C3 * y3[1, c] + kz2 * a2 * (C4 * y4[1, c] + C2 * y2[1, c]))
            )
            dby_lin[c] += h * (
                np.conj(kz1 * a1 * C1 * y[3, c] + kzm * am * C3 * y3[3, c] + kz2 * a2 * (C4 * y4[3, c] + C2 * y2[3, c]))
            )
            dbz_lin[c] += (
                h
                * (
                    np.conj(
                        kz1 * a1 * C1 * y[5, c] + kzm * am * C3 * y3[5, c] + kz2 * a2 * (C4 * y4[5, c] + C2 * y2[5, c])
                    )
                )
                * KAPPA
            )
    else:  # j == COORD_S
        xm = x + 0.5 * h
        x2 = x + h
        a1 = get_phi(zeta0, x, cdivkL)
        am = get_phi(zeta0, xm, cdivkL)
        a2 = get_phi(zeta0, x2, cdivkL)

        for c in range(y.shape[1]):
            dbx_const[c] += h * np.conj(
                a1 * C1 * y[1, c] / x + am * C3 * y3[1, c] / xm + a2 * (C4 * y4[1, c] + C2 * y2[1, c]) / x2
            )
            dby_const[c] += h * np.conj(
                a1 * C1 * y[3, c] / x + am * C3 * y3[3, c] / xm + a2 * (C4 * y4[3, c] + C2 * y2[3, c]) / x2
            )
            dbz_const[c] += KAPPA * h * np.conj(C1 * y[5, c] + C3 * y3[5, c] + C4 * y4[5, c] + C2 * y2[5, c]) * KAPPA
            dbx_lin[c] += h * np.conj((a1 * C1 * y[1, c] + am * C3 * y3[1, c]) + a2 * (C4 * y4[1, c] + C2 * y2[1, c]))
            dby_lin[c] += h * np.conj((a1 * C1 * y[3, c] + am * C3 * y3[3, c]) + a2 * (C4 * y4[3, c] + C2 * y2[3, c]))
            dbz_lin[c] += (
                KAPPA
                * h
                * np.conj((x * C1 * y[5, c] + xm * C3 * y3[5, c]) + x2 * (C4 * y4[5, c] + a2 * C2 * y2[5, c]))
                * KAPPA
            )


@jit
def _rk2_into(
    m: ComplexArray, h: float, my: ComplexArray, y: ComplexArray, ym: ComplexArray, out: ComplexArray
) -> None:
    """RK2 step with the midpoint matrix ``m`` given: ``out = y + h * (m @ (y + h * my / 2))``; ``ym`` is scratch."""
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            ym[r, c] = y[r, c] + h * my[r, c] / 2
    _matmul(np.ascontiguousarray(m), ym, out)
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            out[r, c] = y[r, c] + h * out[r, c]


@jit
def _midpoint_step_into(
    y: ComplexArray,
    x: float,
    h: float,
    j: int,
    kz_x: float,
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
    work: ComplexArray,
    y_upper: ComplexArray,
    y_err: ComplexArray,
) -> float:
    """Modified midpoint step (see :func:`modified_midpoint_integration_step`) writing into ``y_upper``/``y_err``,
    with scratch ``work`` of shape ``(7, *y.shape)``. ``kz_x`` is ``kz`` at ``x`` (``t_to_s(x, ...)`` for COORD_T);
    returns ``kz`` at ``x + h`` for the next step, so each ``t_to_s`` is evaluated once."""
    my, ym, y2, y3, y4, m, m_mid = work[0], work[1], work[2], work[3], work[4], work[5], work[6]
    x_mid = x + h * 0.5
    x_q1 = x + h * 0.5 * 0.5
    x_q3 = x + h * 0.5 + h * 0.5 * 0.5
    x_2 = x + 1.0 * h
    if j == COORD_T:
        kz_mid = t_to_s(x_mid, zeta0, kz0, lastkz, psi0, cdivkL)
        kz_q1 = t_to_s(x_q1, zeta0, kz0, lastkz, psi0, cdivkL)
        kz_q3 = t_to_s(x_q3, zeta0, kz0, lastkz, psi0, cdivkL)
        kz_2 = t_to_s(x_2, zeta0, kz0, lastkz, psi0, cdivkL)
    else:
        kz_mid, kz_q1, kz_q3, kz_2 = x_mid, x_q1, x_q3, x_2
    getM_into(m, j, x, kz_x, kz0, psi0, zeta0, cdivkL, cosbeta, sinbeta)
    _matmul(m, y, my)
    getM_into(m_mid, j, x_mid, kz_mid, kz0, psi0, zeta0, cdivkL, cosbeta, sinbeta)
    _rk2_into(m_mid, h, my, y, ym, y2)
    getM_into(m, j, x_q1, kz_q1, kz0, psi0, zeta0, cdivkL, cosbeta, sinbeta)
    _rk2_into(m, h * 0.5, my, y, ym, y3)
    _matmul(m_mid, y3, my)
    getM_into(m, j, x_q3, kz_q3, kz0, psi0, zeta0, cdivkL, cosbeta, sinbeta)
    _rk2_into(m, h * 0.5, my, y3, ym, y4)
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            y_upper[r, c] = B1 * y4[r, c] + B2 * y2[r, c]
            y_err[r, c] = y_upper[r, c] - y4[r, c]
    _update_forcing(
        y, y2, y3, y4, x, h, j, kz0, lastkz, zeta0, cdivkL, psi0,
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin, kz_x, kz_mid, kz_2,
    )  # fmt: skip
    return kz_2


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
    See eq. (7)-(8) in Numerical Solution.

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
    y_upper = np.empty(y.shape, dtype=np.complex128)
    y_err = np.empty(y.shape, dtype=np.complex128)
    work = np.empty((7, *y.shape), dtype=np.complex128)
    kz_x = t_to_s(x, zeta0, kz0, lastkz, psi0, cdivkL) if j == COORD_T else x
    _midpoint_step_into(
        np.ascontiguousarray(y), x, h, j, kz_x, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta,
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin, work, y_upper, y_err,
    )  # fmt: skip
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
) -> tuple[ComplexArray, float, float, float]:
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
    # Preallocated state buffers, swapped after every step (the input y_lower is never written)
    work = np.empty((7, *y_lower.shape), dtype=np.complex128)
    kz_x = t_to_s(x, zeta0, kz0, lastkz, psi0, cdivkL) if j == COORD_T else x
    y_cur = y_lower.copy()
    y_upper = np.empty_like(y_cur)
    dy_err = np.empty_like(y_cur)
    while True:
        last = x + 1.1 * h > x2  # not (h + x) > x2 because this could give you numerical issues
        if last:
            # step, h, big enough to reach x2 -> take the final step
            h = x2 - x
        kz_x = _midpoint_step_into(
            y_cur,
            x,
            h,
            j,
            kz_x,
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
            work,
            y_upper,
            dy_err,
        )
        y_err += dy_err
        if last:
            h = get_new_h2(h, acc, dy_err, y_upper)
            return y_upper, h, log_s_upper, lastkz

        y_cur, y_upper = y_upper, y_cur
        x = x + h
        h = get_new_h2(h, acc, dy_err, y_cur)
        y_norm = np.linalg.norm(y_cur[:, 0]) / y_norm1
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

            return y_cur, h, log_s_lower, lastkz


@jit
def _store_node(
    nodes: tuple[Any, ...],
    y_lower: ComplexArray,
    r_lower: ComplexArray,
    db: ComplexArray,
    log_s_lower: float,
    log_s_upper: float,
    y_upper: ComplexArray,
) -> tuple[ComplexArray, ComplexArray]:
    """Append the finished node to ``nodes`` and return the QR-factored ``(y_lower, r_lower)`` of the next one."""
    ys, r_lowers, r_uppers, dbs, lowers, uppers = nodes
    ys.append(y_lower.copy())
    r_lowers.append(r_lower.copy())
    dbs.append(db.copy())
    lowers.append(log_s_lower)
    uppers.append(log_s_upper)
    y_next, r_next, r_upper = qr_decomposition(y_upper)
    r_uppers.append(r_upper.copy())
    return y_next, r_next


@jit
def _integrate_segment(
    y_lower: ComplexArray,
    r_lower: ComplexArray,
    db: ComplexArray,
    log_s_lower: float,
    log_s_upper: float,
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
    nodes: tuple[Any, ...],
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, float, float, float]:
    """Integrate one station segment, splitting off substations (appended to ``nodes``) like
    ``PreLUTGenerator.integrate_between_stations``. ``db`` rows are dbx/dby/dbz const then lin."""
    while True:
        y_upper, h, s2, lastkz = integrate_between_stations(
            y_lower,
            log_s_lower,
            log_s_upper,
            db[0],
            db[1],
            db[2],
            db[3],
            db[4],
            db[5],
            h,
            y_err,
            acc,
            j,
            kz0,
            lastkz,
            zeta0,
            cdivkL,
            psi0,
            cosbeta,
            sinbeta,
        )
        if s2 >= log_s_upper:
            return y_upper, y_lower, r_lower, db, log_s_lower, h, lastkz
        y_lower, r_lower = _store_node(nodes, y_lower, r_lower, db, log_s_lower, s2, y_upper)
        db = np.zeros_like(db)
        log_s_lower = s2


@jit(nogil=True)  # lets preLUTs be generated in threads
def integrate_all_stations(
    y_lower: ComplexArray,
    h: float,
    y_err: ComplexArray,
    acc: float,
    ds: float,
    log_s_from: NDArray[np.float64],
    log_s_to: NDArray[np.float64],
    s_tr: float,
    log_smaxx: float,
    max_nodes: int,
    kz0: float,
    lastkz: float,
    zeta0: float,
    cdivkL: float,
    psi0: float,
    cosbeta: float,
    sinbeta: float,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, NDArray[np.float64], NDArray[np.float64]]:
    """Run the station loop of ``PreLUTGenerator.make_prelut`` in one compiled call, with the same
    arithmetic as the node-by-node Python loop (which preludium-equivalent mode still uses).

    Returns stacked node data: ``Y_lower``, ``R_lower``, ``R_upper`` (``(n, N_EQ, N_EQ)``), the forcing
    accumulators ``(n, 6, N_EQ)`` (dbx/dby/dbz const, then lin) and ``log_s_lower``/``log_s_upper``."""
    n_eq = y_lower.shape[0]
    ys = [y_lower.copy()]
    r_lowers = [y_lower.copy()]
    r_uppers = [y_lower.copy()]
    dbs = [np.zeros((6, n_eq), dtype=np.complex128)]
    lowers = [0.0]
    uppers = [0.0]
    for lst in (ys, r_lowers, r_uppers, dbs):
        lst.pop()
    lowers.pop()
    uppers.pop()
    nodes = (ys, r_lowers, r_uppers, dbs, lowers, uppers)

    r_lower = np.eye(n_eq, dtype=np.complex128)
    db = np.zeros((6, n_eq), dtype=np.complex128)
    log_s_lower = 0.0
    log_s_upper = ds
    y_upper, y_lower, r_lower, db, log_s_lower, h, lastkz = _integrate_segment(
        y_lower, r_lower, db, log_s_lower, log_s_upper, h, y_err, acc, COORD_T,
        kz0, lastkz, zeta0, cdivkL, psi0, cosbeta, sinbeta, nodes,
    )  # fmt: skip
    completed = True
    for k in range(len(log_s_from)):
        y_lower, r_lower = _store_node(nodes, y_lower, r_lower, db, log_s_lower, log_s_upper, y_upper)
        db = np.zeros_like(db)
        log_s_lower = log_s_from[k]
        log_s_upper = log_s_to[k]
        j = COORD_T if log_s_lower < s_tr else COORD_S
        y_upper, y_lower, r_lower, db, log_s_lower, h, lastkz = _integrate_segment(
            y_lower, r_lower, db, log_s_lower, log_s_upper, h, y_err, acc, j,
            kz0, lastkz, zeta0, cdivkL, psi0, cosbeta, sinbeta, nodes,
        )  # fmt: skip
        if len(ys) > max_nodes:  # pragma: no cover
            completed = False
            break
    if completed and s_tr < log_smaxx:
        _store_node(nodes, y_lower, r_lower, db, log_s_lower, log_s_upper, y_upper)

    n = len(ys)
    y_out = np.empty((n, n_eq, n_eq), dtype=np.complex128)
    r_lower_out = np.empty_like(y_out)
    r_upper_out = np.empty_like(y_out)
    db_out = np.empty((n, 6, n_eq), dtype=np.complex128)
    lower_out = np.empty(n)
    upper_out = np.empty(n)
    for i in range(n):
        y_out[i] = ys[i]
        r_lower_out[i] = r_lowers[i]
        r_upper_out[i] = r_uppers[i]
        db_out[i] = dbs[i]
        lower_out[i] = lowers[i]
        upper_out[i] = uppers[i]
    return y_out, r_lower_out, r_upper_out, db_out, lower_out, upper_out
