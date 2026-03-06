"""Module: common.py

This module provides numerically optimized functions to compute stability-related parameters
for atmospheric modeling in PyFuga. It leverages JIT compilation (via the jit decorator)
to enhance performance in computational routines associated with stability functions.

Key Components:
- get_cdivkL(zeta0, kz0): Computes the normalized stability parameter (c/(k*L)) that adjusts
    according to the stability condition (neutral, unstable, or stable)
    using provided constants Cm1 and Cm2.
- get_phi_inverse(kz, cdivkL): Computes the inverse of the stability function phi_m for
    unstable conditions.
- get_psi(zeta0, kz, cdivkL): Calculates the stability function (-psi_m), using
    different formulations for unstable and stable atmospheric conditions.
- get_phi(zeta0, kz, cdivkL): Computes the stability function (phi_m) with behavior conditional
    on the sign of zeta0.
- complex_norm(arr): Computes the Euclidean norm across columns of a complex array.
- get_new_h2(h, acc, Yerr, Y): Adjusts the integration step size (h) based on the maximum
   relative error between two complex arrays, ensuring numerical stability and adherence to
   a predefined accuracy goal.

Each function is decorated with the jit decorator from pyfuga.utils to enable JIT compilation for improved
execution speed during numerical computations."""

import numpy as np

from pyfuga.utils import jit

from .constants import CM_STABLE, CM_UNSTABLE


@jit("double(double,double)")
def get_cdivkL(zeta0, kz0):
    """c/(k*L), c is a stability constant: depending on the stability (0, Cm1, Cm2)"""
    if abs(zeta0) < 1e-10:  # Neutral
        return 0.0
    else:
        if zeta0 < 0:  # Unstable
            return zeta0 / kz0 * CM_UNSTABLE
        else:  # Stable
            return zeta0 / kz0 * CM_STABLE


@jit("double(double,double)")
def get_phi_inverse(kz, cdivkl):
    """
    Inverse of stability function phi_m
    for unstable conditions
    phi_m=(1+Cm1*z/L)**(1/4)
    """
    return (1.0 + cdivkl * kz) ** 0.25


@jit("double(double,double,double)")
def get_psi(zeta0, kz, cdivkl):
    """Stability function -psi_m"""
    if zeta0 < 0:
        # Unstable: -psi_m=-ln(1/8*(1+phi_m**-2)*(1+phi_m**-1)**2)+2*atan(phi_m**-1)-pi/2

        aux = get_phi_inverse(kz, cdivkl)
        aux2 = (1.0 + aux) ** 2 * (1 + aux**2)
        return np.log(8.0 / aux2) + 2.0 * np.arctan(cdivkl * kz / aux2)
    else:  # Stable: -psi_m=Cm2*z/L
        return cdivkl * kz


@jit("double(double,double,double)")
def get_phi(zeta0, kz, cdivkl):
    """Stability function phi_m"""
    if zeta0 < 0:  # Unstable: phi_m=(1+Cm1*z/L)**(-1/4)
        return (1.0 + cdivkl * kz) ** (-0.25)
    else:  # Stable: phi_m=1+Cm2*z/L
        return 1.0 + cdivkl * kz


@jit("double[:](complex128[:,:],int32)")
def complex_norm(arr, axis=0):
    """Compute the norm of a complex array along the first axis"""
    return np.sqrt((np.abs(arr) ** 2).sum(0))


@jit("double(double,double,complex128[:,:],complex128[:,:])")
def get_new_h2(h, acc, Yerr, Y):
    """Get new step size based on error between integration estimates"""
    err = np.max(complex_norm(Yerr, axis=0) / complex_norm(Y, axis=0))
    return np.maximum(np.minimum(0.9 * h * (acc / err) ** (1 / 3), 4.0e-1), 1.0e-4)

    # return np.clip(0.9 * h * (acc / err)**(1 / 3), 1.0E-4, 4.0E-1)
