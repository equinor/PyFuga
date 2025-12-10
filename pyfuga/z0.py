import numpy as np
from scipy.optimize import fsolve
from scipy.special import lambertw

Cm1 = 5
Cm2 = -19.3


def phi(zeta):
    zeta = np.atleast_1d(zeta)
    return np.where(zeta <= 0, (1 + Cm2 * zeta) ** -0.25, 1 + Cm1 * zeta)


def psi(zeta, unstable=""):
    zeta = np.atleast_1d(zeta)
    ind = zeta < 0
    psi_n = np.zeros(zeta.shape)
    aux = phi(zeta) ** -1
    if unstable == "Wilson":
        psi_n[ind] = 3 * np.log(1 + (1 + 3.6 * np.abs(zeta[ind]) ** (2 / 3)) ** 0.5)
    else:
        aux2 = (1.0 + aux[ind]) ** 2 * (1 + aux[ind] ** 2)
        psi_n[ind] = -np.log(8.0 / aux2) - 2.0 * np.arctan(Cm2 * zeta[ind] / aux2)
    psi_n[~ind] = 1 - aux[~ind] ** -1
    psi_n[zeta == 0] = 0
    return psi_n


def z0_from_TI(TI, zref, zeta0, z0_limit=1e-5):
    TI = np.atleast_1d(TI)
    if zeta0 == 0:
        z0 = zref * np.exp(-1.0 / TI)  # Paul's formula
    elif zeta0 > 0:
        # analytical expression from residual for stable conditions:
        # 1/ti = np.log(zref/z0)+Cm1*zeta0(zref/z0-1)
        a = Cm1 * zeta0
        b = 1 / TI
        x = np.real(lambertw(a * np.exp(a + b))) / a
        z0 = zref / x
    else:
        # I had to solve it numerically for unstable conditions
        x = np.array(
            [fsolve(lambda x: x / np.exp(psi(x * zeta0) - psi(zeta0)) - np.exp(1 / ti), zref / 0.001)[0] for ti in TI]
        )
        z0 = zref / x
    z0[z0 < z0_limit] = z0_limit
    return z0
