import numpy as np
from PyPreludium.constants import Cm1, Cm2
import xarray as xr


def get_beta(x):
    aj = 0.35  # &      ! parameter in beta table function
    aj2 = aj**2
    bj = 9.0  # ! parameter in beta table function
    bj2 = bj**2

    beta = (np.pi + bj * np.pi - x) / (1 + 2 * bj)  # ! first Newton-Raphson iteration after starting with beta=pi/2Z
    dbeta = [1]
    while any(np.abs(dbeta) > 1.0e-15):
        cosbeta2 = np.cos(beta)**2
        root = np.sqrt(aj2 + bj2 * cosbeta2)
        fj = beta - 2 * aj * bj * np.sin(beta) * np.cos(beta) / root
        dfj = 1 - 2.0 * aj * bj * (cosbeta2 * (2.0 * aj2 + bj2 * cosbeta2) - aj2) / root ** 3  # ! = dfj/dbeta
        dbeta = -(fj - x) / dfj
        beta = beta + dbeta
    return beta


def cdivkL(zeta0, kz0):
    # c/(k*L), c is a stability constant: depending on the stability (0, Cm1, Cm2)

    if abs(zeta0) < 1e-10:  # Neutral
        return 0.0
    else:
        if zeta0 < 0:  # Unstable
            return zeta0 / kz0 * Cm1
        else:  # Stable
            return zeta0 / kz0 * Cm2


def dphiu(kz, cdivkl):
    # Inverse of stability function phi_m
    # for unstable conditions
    # ! phi_m=(1+Cm1*z/L)**(1/4)
    return (1.0 + cdivkl * kz)**0.25


def psi(zeta0, kz):

    # Stability function -psi_m
    if zeta0 < 0:  # Unstable: -psi_m=-ln(1/8*(1+phi_m**-2)*(1+phi_m**-1)**2)+2*atan(phi_m**-1)-pi/2
        raise NotImplementedError()
        # aux=dphiu(kz)
        # aux2=(1.0D0+aux)**2*(1+aux**2)
        # psi=log(8.0/aux2)+2.0D0*atan(cdivkl*kz/aux2)
    else:  # Stable: -psi_m=Cm2*z/L
        return cdivkL(zeta0, kz) * kz


def phi(zeta0, kz, cdivkl):
    # Stability function phi_m
    if zeta0 < 0:  # Unstable: phi_m=(1+Cm1*z/L)**(-1/4)
        return (1.0 + cdivkl * kz)**(-0.25)
    else:  # Stable: phi_m=1+Cm2*z/L
        return 1.0 + cdivkl * kz


def get_new_h2(h, acc, Yerr, Y):
    err = np.max(np.linalg.norm(Yerr, axis=0) / np.linalg.norm(Y, axis=0))
    return np.clip(0.9 * h * (acc * h / err)**(1 / 3), 1.0E-4, 4.0E-1)


def save_complex(dataset, *args, **kwargs):
    ds = dataset.expand_dims('ReIm', axis=-1)  # Add ReIm axis at the end
    ds = xr.concat([ds.real, ds.imag], dim='ReIm')
    return ds.to_netcdf(*args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)


def compare(A):
    B = np.fromfile('a.dat', dtype=np.complex128).reshape(6, -1).T
    A = np.reshape(A, B.shape)

    aerr = np.abs(A - B)
    rerr = np.abs(aerr / np.mean([A, B], 0))
    err = rerr
    i, j = np.unravel_index(np.argmax(err), B.shape)
    print(err.max(), (i, j))
    print("fortran", B[i, j])
    print("python", A[i, j])
    print()
