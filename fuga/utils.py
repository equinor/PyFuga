import numpy as np
from .constants import Cm1, Cm2
import xarray as xr
from numpy import newaxis as na


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


def psi(zeta0, kz, cdivkl):

    # Stability function -psi_m
    if zeta0 < 0:
        # Unstable: -psi_m=-ln(1/8*(1+phi_m**-2)*(1+phi_m**-1)**2)+2*atan(phi_m**-1)-pi/2

        aux = dphiu(kz, cdivkl)
        aux2 = (1.0 + aux)**2 * (1 + aux**2)
        return np.log(8.0 / aux2) + 2.0 * np.arctan(cdivkl * kz / aux2)
    else:  # Stable: -psi_m=Cm2*z/L
        return cdivkL(zeta0, kz) * kz


# def phi(zeta0, kz, cdivkl):
#     # Stability function phi_m
#     if zeta0 < 0:  # Unstable: phi_m=(1+Cm1*z/L)**(-1/4)
#         return (1.0 + cdivkl * kz)**(-0.25)
#     else:  # Stable: phi_m=1+Cm2*z/L
#         return 1.0 + cdivkl * kz


def get_new_h2(h, acc, Yerr, Y):
    err = np.max(np.linalg.norm(Yerr, axis=0) / np.linalg.norm(Y, axis=0))
    return np.clip(0.9 * h * (acc * h / err)**(1 / 3), 1.0E-4, 4.0E-1)


def save_complex(dataset, *args, **kwargs):
    ds = dataset.expand_dims('ReIm', axis=-1)  # Add ReIm axis at the end
    ds = xr.concat([ds.real, ds.imag], dim='ReIm')
    for k in dataset.data_vars:
        if np.iscomplexobj(dataset[k]) is False:
            ds[k] = dataset[k]
    return ds.to_netcdf(*args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    ds_new = ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)
    for k in ds:
        if 'ReIm' not in ds[k].dims:
            ds_new[k] = ds[k]
    ds_new.attrs.update(ds.attrs)
    return ds_new


def equal(A, n):  # pragma: no cover
    ref = np.fromfile(n + '.dat', dtype=np.complex128).reshape(6, -1).T
    a = np.reshape(A, ref.shape)

    err = np.abs(a - ref)

    try:
        assert np.all(a == ref)
    except Exception:
        i, j = np.unravel_index(np.argmax(err), ref.shape)
        print(f'Max error: {np.nanmax(err)} at {(i, j)}')
        print("fortran", ref[i, j])
        print("python ", a[i, j])
        print()
        raise


def compare(A, n, tol=1e-9):  # pragma: no cover
    B = np.fromfile(n + '.dat', dtype=np.complex128).reshape(6, -1).T
    A = np.reshape(A, B.shape)

    def comp(a, ref, real_imag, tol):
        aerr = np.abs(a - ref)
        with np.errstate(divide='ignore', invalid='ignore'):
            rerr = np.abs(aerr / ref)
        rerr[(ref < tol**3)] = np.nan

        for err, abs_rel in [(aerr, 'Abs'), (rerr, 'Rel')]:
            try:
                assert np.nanmax(np.abs(err)) < tol
            except Exception:
                i, j = np.unravel_index(np.argmax(err), ref.shape)
                print(f'Max {abs_rel} error of {real_imag}: {np.nanmax(err)} at {(i, j)}')
                print("fortran", B[i, j])
                print("python", A[i, j])
                print()
                raise

    comp(A.real, B.real, 'real', tol)
    comp(A.imag, B.imag, 'imag', tol)


def rel_err(A, B):  # pragma: no cover
    def comp(a, b):
        aerr = np.abs(a - b)
        rerr = np.abs(aerr / np.mean([a, b], 0))
        rerr[aerr == 0] = 0
        # if np.sum(b == 0):
        #     assert np.abs(a[b == 0]).max() < 1e-50
        # if np.sum(a == 0):
        #     assert np.abs(b[a == 0]).max() < 1e-50
        rerr[(a == 0) | (b == 0)] = 0

        i, j = np.unravel_index(np.argmax(aerr), b.shape)
        print(aerr.max(), (i, j))

        i, j = np.unravel_index(np.argmax(rerr), b.shape)
        print(rerr.max(), (i, j))

    comp(A.real, B.real)
    comp(A.imag, B.imag)
