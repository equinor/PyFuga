import numpy as np
from PyPreludium.constants import Cm1, Cm2
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


def GMRES(prev, next):
    # use params, only: mykind,n
    # use contr, only: Tprelutnode
    # use vector_functions, only: outer
    # implicit none

    # Modified Gram-Schmidt ortonormalization
    # Y, V, R and invR are nxn matrices
    # Columns of Y are linearly independent vectors (the input)
    # Columns of V form an orthonormal basis (V is unitary)
    # R and invR are lower triangular
    # invR is the inverse of R
    # Y=V R*  where R* is the conjugate transpose of R

    # type(Tprelutnode),pointer :: p
    # complex(mykind), dimension(n,n) :: B
    # real(mykind) aux
    # integer(4) i,j,k
    # real(mykind) norm

    # aux = np.linalg.norm(prev.Yright, axis=0)
    # next.dat.Yleft = Yleft = prev.Yright / aux
    # next.dat.Rleft = np.diag(aux)
    # for j in range(5):
    #     next.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j]) @ Yleft[:, j + 1:]
    #     Yleft[:, j + 1:] = Yleft[:, j + 1:] - (Yleft[:, j] * np.conj(Yleft[:, j])[:, na]) @ Yleft[:, j + 1:]

    next.Yleft = Yleft = prev.Yright
    next.Rleft = np.zeros_like(Yleft)
    for j in range(5):
        aux = np.linalg.norm(Yleft[:, j])
        Yleft[:, j] = Yleft[:, j] / aux
        next.Rleft[j, j] = aux
        next.Rleft[j, j + 1:] = np.conj(Yleft[:, j]) @ Yleft[:, j + 1:]
        Yleft[:, j + 1:] = Yleft[:, j + 1:] - \
            np.dot((Yleft[:, j] * np.conj(Yleft[:, j])[:, na]).T, Yleft[:, j + 1:])

    aux = np.linalg.norm(Yleft[:, -1])
    Yleft[:, -1] = Yleft[:, -1] / aux
    next.Rleft[-1, -1] = aux

    # B = np.zeros_like(Yleft)
    # for j in range(1, 6):
    #     B[:j, j] = next.dat.Rleft[:j, j] / next.dat.Rleft[j, j]
    B = np.triu(next.Rleft / np.diag(next.Rleft), 1)  # upper triangle without diagonal
    # prev.dat.Rright = np.diag(1 / np.diag(next.dat.Rleft))
    # for i in range(6):
    #     for j in range(i + 1, 6):
    #         for k in range(i, j):
    #             prev.dat.Rright[i, j] = prev.dat.Rright[i, j] - prev.dat.Rright[i, k] * B[k, j]

    prev.Rright = np.diag(1 / np.diag(next.Rleft))
    for i in range(6):
        for j in range(i + 1, 6):
            prev.Rright[i, j] -= np.sum(prev.Rright[i, i:j] * B[i:j, j])

    next.Rleft = np.conj(next.Rleft.T)
    prev.Rright = np.conj(prev.Rright.T)


def compare(A, n):
    return
    B = np.fromfile(n + '.dat', dtype=np.complex128).reshape(6, -1).T
    A = np.reshape(A, B.shape)

    def comp(a, b):
        aerr = np.abs(a - b)
        rerr = np.abs(aerr / np.mean([a, b], 0))
        rerr[aerr == 0] = 0
        assert np.abs(a[b == 0]).max() < 1e-50
        assert np.abs(a[b == 0]).max() < 1e-50
        rerr[(a == 0) | (b == 0)] = 0
        err = aerr
        i, j = np.unravel_index(np.argmax(err), b.shape)
        import numpy.testing as npt
        try:
            assert np.abs(rerr).max() < 1e-10
        except Exception:
            print(err.max(), (i, j))
            print("fortran", B[i, j])
            print("python", A[i, j])
            print()
            raise

    comp(A.real, B.real)
    comp(A.imag, B.imag)


def rel_err(A, B):
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
