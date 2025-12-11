import importlib
import sys
import time

import numpy as np
import xarray as xr

try:
    from memory_profiler import profile
except ImportError:  # pragma: no cover - trivial no-op fallback
    # Fallback: define a no-op decorator so code can run without memory_profiler
    def profile(func):
        return func


from numba.core.decorators import njit

debugging = sys.gettrace() is not None
numba_jit = False  # and not debugging
jit_modules = set()
preludium_equivalent = False


def jit(*args, **kwargs):
    global jit_modules
    jit_modules.add(sys._getframe(1).f_globals["__name__"])
    if numba_jit:
        return njit(*args, cache=True, **kwargs)
    else:
        if isinstance(args[0], str):
            return lambda f: f
        else:
            # print(f'{args[0]} has no signature')
            return args[0]


def compile(jit=True):
    global numba_jit, jit_modules

    if numba_jit != jit:
        numba_jit = jit
        t = time.time()
        for m in [
            "pyfuga.common",
            "pyfuga.flut",
            "pyfuga.preluts_generator",
        ]:
            # if m in jit_modules:
            importlib.reload(sys.modules[m])
        if numba_jit:
            print(f"JIT compiled in {time.time() - t}s")


def get_beta(x):
    aj = 0.35  # &      ! parameter in beta table function
    aj2 = aj**2
    bj = 9.0  # ! parameter in beta table function
    bj2 = bj**2

    beta = (np.pi + bj * np.pi - x) / (1 + 2 * bj)  # ! first Newton-Raphson iteration after starting with beta=pi/2Z
    dbeta = [1]
    while any(np.abs(dbeta) > 1.0e-15):
        cosbeta2 = np.cos(beta) ** 2
        root = np.sqrt(aj2 + bj2 * cosbeta2)
        fj = beta - 2 * aj * bj * np.sin(beta) * np.cos(beta) / root
        dfj = 1 - 2.0 * aj * bj * (cosbeta2 * (2.0 * aj2 + bj2 * cosbeta2) - aj2) / root**3  # ! = dfj/dbeta
        dbeta = -(fj - x) / dfj
        beta = beta + dbeta
    return beta


def get_beta_lst(nbeta):
    return get_beta(np.pi * np.arange(nbeta + 1) / (2 * nbeta))


def get_kz0_lst(nkz0, kz0min=1e-9, kz0max=0.1):
    jmin = np.floor(0.5 + np.log10(kz0min) * nkz0)
    jmax = np.floor(0.5 + np.log10(kz0max) * nkz0)
    return 10 ** (np.arange(jmin, jmax + 1) / nkz0)


def save_complex(dataset, *args, **kwargs):
    ds = dataset.expand_dims("ReIm", axis=-1)  # Add ReIm axis at the end
    ds = xr.concat([ds.real, ds.imag], dim="ReIm")
    for k in dataset.data_vars:
        if np.iscomplexobj(dataset[k]) is False:
            ds[k] = dataset[k]
    return xr.Dataset.to_netcdf(ds, *args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    ds_new = ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)
    for k in ds:
        if "ReIm" not in ds[k].dims:
            ds_new[k] = ds[k]
    ds_new.attrs.update(ds.attrs)
    return ds_new


class ComplexXRDataset(xr.Dataset):
    __slots__ = []

    def to_netcdf(
        self,
        path=None,
        mode="w",
        format=None,
        group=None,
        engine=None,
        encoding=None,
        unlimited_dims=None,
        compute=True,
        invalid_netcdf=None,
        **kwargs,
    ):
        return save_complex(
            self,
            path,
            mode=mode,
            format=format,
            group=group,
            engine=engine,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            compute=compute,
            invalid_netcdf=invalid_netcdf,
            **kwargs,
        )

    def save(self, filename):
        return self.to_netcdf(filename)

    @classmethod
    def from_netcdf(cls, filename):
        ds = read_complex(filename)
        return cls(ds, attrs=ds.attrs)


def equal(A, n):  # pragma: no cover
    ref = np.fromfile(n + ".dat", dtype=np.complex128).reshape(6, -1).T
    a = np.reshape(A, ref.shape)

    err = np.abs(a - ref)

    try:
        assert np.all(a == ref)
    except Exception:
        i, j = np.unravel_index(np.argmax(err), ref.shape)
        print(f"Max error: {np.nanmax(err)} at {(i, j)}")
        print("fortran", ref[i, j])
        print("python ", a[i, j])
        print()
        raise


def compare(A, n, tol=1e-9):  # pragma: no cover
    B = np.fromfile(n + ".dat", dtype=np.complex128).reshape(6, -1).T
    A = np.reshape(A, B.shape)

    def comp(a, ref, real_imag, tol):
        aerr = np.abs(a - ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            rerr = np.abs(aerr / ref)
        rerr[(ref < tol**3)] = np.nan

        for err, abs_rel in [(aerr, "Abs"), (rerr, "Rel")]:
            try:
                assert np.nanmax(np.abs(err)) < tol
            except Exception:
                i, j = np.unravel_index(np.argmax(err), ref.shape)
                print(f"Max {abs_rel} error of {real_imag}: {np.nanmax(err)} at {(i, j)}")
                print("fortran", B[i, j])
                print("python", A[i, j])
                print()
                raise

    comp(A.real, B.real, "real", tol)
    comp(A.imag, B.imag, "imag", tol)


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


def mprof_tag(id):  # pragma: no cover
    try:

        def f():
            pass

        f.__name__ = id.replace(" ", "_")
        profile(f)()
    except Exception:
        pass
