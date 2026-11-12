"""utils.py
=========
Utility functions and decorators for PyFuga.

This module provides:
    - JIT compilation control via the :func:`jit` decorator (using Numba)
    - Timing and profiling decorators (:func:`timeit`)
    - Custom xarray dataset wrapper (:class:`ComplexXRDataset`)
    - Beta angle and wavenumber grid generation
    - File I/O helpers for reading/writing datasets

Also manages the global Preludium compatibility flag used for testing
alternative formulations of atmospheric code.
"""

import hashlib
import importlib
import importlib.metadata
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
from typing import Any, ParamSpec, TypedDict, TypeVar, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

try:
    from memory_profiler import profile
except ImportError:  # pragma: no cover - trivial no-op fallback
    # Fallback: define a no-op decorator so code can run without memory_profiler
    _F = TypeVar("_F", bound=Callable[..., Any])

    def profile(func: _F) -> _F:
        return func


from numba.core.decorators import njit  # type: ignore[import-untyped]

debugging = sys.gettrace() is not None
numba_jit = False  # and not debugging
jit_modules: set[str] = set()
preludium_equivalent = False

P = ParamSpec("P")
_R = TypeVar("_R")


@overload
def jit(signature: str, /, **kwargs: Any) -> Callable[[Callable[P, _R]], Callable[P, _R]]: ...


@overload
def jit(func: Callable[P, _R], /, **kwargs: Any) -> Callable[P, _R]: ...


@overload
def jit(**kwargs: Any) -> Callable[[Callable[P, _R]], Callable[P, _R]]: ...


def jit(*args: Any, **kwargs: Any) -> Any:
    global jit_modules
    jit_modules.add(sys._getframe(1).f_globals["__name__"])
    if numba_jit:
        return njit(*args, cache=True, **kwargs)
    if not args or isinstance(args[0], str):

        def _identity(func: Callable[P, _R]) -> Callable[P, _R]:
            return func

        return _identity
    # print(f'{args[0]} has no signature')
    return args[0]


def compile(jit: bool = True) -> None:
    global numba_jit, jit_modules

    if numba_jit != jit:
        numba_jit = jit
        t = time.time()
        # Dependencies first: a reload rebinds names imported from modules reloaded before it
        for m in [
            "pyfuga.common",
            "pyfuga.flut",
            "pyfuga.preluts_generator.transforms",
            "pyfuga.preluts_generator.matrices",
            "pyfuga.preluts_generator.nodes",
            "pyfuga.preluts_generator.integration",
            "pyfuga.preluts_generator.generator",
            "pyfuga.preluts_generator",
        ]:
            mod = sys.modules.get(m)
            if mod is not None:
                importlib.reload(mod)
        if numba_jit:
            print(f"JIT compiled in {time.time() - t}s")


def validate_n_cpu(n_cpu: int | None) -> None:
    """Raise ValueError if ``n_cpu`` is not a valid request (``None`` or ``int`` >= 1)."""
    if n_cpu is None:
        return
    if n_cpu < 1:
        raise ValueError(f"n_cpu must be None or >= 1, got {n_cpu}")


def normalise_n_cpu_workers(n_cpu: int | None) -> int:
    """Map ``None`` to a worker count (all CPUs) or pass through explicit ``int`` >= 1."""
    validate_n_cpu(n_cpu)
    if n_cpu is None:
        return max(1, os.cpu_count() or 1)
    return n_cpu


@contextmanager
def multiprocessing_map(
    workers: int, chunksize: int = 1
) -> Iterator[Callable[[Callable[..., Any], Iterable[Any]], Iterator[Any]]]:
    """Yield a map-like callable (serial ``map`` or ``spawn`` pool ``imap``)."""
    if workers == 1:
        yield map
        return

    ctx = mp.get_context("spawn")  # or "forkserver"
    with ctx.Pool(processes=workers) as pool:

        def pool_map(func: Callable[..., Any], items: Iterable[Any]) -> Iterator[Any]:
            return pool.imap(func, items, chunksize=chunksize)

        yield pool_map


@contextmanager
def threading_map(workers: int) -> Iterator[Callable[[Callable[..., Any], Iterable[Any]], Iterator[Any]]]:
    """Yield an ordered thread map with bounded in-flight work and no array serialization."""
    if workers == 1:
        yield map
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def thread_map(func: Callable[..., Any], items: Iterable[Any]) -> Iterator[Any]:
            iterator = iter(items)
            pending = deque(pool.submit(func, item) for item in islice(iterator, workers))
            while pending:
                result = pending.popleft().result()
                for item in islice(iterator, 1):
                    pending.append(pool.submit(func, item))
                yield result

        yield thread_map


class NCpuStageConfig(TypedDict, total=False):
    preluts: int | None
    fluts: int | None
    trafalgar: int | None


class ResolvedNCpuStageConfig(TypedDict):
    preluts: int
    fluts: int
    trafalgar: int


def resolve_n_cpu_stage(n_cpu: int | None, n_cpu_stage: NCpuStageConfig | None = None) -> ResolvedNCpuStageConfig:
    """Resolve per-stage workers with stage-aware defaults and optional global override.

    Default policy (when both ``n_cpu`` and ``n_cpu_stage`` are omitted/``None``):
        - ``preluts``: all CPUs
        - ``fluts``: all CPUs
        - ``trafalgar``: all CPUs

    ``n_cpu`` (when provided as an ``int`` >= 1) acts as a global override for all
    stages unless a specific stage is overridden via ``n_cpu_stage``.

    Precedence:
        1) Start from default policy above.
        2) If ``n_cpu`` is an integer, apply it to all stages.
        3) Apply ``n_cpu_stage`` overrides for provided stage keys.
    """
    validate_n_cpu(n_cpu)

    default_stage: dict[str, int | None] = {"preluts": None, "fluts": None, "trafalgar": None}

    if n_cpu_stage is None:
        merged_default: dict[str, int | None] = (
            default_stage if n_cpu is None else {"preluts": n_cpu, "fluts": n_cpu, "trafalgar": n_cpu}
        )
        return {
            "preluts": normalise_n_cpu_workers(merged_default["preluts"]),
            "fluts": normalise_n_cpu_workers(merged_default["fluts"]),
            "trafalgar": normalise_n_cpu_workers(merged_default["trafalgar"]),
        }

    allowed = {"preluts", "fluts", "trafalgar"}
    unknown = set(n_cpu_stage) - allowed
    if unknown:
        raise ValueError(f"n_cpu_stage has unknown stage keys: {sorted(unknown)}")

    merged: dict[str, int | None] = (
        default_stage.copy() if n_cpu is None else {"preluts": n_cpu, "fluts": n_cpu, "trafalgar": n_cpu}
    )
    for stage, stage_n_cpu in n_cpu_stage.items():
        if stage_n_cpu is not None and not isinstance(stage_n_cpu, int):
            raise ValueError(f"n_cpu_stage['{stage}'] must be an int or None, got {type(stage_n_cpu).__name__}")
        validate_n_cpu(stage_n_cpu)
        merged[stage] = stage_n_cpu
    return {
        "preluts": normalise_n_cpu_workers(merged["preluts"]),
        "fluts": normalise_n_cpu_workers(merged["fluts"]),
        "trafalgar": normalise_n_cpu_workers(merged["trafalgar"]),
    }


def get_beta(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=float)
    aj = 0.35  # &      ! parameter in beta table function
    aj2 = aj**2
    bj = 9.0  # ! parameter in beta table function
    bj2 = bj**2

    beta = (np.pi + bj * np.pi - x_arr) / (
        1 + 2 * bj
    )  # ! first Newton-Raphson iteration after starting with beta=pi/2Z
    dbeta: npt.NDArray[np.float64] = np.array([1.0])
    while any(np.abs(dbeta) > 1.0e-15):
        cosbeta2 = np.cos(beta) ** 2
        root = np.sqrt(aj2 + bj2 * cosbeta2)
        fj = beta - 2 * aj * bj * np.sin(beta) * np.cos(beta) / root
        dfj = 1 - 2.0 * aj * bj * (cosbeta2 * (2.0 * aj2 + bj2 * cosbeta2) - aj2) / root**3  # ! = dfj/dbeta
        dbeta = -(fj - x_arr) / dfj
        beta = beta + dbeta
    return beta


def get_beta_lst(nbeta: int) -> npt.NDArray[np.float64]:
    return get_beta(np.pi * np.arange(nbeta + 1) / (2 * nbeta))


def get_kz0_lst(nkz0: int, kz0min: float = 1e-9, kz0max: float = 0.1) -> npt.NDArray[np.float64]:
    jmin = np.floor(0.5 + np.log10(kz0min) * nkz0)
    jmax = np.floor(0.5 + np.log10(kz0max) * nkz0)
    return 10 ** (np.arange(jmin, jmax + 1) / nkz0)


def save_complex(dataset: xr.Dataset, *args: Any, **kwargs: Any) -> Any:
    ds = dataset.expand_dims("ReIm", axis=-1)  # Add ReIm axis at the end
    ds = xr.concat([ds.real, ds.imag], dim="ReIm")
    for k in dataset.data_vars:
        if np.iscomplexobj(dataset[k]) is False:
            ds[k] = dataset[k]
    return xr.Dataset.to_netcdf(ds, *args, **kwargs)


def read_complex(*args: Any, **kwargs: Any) -> xr.Dataset:
    ds = xr.open_dataset(*args, **kwargs)
    ds_new = ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)
    for k in ds:
        if "ReIm" not in ds[k].dims:
            ds_new[k] = ds[k]
    ds_new.attrs.update(ds.attrs)
    ds_typed: xr.Dataset = ds_new
    return ds_typed


def lut_schema_fingerprint(ds: xr.Dataset) -> str:
    """Return a short, stable hash of a dataset's *structure* (not its values).

    The fingerprint covers the sorted set of variable/coordinate names and their
    dimensions. Values and dtypes are deliberately excluded so that the fingerprint
    round-trips cleanly through the complex->ReIm split and NetCDF (de)serialisation:
    it is computed on the in-memory dataset at write time and recomputed on the
    reconstructed dataset at read time, and the two must match for an unchanged schema.

    A structural change (added, removed or renamed variable/dimension) flips the
    fingerprint, which lets stale caches be rejected even if someone forgot to bump
    :data:`pyfuga.constants.LUT_FORMAT_VERSION`.
    """
    sig = repr(sorted((str(name), tuple(map(str, var.dims))) for name, var in ds.variables.items()))
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def _pyfuga_version() -> str:
    """Installed pyfuga version, read lazily to avoid a circular import with ``__init__``."""
    try:
        return importlib.metadata.version("pyfuga")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def stamp_lut_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Stamp cache-validation metadata onto ``ds`` in place and return it.

    Adds ``lut_format_version`` (compatibility key), ``lut_schema_fingerprint``
    (structural safety net) and ``pyfuga_version`` (provenance only).
    """
    from pyfuga.constants import LUT_FORMAT_VERSION

    ds.attrs["lut_format_version"] = LUT_FORMAT_VERSION
    ds.attrs["lut_schema_fingerprint"] = lut_schema_fingerprint(ds)
    ds.attrs["pyfuga_version"] = _pyfuga_version()
    return ds


def cache_is_current(ds: xr.Dataset) -> bool:
    """True if a loaded cached LUT carries compatible version metadata.

    Rejects legacy caches (which lack the attributes entirely), caches from an
    incompatible :data:`LUT_FORMAT_VERSION`, and caches whose stored structural
    fingerprint no longer matches their actual structure.
    """
    from pyfuga.constants import LUT_FORMAT_VERSION

    if ds.attrs.get("lut_format_version") != LUT_FORMAT_VERSION:
        return False
    return ds.attrs.get("lut_schema_fingerprint") == lut_schema_fingerprint(ds)


def load_lut_if_current(ds: xr.Dataset, path: Any) -> bool:
    """Warn (and return ``False``) when a cached LUT is incompatible and must be regenerated."""
    if cache_is_current(ds):
        return True
    warnings.warn(
        f"Ignoring incompatible cached LUT '{getattr(path, 'name', path)}' "
        f"(missing or mismatched version metadata); regenerating.",
        UserWarning,
        stacklevel=2,
    )
    return False


class ComplexXRDataset(xr.Dataset):  # type: ignore[no-untyped-call]
    __slots__: list[str] = []

    def to_netcdf(  # type: ignore[override]
        self,
        path: Any = None,
        mode: Any = "w",
        format: Any = None,
        group: Any = None,
        engine: Any = None,
        encoding: Any = None,
        unlimited_dims: Any = None,
        compute: Any = True,
        invalid_netcdf: Any = None,
        **kwargs: Any,
    ) -> Any:
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

    def save(self, filename: Any) -> Any:
        return self.to_netcdf(filename)

    @classmethod
    def from_netcdf(cls, filename: Any) -> "ComplexXRDataset":
        ds = read_complex(filename)
        # xarray no longer accepts passing a Dataset instance as `data_vars`
        # to Dataset subclass constructors. Rebuild explicitly from components.
        return cls(ds.data_vars, coords=ds.coords, attrs=ds.attrs)


def equal(A: Any, n: str) -> None:  # pragma: no cover
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


def compare(A: Any, n: str, tol: float = 1e-9) -> None:  # pragma: no cover
    B = np.fromfile(n + ".dat", dtype=np.complex128).reshape(6, -1).T
    A = np.reshape(A, B.shape)

    def comp(a: npt.NDArray[np.float64], ref: npt.NDArray[np.float64], real_imag: str, tol: float) -> None:
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


def rel_err(A: Any, B: Any) -> None:  # pragma: no cover
    def comp(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> None:
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


def mprof_tag(id: str) -> None:  # pragma: no cover
    try:

        def f() -> None:
            pass

        f.__name__ = id.replace(" ", "_")
        profile(f)()
    except Exception:
        pass
