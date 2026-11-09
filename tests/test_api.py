from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import pyfuga
import pyfuga._fuga as _fuga
from pyfuga import get_luts
from pyfuga.paths import get_fluts_path, get_luts_path, get_preluts_path
from pyfuga.utils import normalise_n_cpu_workers, resolve_n_cpu_stage, validate_n_cpu


@pytest.mark.parametrize("name", ["get_luts", "get_fluts", "get_preluts"])
def test_wrapper_delegates_to__fuga(monkeypatch, name: str):
    sentinel = object()

    def fake(*args, **kwargs):
        # return something distinctive so we can assert delegation
        return (sentinel, args, kwargs)

    monkeypatch.setattr(_fuga, name, fake)

    out = getattr(pyfuga, name)(1, 2, foo="bar")

    assert out[0] is sentinel
    assert out[1] == (1, 2)
    assert out[2] == {"foo": "bar"}


def test___version___falls_back_when_package_metadata_missing(monkeypatch):
    def raise_not_found(_dist_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError("pyfuga")

    with monkeypatch.context() as m:
        m.setattr(importlib_metadata, "version", raise_not_found)
        reloaded = importlib.reload(pyfuga)
        assert reloaded.__version__ == "0+unknown"

    importlib.reload(pyfuga)


def test___version___uses_metadata_when_available(monkeypatch):
    expected_version = "9.9.9-test"

    with monkeypatch.context() as m:
        m.setattr(importlib_metadata, "version", lambda _dist_name: expected_version)
        reloaded = importlib.reload(pyfuga)
        assert reloaded.__version__ == expected_version

    importlib.reload(pyfuga)


def test_compile_jit_does_not_keyerror():
    from pyfuga import utils

    utils.compile(jit=True)


def test_compile_recompiles_preluts_generator_submodules():
    import sys

    from pyfuga import utils

    utils.compile(jit=False)
    import pyfuga.preluts_generator.generator  # noqa: F401  (imported uncompiled)

    utils.compile(jit=True)
    integration = sys.modules["pyfuga.preluts_generator.integration"]
    generator = sys.modules["pyfuga.preluts_generator.generator"]
    assert hasattr(integration.integrate_all_stations, "py_func")
    assert generator.integrate_all_stations is integration.integrate_all_stations


@pytest.mark.local
def test_get_luts_public_api_smoke(tmp_path: Path):
    out_dir = tmp_path / "luts"

    luts = get_luts(
        folder=str(out_dir),
        zeta0=0,
        nkz0=2,
        nbeta=8,
        diameter=80,
        zhub=70,
        z0=1e-5,
        zi=400,
        zlow=70,
        zhigh=70,
        lut_vars=["UL"],
        nx=256,
        ny=64,
        dx=None,
        dy=None,
        jit=True,
        n_cpu=1,
    )

    # Basic "public API returned something" checks
    assert luts is not None

    if hasattr(luts, "to_netcdf") or hasattr(luts, "data_vars"):
        if hasattr(luts, "data_vars"):
            assert "UL" in luts.data_vars or "UL" in luts
    else:
        if isinstance(luts, dict):
            assert "UL" in luts or len(luts) > 0

    # Check outputs were actually written
    assert out_dir.exists()
    produced = list(out_dir.rglob("*"))
    assert len(produced) > 0, "Expected some output files to be produced"

    # Cheap numeric sanity tests
    if hasattr(luts, "to_array"):
        arr = luts.to_array().values
        assert np.isfinite(arr).any(), "Expected some finite values in LUTs"


@pytest.mark.local
def test_get_luts_orchestration_and_caching(tmp_path: Path):
    out_dir = tmp_path / "luts"
    params = {
        "folder": out_dir,
        "zeta0": 0,
        "nkz0": 2,
        "nbeta": 8,
        "diameter": 80,
        "zhub": 70,
        "z0": 1e-5,
        "zi": 400,
        "zlow": 70,
        "zhigh": 70,
        "lut_vars": ["UL"],
        "nx": 256,
        "ny": 64,
        "dx": 20.0,
        "dy": 5.0,
    }

    luts = get_luts(**params, jit=True, n_cpu=1, cache_intermediate=True)

    luts_path = get_luts_path(**params)
    assert luts_path.exists()
    assert luts.attrs["name"] == luts_path.stem

    assert get_preluts_path(
        folder=params["folder"], zeta0=params["zeta0"], nkz0=params["nkz0"], nbeta=params["nbeta"]
    ).exists()
    assert get_fluts_path(**{k: v for k, v in params.items() if k not in ("nx", "ny", "dx", "dy")}).exists()

    before = {p: p.stat().st_mtime for p in out_dir.glob("*.nc")}
    luts2 = get_luts(**params, jit=True, n_cpu=1)
    after = {p: p.stat().st_mtime for p in out_dir.glob("*.nc")}

    assert before == after, "cache files were rewritten on second call"
    xr.testing.assert_identical(luts, luts2)


def _rewrite_attrs(path: Path, mutate) -> None:
    """Load a cached LUT, apply ``mutate(attrs)`` and write it back to the same path."""
    ds = xr.load_dataset(path)
    mutate(ds.attrs)
    ds.to_netcdf(path)


@pytest.mark.local
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda a: [a.pop(k, None) for k in ("lut_format_version", "lut_schema_fingerprint")], id="legacy"),
        pytest.param(lambda a: a.__setitem__("lut_format_version", -1), id="wrong-version"),
        pytest.param(lambda a: a.__setitem__("lut_schema_fingerprint", "deadbeef0000"), id="bad-fingerprint"),
    ],
)
def test_get_luts_regenerates_incompatible_cache(tmp_path: Path, mutate) -> None:
    """A final LUT cache with missing/incompatible version metadata is rejected and regenerated."""
    out_dir = tmp_path / "luts"
    params = {
        "folder": out_dir,
        "zeta0": 0,
        "nkz0": 2,
        "nbeta": 8,
        "diameter": 80,
        "zhub": 70,
        "z0": 1e-5,
        "zi": 400,
        "zlow": 70,
        "zhigh": 70,
        "lut_vars": ["UL"],
        "nx": 256,
        "ny": 64,
        "dx": 20.0,
        "dy": 5.0,
    }
    get_luts(**params, jit=True, n_cpu=1, cache_intermediate=True)
    luts_path = get_luts_path(**params)

    # Corrupt only the final LUT cache; the valid fLUTs/preLUTs caches make regeneration cheap.
    _rewrite_attrs(luts_path, mutate)

    with pytest.warns(UserWarning, match="incompatible cached LUT"):
        luts = get_luts(**params, jit=True, n_cpu=1)

    from pyfuga.constants import LUT_FORMAT_VERSION
    from pyfuga.utils import cache_is_current

    # The file on disk was rewritten with valid, self-consistent metadata.
    assert luts.attrs["lut_format_version"] == LUT_FORMAT_VERSION
    reloaded = xr.load_dataset(luts_path)
    assert cache_is_current(reloaded), "incompatible cache was not regenerated with valid metadata"
    assert reloaded.attrs["pyfuga_version"] == pyfuga.__version__


@pytest.mark.local
def test_get_luts_stamps_version_metadata(tmp_path: Path) -> None:
    """Every freshly generated cache tier carries the three version attributes."""
    out_dir = tmp_path / "luts"
    params = {
        "folder": out_dir,
        "zeta0": 0,
        "nkz0": 2,
        "nbeta": 8,
        "diameter": 80,
        "zhub": 70,
        "z0": 1e-5,
        "zi": 400,
        "zlow": 70,
        "zhigh": 70,
        "lut_vars": ["UL"],
        "nx": 256,
        "ny": 64,
        "dx": 20.0,
        "dy": 5.0,
    }
    from pyfuga.constants import LUT_FORMAT_VERSION

    get_luts(**params, jit=True, n_cpu=1, cache_intermediate=True)
    for nc in out_dir.glob("*.nc"):
        attrs = xr.open_dataset(nc).attrs
        assert attrs.get("lut_format_version") == LUT_FORMAT_VERSION, nc.name
        assert "lut_schema_fingerprint" in attrs, nc.name
        assert attrs.get("pyfuga_version") == pyfuga.__version__, nc.name


def test_validate_n_cpu_rejects_non_positive() -> None:
    validate_n_cpu(None)
    with pytest.raises(ValueError, match="None or >= 1"):
        validate_n_cpu(0)
    with pytest.raises(ValueError, match="None or >= 1"):
        validate_n_cpu(-3)


def test_normalise_n_cpu_workers_none_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 11)
    assert normalise_n_cpu_workers(None) == 11


def test_normalise_n_cpu_workers_none_falls_back_when_cpu_count_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: None)
    assert normalise_n_cpu_workers(None) == 1


def test_resolve_n_cpu_stage_defaults_to_global() -> None:
    assert resolve_n_cpu_stage(3, None) == {"preluts": 3, "fluts": 3, "trafalgar": 3}


def test_resolve_n_cpu_stage_defaults_none_to_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 7)
    assert resolve_n_cpu_stage(None, None) == {"preluts": 7, "fluts": 7, "trafalgar": 7}


def test_resolve_n_cpu_stage_applies_overrides() -> None:
    assert resolve_n_cpu_stage(4, {"trafalgar": 1, "preluts": 2}) == {"preluts": 2, "fluts": 4, "trafalgar": 1}


def test_resolve_n_cpu_stage_none_override_for_one_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 6)
    assert resolve_n_cpu_stage(2, {"trafalgar": None}) == {"preluts": 2, "fluts": 2, "trafalgar": 6}


def test_resolve_n_cpu_stage_stage_aware_defaults_with_partial_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 5)
    assert resolve_n_cpu_stage(None, {"trafalgar": 2}) == {"preluts": 5, "fluts": 5, "trafalgar": 2}


def test_resolve_n_cpu_stage_rejects_unknown_or_invalid() -> None:
    with pytest.raises(ValueError, match="unknown stage keys"):
        resolve_n_cpu_stage(1, {"stage3": 2})  # type: ignore[typeddict-unknown-key]
    with pytest.raises(ValueError, match="None or >= 1"):
        resolve_n_cpu_stage(1, {"fluts": 0})
    with pytest.raises(ValueError, match="int or None"):
        resolve_n_cpu_stage(1, {"fluts": 1.5})  # type: ignore[arg-type]


def test_get_luts_rejects_invalid_n_cpu(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="None or >= 1"):
        get_luts(
            folder=tmp_path,
            zeta0=0,
            nkz0=2,
            nbeta=8,
            diameter=80,
            zhub=70,
            z0=1e-5,
            zi=400,
            zlow=70,
            zhigh=70,
            lut_vars=["UL"],
            nx=64,
            ny=32,
            n_cpu=0,
        )


@pytest.mark.parametrize("passed_n_cpu, expected", [(4, 4), (1, 1)])
def test_get_luts_forwards_n_cpu_to_trafalgar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed_n_cpu: int, expected: int
) -> None:
    recorded: list[int] = []

    class FakeTrafalgar:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def make_luts(self, *, interpolation_order: int = 0, n_cpu: int = -1, **kwargs: object) -> xr.Dataset:
            recorded.append(n_cpu)
            return xr.Dataset()

    monkeypatch.setattr(_fuga, "Trafalgar", FakeTrafalgar)
    monkeypatch.setattr(_fuga, "get_fluts", lambda *args, **kwargs: object())

    base: dict[str, Any] = {
        "folder": tmp_path,
        "zeta0": 0,
        "nkz0": 2,
        "nbeta": 8,
        "diameter": 80,
        "zhub": 70,
        "z0": 1e-5,
        "zi": 400,
        "zlow": 70,
        "zhigh": 70,
        "lut_vars": ["UL"],
        "nx": 64,
        "ny": 32,
    }
    base["n_cpu"] = passed_n_cpu
    _fuga.get_luts(**base)
    assert recorded == [expected]


def test_get_luts_default_n_cpu_none_forwards_all_cpu_to_trafalgar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 9)
    recorded: list[int] = []

    class FakeTrafalgar:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def make_luts(self, *, interpolation_order: int = 0, n_cpu: int = -1, **kwargs: object) -> xr.Dataset:
            recorded.append(n_cpu)
            return xr.Dataset()

    monkeypatch.setattr(_fuga, "Trafalgar", FakeTrafalgar)
    monkeypatch.setattr(_fuga, "get_fluts", lambda *args, **kwargs: object())

    _fuga.get_luts(
        folder=tmp_path,
        zeta0=0,
        nkz0=2,
        nbeta=8,
        diameter=80,
        zhub=70,
        z0=1e-5,
        zi=400,
        zlow=70,
        zhigh=70,
        lut_vars=["UL"],
        nx=64,
        ny=32,
        n_cpu=None,
    )
    assert recorded == [9]


def test_get_luts_forwards_stage_specific_n_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    class FakeTrafalgar:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def make_luts(self, *, interpolation_order: int = 0, n_cpu: int = -1, **kwargs: object) -> xr.Dataset:
            calls["trafalgar"] = n_cpu
            return xr.Dataset()

    def fake_get_fluts(*args: object, **kwargs: object) -> object:
        calls["fluts"] = int(args[12])  # n_cpu positional arg in get_fluts call
        calls["preluts"] = int(kwargs["n_cpu_stage"]["preluts"])  # forwarded from get_luts
        return object()

    monkeypatch.setattr(_fuga, "Trafalgar", FakeTrafalgar)
    monkeypatch.setattr(_fuga, "get_fluts", fake_get_fluts)

    _fuga.get_luts(
        folder=tmp_path,
        zeta0=0,
        nkz0=2,
        nbeta=8,
        diameter=80,
        zhub=70,
        z0=1e-5,
        zi=400,
        zlow=70,
        zhigh=70,
        lut_vars=["UL"],
        nx=64,
        ny=32,
        n_cpu=4,
        n_cpu_stage={"preluts": 2, "fluts": 3, "trafalgar": 1},
    )
    assert calls == {"preluts": 2, "fluts": 3, "trafalgar": 1}


def test_get_luts_defaults_to_parallel_preluts_fluts_and_trafalgar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 8)
    calls: dict[str, int] = {}

    class FakeTrafalgar:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def make_luts(self, *, interpolation_order: int = 0, n_cpu: int = -1, **kwargs: object) -> xr.Dataset:
            calls["trafalgar"] = n_cpu
            return xr.Dataset()

    def fake_get_fluts(*args: object, **kwargs: object) -> object:
        calls["fluts"] = int(args[12])  # n_cpu positional arg in get_fluts call
        calls["preluts"] = int(kwargs["n_cpu_stage"]["preluts"])
        return object()

    monkeypatch.setattr(_fuga, "Trafalgar", FakeTrafalgar)
    monkeypatch.setattr(_fuga, "get_fluts", fake_get_fluts)

    _fuga.get_luts(
        folder=tmp_path,
        zeta0=0,
        nkz0=2,
        nbeta=8,
        diameter=80,
        zhub=70,
        z0=1e-5,
        zi=400,
        zlow=70,
        zhigh=70,
        lut_vars=["UL"],
        nx=64,
        ny=32,
    )
    assert calls == {"preluts": 8, "fluts": 8, "trafalgar": 8}


def test_get_fluts_forwards_stage_specific_n_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    class FakeFluts:
        # Minimal xarray-like surface touched by utils.stamp_lut_metadata().
        attrs: dict[str, object] = {}
        variables: dict[str, object] = {}

        def save(self, filename: object) -> None:
            Path(filename).touch()

    class FakeGenerator:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def make_hubheight_luts(self, *args: object, **kwargs: object) -> FakeFluts:
            calls["fluts"] = int(kwargs["n_cpu"])
            return FakeFluts()

    def fake_get_preluts(*args: object, **kwargs: object) -> object:
        calls["preluts"] = int(args[5])  # n_cpu positional arg in get_preluts call
        return object()

    monkeypatch.setattr(_fuga, "get_preluts", fake_get_preluts)
    monkeypatch.setattr(_fuga, "FourierLUTGenerator", FakeGenerator)
    monkeypatch.setattr(_fuga, "get_level_range", lambda *a, **k: (9999, 9999))

    _fuga.get_fluts(
        folder=tmp_path,
        zeta0=0,
        nkz0=2,
        nbeta=8,
        diameter=80,
        zhub=70,
        z0=1e-5,
        zi=400,
        zlow=70,
        zhigh=70,
        lut_vars=["UL"],
        n_cpu=4,
        n_cpu_stage={"preluts": 2, "fluts": 3},
    )
    assert calls == {"preluts": 2, "fluts": 3}


class _FakeLutFile:
    # Minimal xarray-like surface touched by utils.stamp_lut_metadata() and save().
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self.variables: dict[str, object] = {}

    def save(self, filename: object) -> None:
        Path(filename).touch()


_LUT_PARAMS: dict[str, Any] = {
    "zeta0": 0,
    "nkz0": 2,
    "nbeta": 8,
    "diameter": 80,
    "zhub": 70,
    "z0": 1e-5,
    "zi": 400,
    "zlow": 70,
    "zhigh": 70,
    "lut_vars": ["UL"],
}


@pytest.mark.parametrize(
    "verbose_kwargs, expected", [({}, True), ({"verbose": True}, True), ({"verbose": False}, False)]
)
def test_get_luts_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verbose_kwargs: dict[str, bool], expected: bool
) -> None:
    calls: dict[str, object] = {}

    class FakeTrafalgar:
        def __init__(self, *a: object, **k: object) -> None:
            calls["trafalgar"] = k["verbose"]

        def make_luts(self, **kwargs: object) -> xr.Dataset:
            return xr.Dataset()

    def fake_get_fluts(*args: object, **kwargs: object) -> object:
        calls["fluts"] = kwargs["verbose"]
        return object()

    monkeypatch.setattr(_fuga, "Trafalgar", FakeTrafalgar)
    monkeypatch.setattr(_fuga, "get_fluts", fake_get_fluts)

    _fuga.get_luts(folder=tmp_path, **_LUT_PARAMS, nx=64, ny=32, **verbose_kwargs)
    assert calls == {"fluts": expected, "trafalgar": expected}


@pytest.mark.parametrize(
    "verbose_kwargs, expected", [({}, True), ({"verbose": True}, True), ({"verbose": False}, False)]
)
def test_get_fluts_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verbose_kwargs: dict[str, bool], expected: bool
) -> None:
    calls: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *a: object, **k: object) -> None:
            calls["fluts"] = k["verbose"]

        def make_hubheight_luts(self, *args: object, **kwargs: object) -> _FakeLutFile:
            return _FakeLutFile()

    def fake_get_preluts(*args: object, **kwargs: object) -> object:
        calls["preluts"] = kwargs["verbose"]
        return object()

    monkeypatch.setattr(_fuga, "get_preluts", fake_get_preluts)
    monkeypatch.setattr(_fuga, "FourierLUTGenerator", FakeGenerator)
    monkeypatch.setattr(_fuga, "get_level_range", lambda *a, **k: (9999, 9999))

    _fuga.get_fluts(folder=tmp_path, **_LUT_PARAMS, **verbose_kwargs)
    assert calls == {"preluts": expected, "fluts": expected}


@pytest.mark.parametrize(
    "verbose_kwargs, expected", [({}, True), ({"verbose": True}, True), ({"verbose": False}, False)]
)
def test_get_preluts_forwards_verbose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verbose_kwargs: dict[str, bool], expected: bool
) -> None:
    calls: dict[str, object] = {}

    class FakePreLUTs:
        @staticmethod
        def make_preluts(**kwargs: object) -> _FakeLutFile:
            calls["preluts"] = kwargs["verbose"]
            return _FakeLutFile()

    monkeypatch.setattr(_fuga, "PreLUTs", FakePreLUTs)

    _fuga.get_preluts(folder=tmp_path, zeta0=0, nkz0=2, nbeta=8, **verbose_kwargs)
    assert calls == {"preluts": expected}


@pytest.mark.parametrize(
    "zeta0, heights, n_cpu",
    [(zeta0, heights, 1) for zeta0 in [0, -1e-6, 1e-6] for heights in [(70, 70), (30, 110)]] + [(0, (70, 70), 2)],
)
def test_final_only_pipeline_matches_cached_pipeline(
    tmp_path: Path, zeta0: float, heights: tuple[int, int], n_cpu: int
) -> None:
    params = {
        **_LUT_PARAMS,
        "zeta0": zeta0,
        "nkz0": 1,
        "nbeta": 4,
        "zlow": heights[0],
        "zhigh": heights[1],
        "lut_vars": ["UL", "UT", "VL", "VT", "WL", "WT", "PL", "PT"],
        "nx": 64,
        "ny": 32,
        "n_cpu": n_cpu,
        "verbose": False,
    }
    cached = get_luts(folder=tmp_path / "cached", **params, cache_intermediate=True)
    final = get_luts(folder=tmp_path / "final", **params)
    xr.testing.assert_identical(cached, final)
    files = list((tmp_path / "final").glob("*.nc"))
    assert len(files) == 1
    xr.testing.assert_identical(xr.load_dataset(files[0]).assign_attrs(name=final.attrs["name"]), final)
    xr.testing.assert_identical(get_luts(folder=tmp_path / "final", **params), final)


def test_final_only_reuses_existing_fourier_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    params = {**_LUT_PARAMS, "nkz0": 1, "nbeta": 4, "n_cpu": 1, "verbose": False}
    _fuga.get_fluts(folder=tmp_path, **params)

    def fail(*args, **kwargs):
        raise AssertionError("Existing Fourier cache should avoid prelut generation")

    monkeypatch.setattr(_fuga, "get_preluts", fail)
    get_luts(folder=tmp_path, **params, nx=64, ny=32)


def test_final_only_reuses_existing_prelut_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    params = {**_LUT_PARAMS, "nkz0": 1, "nbeta": 4, "n_cpu": 1, "verbose": False}
    _fuga.get_preluts(folder=tmp_path, zeta0=0, nkz0=1, nbeta=4, n_cpu=1, verbose=False)

    def fail(*args, **kwargs):
        raise AssertionError("Existing prelut cache should avoid integration")

    monkeypatch.setattr(_fuga, "_make_preluts", fail)
    final = get_luts(folder=tmp_path, **params, nx=64, ny=32)
    assert np.isfinite(final.UL.values).all()
    assert len(list(tmp_path.glob("*.nc"))) == 2
