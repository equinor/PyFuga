from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pyfuga
import pyfuga._fuga as _fuga
from pyfuga import get_luts
from pyfuga.paths import get_fluts_path, get_luts_path, get_preluts_path


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
        n_cpu=None,
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

    luts = get_luts(**params, jit=True, n_cpu=1)

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
