from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyfuga
import pyfuga._fuga as _fuga
from pyfuga import get_luts


def test_get_luts_wrapper_delegates_to__fuga(monkeypatch):
    sentinel = object()

    def fake_get_luts(*args, **kwargs):
        # return something distinctive so we can assert delegation
        return (sentinel, args, kwargs)

    monkeypatch.setattr(_fuga, "get_luts", fake_get_luts)

    out = pyfuga.get_luts(1, 2, foo="bar")

    assert out[0] is sentinel
    assert out[1] == (1, 2)
    assert out[2] == {"foo": "bar"}


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
