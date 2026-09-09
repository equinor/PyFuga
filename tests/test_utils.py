import warnings

import numpy as np
import pytest
import xarray as xr

import pyfuga
from pyfuga import utils
from pyfuga.constants import LUT_FORMAT_VERSION


def _sample_lut() -> xr.Dataset:
    return xr.Dataset(
        data_vars={"UL": (("z", "y", "x"), np.zeros((2, 3, 4)))},
        coords={"z": [10.0, 20.0], "y": np.arange(3.0), "x": np.arange(4.0)},
    )


def test_stamp_lut_metadata_sets_all_three_attributes():
    ds = utils.stamp_lut_metadata(_sample_lut())
    assert ds.attrs["lut_format_version"] == LUT_FORMAT_VERSION
    assert ds.attrs["lut_schema_fingerprint"] == utils.lut_schema_fingerprint(ds)
    assert ds.attrs["pyfuga_version"] == pyfuga.__version__


def test_lut_schema_fingerprint_is_structural_not_value_based():
    a = _sample_lut()
    b = _sample_lut()
    b["UL"].values[:] = 999.0  # different values, same structure
    assert utils.lut_schema_fingerprint(a) == utils.lut_schema_fingerprint(b)


def test_lut_schema_fingerprint_changes_on_structural_change():
    base = utils.lut_schema_fingerprint(_sample_lut())
    renamed = _sample_lut().rename({"UL": "UT"})
    added = _sample_lut()
    added["UT"] = (("z", "y", "x"), np.zeros((2, 3, 4)))
    assert utils.lut_schema_fingerprint(renamed) != base
    assert utils.lut_schema_fingerprint(added) != base


def test_pyfuga_version_falls_back_when_metadata_missing(monkeypatch):
    import importlib.metadata as importlib_metadata

    def raise_not_found(_dist_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError("pyfuga")

    monkeypatch.setattr(importlib_metadata, "version", raise_not_found)
    assert utils._pyfuga_version() == "0+unknown"


def test_stamp_lut_metadata_uses_pyfuga_version_fallback(monkeypatch):
    monkeypatch.setattr(utils, "_pyfuga_version", lambda: "0+unknown")
    ds = utils.stamp_lut_metadata(_sample_lut())
    assert ds.attrs["pyfuga_version"] == "0+unknown"


def test_cache_is_current_accepts_freshly_stamped():
    assert utils.cache_is_current(utils.stamp_lut_metadata(_sample_lut()))


def test_cache_is_current_rejects_legacy_without_metadata():
    assert not utils.cache_is_current(_sample_lut())


def test_cache_is_current_rejects_wrong_format_version():
    ds = utils.stamp_lut_metadata(_sample_lut())
    ds.attrs["lut_format_version"] = LUT_FORMAT_VERSION - 1
    assert not utils.cache_is_current(ds)


def test_cache_is_current_rejects_mismatched_fingerprint():
    ds = utils.stamp_lut_metadata(_sample_lut())
    ds.attrs["lut_schema_fingerprint"] = "deadbeef0000"  # correct version, corrupted fingerprint
    assert not utils.cache_is_current(ds)


def test_load_lut_if_current_warns_and_returns_false_for_legacy():
    with pytest.warns(UserWarning, match="incompatible cached LUT"):
        assert utils.load_lut_if_current(_sample_lut(), "legacy.nc") is False


def test_load_lut_if_current_silent_true_for_current():
    ds = utils.stamp_lut_metadata(_sample_lut())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any UserWarning would fail the test
        assert utils.load_lut_if_current(ds, "ok.nc") is True


def test_profile_decorator_behaves_like_decorator():
    calls = {}

    @utils.profile
    def add_one(x):
        calls["x"] = x
        return x + 1

    result = add_one(2)

    # Whatever profile is (real or dummy), the wrapped function should still:
    # - be callable
    # - return the original result
    # - execute the body
    assert result == 3
    assert calls["x"] == 2
