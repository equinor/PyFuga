"""Guard test pinning the on-disk LUT structure of each pipeline tier.

The cache-validation logic (:func:`pyfuga.utils.cache_is_current`) rejects a cached LUT whose
``lut_schema_fingerprint`` no longer matches, and bumps of ``LUT_FORMAT_VERSION`` invalidate
older caches. These tests make a *structural* change impossible to ship silently: if the set of
variables/dimensions written for any tier changes, the golden schema below stops matching and the
test fails with a reminder to bump the version.

If a failure here is expected, update the golden schema **and** bump ``LUT_FORMAT_VERSION`` in
``pyfuga/constants.py`` (see ``_BUMP_MSG``).

The datasets are generated with the smallest committed fixture and a low interpolation order:
values are irrelevant here, only structure (variable and dimension names) is checked, and that
structure is independent of the interpolation order and grid size.
"""

from __future__ import annotations

import pytest

from pyfuga.constants import UVW_LT
from pyfuga.flut import FourierLUTGenerator
from pyfuga.preluts import PreLUTs
from pyfuga.trafalgar import Trafalgar
from pyfuga.utils import lut_schema_fingerprint

from .helpers import expose_new_names
from .test_files import tfp

_BUMP_MSG = (
    "LUT schema changed. If this is intended, bump LUT_FORMAT_VERSION in pyfuga/constants.py "
    "(so stale on-disk caches are rejected and regenerated) and update the golden schema in "
    "tests/test_lut_schema.py."
)

EXPECTED_PRELUTS_SCHEMA = {
    "R_lower": ("kz0_beta_i", "j", "k"),
    "R_upper": ("kz0_beta_i", "j", "k"),
    "Y_lower": ("kz0_beta_i", "j", "k"),
    "beta": ("kz0_beta_i",),
    "dbx_const": ("kz0_beta_i", "j"),
    "dbx_lin": ("kz0_beta_i", "j"),
    "dby_const": ("kz0_beta_i", "j"),
    "dby_lin": ("kz0_beta_i", "j"),
    "dbz_const": ("kz0_beta_i", "j"),
    "dbz_lin": ("kz0_beta_i", "j"),
    "kz0": ("kz0_beta_i",),
    "level": ("kz0_beta_i",),
    "log_s_lower": ("kz0_beta_i",),
    "log_s_upper": ("kz0_beta_i",),
}

EXPECTED_FLUTS_SCHEMA = {
    **dict.fromkeys(UVW_LT, ("beta", "kz0", "level")),
    "beta": ("beta",),
    "diameter": (),
    "hubheight": (),
    "k": ("kz0",),
    "kz0": ("kz0",),
    "level": ("level",),
    "z": ("level",),
    "z0": (),
}

EXPECTED_LUTS_SCHEMA = {
    **dict.fromkeys(UVW_LT, ("z", "x", "y")),
    "diameter": (),
    "hubheight": (),
    "x": ("x",),
    "y": ("y",),
    "z": ("z",),
    "z0": (),
}


def _schema(ds) -> dict[str, tuple[str, ...]]:
    return {str(name): tuple(map(str, var.dims)) for name, var in ds.variables.items()}


@pytest.fixture(scope="module")
def preluts() -> PreLUTs:
    return expose_new_names(PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc"))


@pytest.fixture(scope="module")
def hubheight_fluts(preluts: PreLUTs):
    return FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=1e-5, luts=UVW_LT, n_cpu=1
    )


def test_preluts_schema_is_stable() -> None:
    pre_ref = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    generated = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=pre_ref.beta[:2].values,
        kzmax=1e-7,
        ds=0.05,
        accgoal=1e-4,
        jit=False,
        n_cpu=1,
        verbose=False,
    )
    assert _schema(generated) == EXPECTED_PRELUTS_SCHEMA, _BUMP_MSG


def test_fluts_schema_is_stable(hubheight_fluts) -> None:
    assert _schema(hubheight_fluts) == EXPECTED_FLUTS_SCHEMA, _BUMP_MSG


def test_luts_schema_is_stable(hubheight_fluts) -> None:
    luts = Trafalgar(hubheight_fluts, nx=64, ny=32, verbose=False).make_luts(1, n_cpu=1)
    assert _schema(luts) == EXPECTED_LUTS_SCHEMA, _BUMP_MSG


def test_fingerprint_flips_on_structural_change(hubheight_fluts) -> None:
    """The runtime safety net must actually distinguish a structural change."""
    base = lut_schema_fingerprint(hubheight_fluts)
    assert lut_schema_fingerprint(hubheight_fluts.rename({"UL": "renamed"})) != base
