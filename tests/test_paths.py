from pyfuga.paths import get_fluts_path, get_level_range, get_luts_path, get_preluts_path


def test_path_format_stability(tmp_path):
    p = {"zeta0": 0.0, "nkz0": 1, "nbeta": 2}
    f = {**p, "diameter": 80, "zhub": 70, "z0": 1e-5, "zi": 400, "zlow": 70, "zhigh": 70, "lut_vars": ["UL"]}

    assert get_preluts_path(tmp_path, **p).name == "preLUTs_Zeta0=0.00e+00_1_2.nc"
    assert get_fluts_path(tmp_path, **f).name == ("fLUTs_Zeta0=0.00e+00_1_2_D80_zhub70_zi400_z0=0.00001000_z70.0_UL.nc")
    assert "_UT" in get_fluts_path(tmp_path, **{**f, "lut_vars": ["UT"]}).name
    assert get_luts_path(tmp_path, **f, nx=256, ny=64, dx=20.0, dy=5.0).name == (
        "LUTs_Zeta0=0.00e+00_1_2_D80_zhub70_zi400_z0=0.00001000_z70.0_UL_nx256_ny64_dx20.0_dy5.0.nc"
    )


def test_multiheight_produces_different_path(tmp_path):
    common = {"zeta0": 0.0, "nkz0": 1, "nbeta": 2, "diameter": 80, "zhub": 70, "z0": 1e-5, "zi": 400}
    hub = get_fluts_path(tmp_path, **common, zlow=70, zhigh=70, lut_vars=["UL"])
    multi = get_fluts_path(tmp_path, **common, zlow=50, zhigh=90, lut_vars=["UL"])
    assert hub != multi


def test_get_level_range():
    assert get_level_range(zlow=70, zhigh=70, zhub=70, z0=1e-5) == (9999, 9999)
    low, high = get_level_range(zlow=50, zhigh=90, zhub=70, z0=1e-5)
    assert low < high
