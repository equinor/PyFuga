import multiprocessing as mp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pyfuga.constants import UVW_LT
from pyfuga.file_readers import read_lut_file
from pyfuga.flut import FourierLUTGenerator, solve_layer
from pyfuga.preluts import PreLUTs
from pyfuga.profiling import timeit
from pyfuga.utils import compile, get_beta_lst

from .helpers import expose_new_names
from .test_files import tfp


@pytest.mark.parametrize("zeta0", [0, -1, 1])
def test_make_hubheight_luts(zeta0):
    ref = read_lut_file(
        tfp + f"D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0={zeta0}.00E+00/UL9999.lut",
        prelut_folder=tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2",
    )
    preluts = PreLUTs.from_netcdf(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=0.00001, luts=["UL"]
    )

    assert_array_almost_equal(ref.UL.sel(kz0=luts.kz0), luts.sel(level=ref.level).UL)

    assert_array_equal(ref.UL[:, len(luts.kz0) :], 0)


def test_make_hubheight_luts_lo_eq_hi():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    z0 = 70 / np.exp(315 * 0.05)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=z0, luts=["UL"]
    )
    ref = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0, low_level_out=315, high_level_out=315, luts=["UL"]
    )

    assert_array_almost_equal(ref.UL, luts.UL)


def test_all_vars():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    preluts = expose_new_names(preluts)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0=0.00001, low_level_out=314, high_level_out=314
    )
    for v in UVW_LT:
        ref = read_lut_file(
            tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{v}0314.lut",
            prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
        )
        assert_array_almost_equal(luts[v], ref[v][:, : len(luts.kz0)])
        assert_array_equal(0, ref[v][:, len(luts.kz0) :])


def test_compact_preluts():
    preluts_compact = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=get_beta_lst(1),
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
        compact=True,
    )
    preluts = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=get_beta_lst(1),
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
        compact=False,
    )

    luts_c = FourierLUTGenerator(preluts_compact, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0=0.00001, low_level_out=314, high_level_out=314
    )
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0=0.00001, low_level_out=314, high_level_out=314
    )

    for beta in luts.beta.values:
        for kz0 in luts.kz0.values:
            assert preluts.sel(beta=beta, kz0=kz0).equals(preluts_compact.sel(beta=beta, kz0=kz0))
    for k in UVW_LT:
        assert_array_almost_equal(luts[k], luts_c[k])


def test_rotor_luts():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    preluts = expose_new_names(preluts)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_rotor_luts(
        z0=0.00001, luts=["UL"]
    )
    assert luts.z[0] < 70 - 40
    assert luts.z[-1] > 70 + 40


def test_fluts_ncpu():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    preluts = expose_new_names(preluts)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_rotor_luts(
        z0=0.00001, luts=["UL"]
    )
    pluts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_rotor_luts(
        z0=0.00001, luts=["UL"]
    )
    assert luts.equals(pluts)


def test_jit_luts():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    preluts = expose_new_names(preluts)
    lut_generator = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    r1, t1 = timeit(lut_generator.make_lut)(z0=0.00001, low_level_out=315, high_level_out=315, luts=["UL"])
    compile(jit=True)
    r2, t2 = timeit(lut_generator.make_lut, min_runs=2)(z0=0.00001, low_level_out=315, high_level_out=315, luts=["UL"])
    # print(t1, t2)
    assert t2[1] < t1[0]
    assert r1 is not None, "make_lut returned None for r1"
    assert r2 is not None, "make_lut returned None for r2"
    for k in r1:
        # print(k, r1[k].shape)
        assert_array_almost_equal(r1[k], r2[k], 10)


def test_solve_layers_invalid_forcing_raises_valueerror():
    from pyfuga.flut import solve_layers

    args = (
        None,  # prelut (won't be used if forcing check happens first)
        0.0,  # beta
        0.0,  # kz0
        1.0,  # z0
        70.0,  # zhub
        40.0,  # radius
        "X",  # forcing (invalid)
        0,  # lowerjf
        0,  # upperjf
        0,  # minlevel
        0,  # maxlevel
        0,  # low_level_out
        0,  # high_level_out
    )

    import pytest

    with pytest.raises(ValueError, match=r"forcing must be 'L' or 'T'"):
        solve_layers(args)


def test_make_lut_parallel_runs_twice_cleanly():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)

    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    # Run twice to catch "Pool not closed" issues that often show up on second run
    gen.make_lut(z0=0.1, low_level_out=47, high_level_out=165, luts=["UL"], n_cpu=2)
    gen.make_lut(z0=0.1, low_level_out=47, high_level_out=165, luts=["UL"], n_cpu=2)


def test_make_lut_serial_does_not_construct_pool(monkeypatch):
    called = {"pool": 0}

    def fake_pool(*args, **kwargs):
        called["pool"] += 1
        raise AssertionError("Pool should not be constructed in serial mode")

    monkeypatch.setattr(mp.get_context("spawn"), "Pool", fake_pool)

    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)

    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL"], n_cpu=1)

    assert called["pool"] == 0


def test_make_lut_parallel_uses_spawn_context(monkeypatch):
    real_get_context = mp.get_context
    seen = {"arg": None}

    def spy_get_context(arg=None):
        seen["arg"] = arg
        return real_get_context(arg)

    monkeypatch.setattr(mp, "get_context", spy_get_context)

    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)

    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL"], n_cpu=2)

    assert seen["arg"] == "spawn"


def test_make_lut_clamps_low_level_out(capsys):
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    # With z0=0.00001 and ds=0.05, minlevel=230 so minlevel+1=231.
    # Passing low_level_out=229 should be clamped up to 231.
    luts = gen.make_lut(z0=0.00001, low_level_out=229, high_level_out=235, luts=["UL"])

    out = capsys.readouterr().out
    assert "LoLevelOut" in out
    assert luts.level.values[0] == 231


def test_make_lut_clamps_high_level_out(capsys):
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    # With z0=0.00001, zi=400 and ds=0.05, maxlevel=351 so maxlevel-1=350.
    # Passing high_level_out=352 should be clamped down to 350.
    luts = gen.make_lut(z0=0.00001, low_level_out=346, high_level_out=352, luts=["UL"])

    out = capsys.readouterr().out
    assert "HiLevelOut" in out
    assert luts.level.values[-1] == 350


def test_solve_layer_raises_on_forcing_increments_up_mismatch():
    # R_upper has only 1 row so R_upper[icl_m1:icl_p1] yields 1 row, while
    # forcing_increments_up concatenates db_const[0:1] and db_const[1:2] → 2 rows.
    n = 3
    icl_m1, icl, icl_p1 = np.array(0), np.array(1), np.array(2)
    R_upper = np.zeros((1, 6, 6), dtype=np.complex128)
    R_lower = np.zeros((n, 6, 6), dtype=np.complex128)
    Q = np.zeros((n, 6, 6), dtype=np.complex128)
    levels = np.arange(n)
    db_const = np.zeros((n, 6), dtype=np.complex128)
    db_lin = np.zeros((n, 6), dtype=np.complex128)

    with pytest.raises(ValueError, match="Length mismatch between forcing_increments_up and R_upper slice"):
        solve_layer(R_upper, R_lower, Q, levels, db_const, db_lin, 1.0, 0.0, 1.0, 0.0, icl_m1, icl, icl_p1)


def test_solve_layer_raises_on_filtered_levels_mismatch():
    # Q has m=3 rows while b_full_6 ends up with 6 entries (built by the algorithm
    # iterating over n=6 levels), so zip(Q[1:], new_level) filters only 2 items
    # while zip(b_full_6[::-1][1:], new_level) filters 5 items → mismatch.
    n = 6
    m = 3  # intentionally fewer than b_full_6 will have
    icl_m1, icl, icl_p1 = np.array(1), np.array(2), np.array(3)

    R_upper = np.tile(np.eye(6, dtype=np.complex128), (n, 1, 1))
    R_lower = np.tile(np.eye(6, dtype=np.complex128), (n, 1, 1))

    Q = np.zeros((m, 6, 6), dtype=np.complex128)
    # Q[-1, :3] together with the fixed rows forms an invertible permutation matrix for linalg.solve.
    Q[-1, 0, 1] = 1
    Q[-1, 1, 3] = 1
    Q[-1, 2, 5] = 1
    Q[-1, 3, 0] = 1
    Q[-1, 4, 2] = 1
    Q[-1, 5, 4] = 1

    levels = np.arange(n)
    db_const = np.zeros((n, 6), dtype=np.complex128)
    db_lin = np.zeros((n, 6), dtype=np.complex128)

    with pytest.raises(ValueError, match="Length mismatch between filtered Q and b levels"):
        solve_layer(R_upper, R_lower, Q, levels, db_const, db_lin, 1.0, 0.0, 1.0, 0.0, icl_m1, icl, icl_p1)


def test_solve_layer_raises_on_forcing_increments_down_mismatch():
    # With icl_m1=1, icl=2, icl_p1=3:
    #   forcing_increments_down (after decrement) = db_const[2:1:-1] + db_const[1:0:-1] → 2 rows
    #   R_lower_slice = R_lower[3:1:-1] → with only 3 rows numpy clips to 1 row → mismatch
    # R_upper needs ≥ 3 rows so the earlier up-pass check passes (R_upper[1:3] = 2 rows = forcing_increments_up).
    # R_upper[3:-1] is empty (4 rows total), so the "cl+1 to max level" loop doesn't run.
    # R_lower[-1:3:-1] with 3 rows = R_lower[2:3:-1] = empty, so "max level to cl+1" loop doesn't run either.
    icl_m1, icl, icl_p1 = np.array(1), np.array(2), np.array(3)

    R_upper = np.tile(np.eye(6, dtype=np.complex128), (4, 1, 1))
    R_lower = np.tile(np.eye(6, dtype=np.complex128), (3, 1, 1))  # only 3 rows → triggers mismatch

    Q = np.zeros((1, 6, 6), dtype=np.complex128)
    # Q[-1, :3] combined with the fixed rows forms a permutation matrix → Y_tilde invertible
    Q[0, 0, 1] = 1
    Q[0, 1, 3] = 1
    Q[0, 2, 5] = 1
    Q[0, 3, 0] = 1
    Q[0, 4, 2] = 1
    Q[0, 5, 4] = 1

    n = 4
    levels = np.arange(n)
    db_const = np.zeros((n, 6), dtype=np.complex128)
    db_lin = np.zeros((n, 6), dtype=np.complex128)

    with pytest.raises(ValueError, match="Length mismatch between forcing_increments_down and R_lower slice"):
        solve_layer(R_upper, R_lower, Q, levels, db_const, db_lin, 1.0, 0.0, 1.0, 0.0, icl_m1, icl, icl_p1)
