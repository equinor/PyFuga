import multiprocessing as mp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pyfuga.flut as flut_module
from pyfuga.constants import UVW_LT
from pyfuga.file_readers import read_lut_file
from pyfuga.flut import (
    FourierLUTGenerator,
    SolveLayersJob,
    _iter_solve_layer_jobs,
    _select_prelut_arrays,
    _sweep,
    solve_layer,
    solve_layers,
)
from pyfuga.preluts import CompactPreLUTs, PreLUTs
from pyfuga.profiling import timeit
from pyfuga.utils import compile, get_beta_lst, normalise_n_cpu_workers

from .helpers import expose_new_names
from .test_files import tfp

# NOTE: These tests pin LUT *values* against reference files. If a change here is expected
# (e.g. a physics/numerics fix that alters LUT output), bump ``LUT_FORMAT_VERSION`` in
# ``pyfuga/constants.py`` as well, so stale on-disk caches are rejected and regenerated.
# Structural changes (new/renamed variables) are additionally guarded by tests/test_lut_schema.py.


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
    # Timing can fluctuate on CI; keep this as a smoke check rather than a strict speed assertion.
    assert len(t1) > 0 and len(t2) > 0
    assert r1 is not None, "make_lut returned None for r1"
    assert r2 is not None, "make_lut returned None for r2"
    for k in r1:
        # print(k, r1[k].shape)
        assert_array_almost_equal(r1[k], r2[k], 10)


def test_solve_layers_invalid_forcing_raises_valueerror():
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


def test_make_lut_compiled_runs_in_process(monkeypatch):
    def fake_pool(*args, **kwargs):
        raise AssertionError("Pool should not be constructed when the kernels are compiled")

    monkeypatch.setattr(mp.get_context("spawn"), "Pool", fake_pool)
    monkeypatch.setattr(flut_module, "_up_sweep", flut_module._up_sweep_nb)

    preluts = expose_new_names(PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc"))
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL"], n_cpu=2)


def test_sweep_same_bits_for_strided_and_contiguous_y():
    # In-process jobs get strided views of the preLUT table; worker processes get C-ordered pickled copies.
    rng = np.random.default_rng(0)

    def rand(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    n = 30
    y, r_upper, r_lower, f_up, f_down = rand(n, 6, 6), rand(n, 6, 6), rand(n, 6, 6), rand(n, 3), rand(n, 3)
    strided = np.ascontiguousarray(y.transpose(0, 2, 1)).transpose(0, 2, 1)
    out = [_sweep(r_upper, r_lower, q, f_up, f_down, 3) for q in (y, strided)]
    assert out[0].tobytes() == out[1].tobytes()


def test_make_lut_parallel_uses_spawn_context(monkeypatch):
    monkeypatch.setattr(flut_module, "_up_sweep", flut_module._up_sweep_np)  # NumPy fallback: worker processes
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


def test_make_lut_clamps_low_level_out():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    # With z0=0.00001 and ds=0.05, minlevel=230 so minlevel+1=231.
    # Passing low_level_out=229 should be clamped up to 231.
    with pytest.warns(UserWarning, match=r"LoLevelOut \(229\) raised to MinLevel \(231\)"):
        luts = gen.make_lut(z0=0.00001, low_level_out=229, high_level_out=235, luts=["UL"])

    assert luts.level.values[0] == 231


def test_make_lut_clamps_high_level_out():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    # With z0=0.00001, zi=400 and ds=0.05, maxlevel=351 so maxlevel-1=350.
    # Passing high_level_out=352 should be clamped down to 350.
    with pytest.warns(UserWarning, match=r"HiLevelOut \(352\) lowered to MaxLevel \(350\)"):
        luts = gen.make_lut(z0=0.00001, low_level_out=346, high_level_out=352, luts=["UL"])

    assert luts.level.values[-1] == 350


def test_make_lut_rejects_invalid_lut_names():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)

    with pytest.raises(ValueError, match=r"invalid entries: \['XX'\]"):
        gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL", "XX"])


def test_solve_layers_accepts_solve_layers_job_without_tuple_coercion():
    # isinstance(job, SolveLayersJob) is True, so SolveLayersJob._make is skipped.
    job = SolveLayersJob(
        prelut=None,
        beta=0.0,
        kz0=0.0,
        z0=1.0,
        zhub=70.0,
        radius=40.0,
        forcing="X",  # type: ignore[arg-type]
        lowerjf=0,
        upperjf=0,
        minlevel=0,
        maxlevel=0,
        low_level_out=0,
        high_level_out=0,
    )
    with pytest.raises(ValueError, match=r"forcing must be 'L' or 'T'"):
        solve_layers(job)


def test_iter_solve_layer_jobs_forcing_and_length():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    beta_kz0 = [(b, k) for b in preluts.beta.values[:2] for k in preluts.kz0.values[:2]]
    jobs = list(
        _iter_solve_layer_jobs(
            preluts,
            beta_kz0,
            z0=0.00001,
            zhub=70.0,
            radius=40.0,
            forcing="T",
            lowerjf=10,
            upperjf=20,
            minlevel=1,
            maxlevel=300,
            low_level_out=314,
            high_level_out=314,
        )
    )
    assert len(jobs) == len(beta_kz0)
    assert all(j.forcing == "T" for j in jobs)
    assert jobs[0].prelut is not None


def test_compact_array_selection_matches_sel():
    ds = CompactPreLUTs(
        {"level": ("kz0_beta_i", [0, 1, 2, 0, 1]), "Y_lower": ("kz0_beta_i", np.arange(5))},
        coords={"beta": ("kz0_beta_i", [1, 1, 1, 0, 0]), "kz0": ("kz0_beta_i", [2, 2, 2, 3, 3])},
        attrs={"ds": 0.05},
    )
    for selected in (ds, ds.isel(kz0_beta_i=[0, 3, 1, 4, 2])):
        selected = CompactPreLUTs(selected.data_vars, coords=selected.coords, attrs=selected.attrs)
        pairs = [(0, 3), (1, 2)]
        for pair, arrays in zip(pairs, _select_prelut_arrays(selected, pairs, ["level", "Y_lower"]), strict=True):
            ref = selected.sel(beta=pair[0], kz0=pair[1])
            for name in ["level", "Y_lower"]:
                assert_array_equal(arrays[name], ref[name])
            assert arrays.ds == ref.ds


def test_make_lut_t_only_skips_longitudinal_solve():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    luts = gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UT"])
    assert "UT" in luts.data_vars
    assert luts.UT.shape == (len(luts.beta), len(luts.kz0), 1)


def test_make_lut_ncpu_none_matches_explicit_worker_count():
    preluts = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    preluts = expose_new_names(preluts)
    gen = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    workers = normalise_n_cpu_workers(None)
    a = gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL"], n_cpu=None)
    b = gen.make_lut(z0=0.00001, low_level_out=314, high_level_out=314, luts=["UL"], n_cpu=workers)
    assert a.equals(b)


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


def test_solve_layers_returns_zeros_when_max_table_level_below_minlevel():
    class DummyPrelut:
        ds = 0.05
        level = np.array([0, 1], dtype=int)

    job = SolveLayersJob(
        prelut=DummyPrelut(),
        beta=0.0,
        kz0=1.0,
        z0=1.0,
        zhub=70.0,
        radius=40.0,
        forcing="L",
        lowerjf=0,
        upperjf=2,
        minlevel=10,
        maxlevel=12,
        low_level_out=10,
        high_level_out=11,
    )

    out = solve_layers(job)
    assert out.shape == (2, 6)
    assert np.all(out == 0)


def test_solve_layer_returns_paired_levels_without_mismatch():
    icl_m1, icl, icl_p1 = np.array(1), np.array(2), np.array(3)
    R_upper = np.tile(np.eye(6, dtype=np.complex128), (4, 1, 1))
    R_lower = np.tile(np.eye(6, dtype=np.complex128), (4, 1, 1))
    Q = np.zeros((4, 6, 6), dtype=np.complex128)
    # Make Y_tilde invertible at max level.
    Q[-1, 0, 1] = 1
    Q[-1, 1, 3] = 1
    Q[-1, 2, 5] = 1
    Q[-1, 3, 0] = 1
    Q[-1, 4, 2] = 1
    Q[-1, 5, 4] = 1
    levels = np.array([0, 1, 2, 3], dtype=int)
    db_const = np.zeros((4, 6), dtype=np.complex128)
    db_lin = np.zeros((4, 6), dtype=np.complex128)

    out = solve_layer(R_upper, R_lower, Q, levels, db_const, db_lin, 1.0, 0.0, 1.0, 0.0, icl_m1, icl, icl_p1)
    assert isinstance(out, list)
    assert len(out) > 0
