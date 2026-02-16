import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pyfuga.z0 as z0
from pyfuga.z0 import _g_unstable, phi, psi, z0_from_TI


def test_phi():
    zeta = np.linspace(-0.1, 0.1, 11)
    print(list(np.round(phi(zeta), 3)))
    if 0:
        plt.plot(zeta, phi(zeta))
        plt.show()
    assert_array_almost_equal(phi(zeta), [0.766, 0.794, 0.827, 0.868, 0.923, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 3)


def test_psi():
    zeta = np.linspace(-0.1, 0.1, 51)
    print(list(np.round(psi(zeta[::5]), 3)))
    print(list(np.round(psi(zeta[::5], "Wilson"), 3)))
    if 0:
        plt.plot(zeta, psi(zeta))
        plt.plot(zeta, psi(zeta, "Wilson"), label="Wilson")
        plt.legend()
        plt.show()
    assert_array_almost_equal(psi(zeta[::5]), [0.322, 0.273, 0.218, 0.157, 0.085, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)
    assert_array_almost_equal(
        psi(zeta[::5], "Wilson"), [2.541, 2.488, 2.427, 2.355, 2.261, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3
    )


def test_z0_from_TI():
    if 0:
        TI = np.linspace(3, 10, 100) / 100
        zref = 70
        plt.figure()

        z0_stable = z0_from_TI(TI, zref, 6e-7)
        z0_neutral = z0_from_TI(TI, zref, 0.0)
        z0_unstable = z0_from_TI(TI, zref, -6e-7)
        plt.plot(TI * 100, z0_stable, label="stable")
        plt.plot(TI * 100, z0_neutral, label="neutral")
        plt.plot(TI * 100, z0_unstable, label="unstable")

        z0_unstable = z0_from_TI(TI, zref, -6e-7)
        plt.plot(TI * 100, z0_unstable, "--", label="unstable")

        plt.xlabel("ti%")
        plt.ylabel("z0")
        plt.legend()
        plt.show()
    assert_array_almost_equal(
        [z0_from_TI([0.06, 0.12], 70, zeta0) for zeta0 in [-6e-7, 0, 6e-7]],
        [[1e-05, 1.663e-02], [1e-05, 1.683e-02], [7.269e-05, 1.703e-02]],
        5,
    )


def test_z0_from_TI_unstable_hits_clamp_returns_limit_no_error():
    # Choose TI small enough that neutral would imply z0 < z0_limit
    TI = np.array([0.06])
    zref = 70.0
    zeta0 = -6e-7
    z0_limit = 1e-5

    z0 = z0_from_TI(TI, zref, zeta0, z0_limit=z0_limit)
    assert np.allclose(z0, z0_limit)

    x = zref / z0
    assert np.allclose(x, zref / z0_limit)


def test_z0_from_TI_unstable_nonclamp_returns_finite_and_above_limit():
    TI = np.array([0.12])
    zref = 70.0
    zeta0 = -6e-7
    z0_limit = 1e-5

    z0 = z0_from_TI(TI, zref, zeta0, z0_limit=z0_limit)

    assert np.isfinite(z0).all()
    assert (z0 > z0_limit).all()


def test_z0_from_TI_unstable_solution_satisfies_residual_when_not_clamped():
    ti = 0.12
    zref = 70.0
    zeta0 = -6e-7
    z0_limit = 1e-5

    z0 = z0_from_TI([ti], zref, zeta0, z0_limit=z0_limit)[0]
    x = zref / z0

    # Only meaningful if we are not clamped
    assert z0 > z0_limit

    g = x / np.exp(psi(x * zeta0) - psi(zeta0)) - np.exp(1.0 / ti)
    assert abs(g) < 1e-8  # pick tolerance based on observed behaviour


def test_z0_from_TI_unstable_no_fsolve_progress_warning():
    ti = 0.12
    zref = 70.0
    zeta0 = -6e-7

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        z0_from_TI([ti], zref, zeta0)

    msgs = [str(w.message) for w in rec]
    assert not any("not making good progress" in m for m in msgs)


def test_z0_from_TI_unstable_vectorised_shape_and_order():
    TI = np.array([0.08, 0.10, 0.12])
    zref = 70.0
    zeta0 = -6e-7

    z0 = z0_from_TI(TI, zref, zeta0)

    assert z0.shape == TI.shape
    assert np.isfinite(z0).all()

    # Optional physical sanity: higher TI -> larger z0 (roughly)
    assert np.all(np.diff(z0) >= 0)


def test__g_unstable_rejects_nonpositive_x_with_infinite_value():
    ti = 0.12
    zeta0 = -6e-7

    val0 = _g_unstable(0.0, ti, zeta0)
    valneg = _g_unstable(-1.0, ti, zeta0)

    assert np.isinf(val0)
    assert np.isinf(valneg)
    assert val0 < 0 and valneg < 0


def test__solve_x_unstable_uses_fallback_bracket_between_xhi_good_and_xmax(monkeypatch):
    ti = 0.12
    zeta0 = -6e-7
    zref = 70.0
    z0_limit = 1e-5

    x_max = zref / z0_limit
    x_hi_initial = min(x_max, max(10.0, zref / 1e-3))  # 70000.0

    calls_at_xmax = {"n": 0}

    def fake_g(x, ti_, zeta0_):
        assert ti_ == ti
        assert zeta0_ == zeta0

        # f_lo: finite negative
        if x <= 1e-12:
            return -1.0

        # initial f_hi: finite negative -> x_hi_good = x_hi_initial
        if abs(x - x_hi_initial) < 1e-9:
            return -1.0

        # Intermediate probes: non-finite so x_hi_good doesn't change
        if x_hi_initial < x < x_max:
            return np.nan

        # Key: x_max is queried twice:
        # - once inside the loop (f_hi at x_hi=x_max)
        # - once after the loop (f_at_max)
        if abs(x - x_max) < 1e-6:
            calls_at_xmax["n"] += 1
            if calls_at_xmax["n"] == 1:
                return np.nan  # loop's f_hi -> prevents normal bracketing solve
            return +1.0  # f_at_max -> triggers fallback block

        return np.nan

    monkeypatch.setattr(z0, "_g_unstable", fake_g)

    class DummySol:
        converged = True
        root = 1234.0

    def fake_root_scalar(func, args, bracket, method):
        assert method == "brentq"
        assert func is z0._g_unstable
        assert args == (ti, zeta0)
        # Now we really do want the fallback bracket:
        assert bracket == (x_hi_initial, x_max)
        return DummySol()

    monkeypatch.setattr(z0, "root_scalar", fake_root_scalar)

    out = z0._solve_x_unstable(ti, zeta0, zref, z0_limit)
    assert out == 1234.0


def test__solve_x_unstable_falls_back_to_fsolve_and_returns_root(monkeypatch):
    ti = 0.12
    zeta0 = -6e-7
    zref = 70.0
    z0_limit = 1e-5

    x_max = zref / z0_limit
    x_hi_expected = min(x_max, max(10.0, zref / 1e-3))  # 70000.0

    def fake_g(x, ti_, zeta0_):
        # f_lo finite
        if x <= 1e-12:
            return -1.0
        # f_hi non-finite => normal bracketing cannot happen
        if abs(x - x_hi_expected) < 1e-9:
            return np.nan
        # f_at_max non-finite => skip both "clamp applies" and "(x_hi, x_max) fallback" blocks
        if abs(x - x_max) < 1e-6:
            return np.nan
        return -1.0

    monkeypatch.setattr(z0, "_g_unstable", fake_g)

    # root_scalar must not be used
    def bomb_root_scalar(*args, **kwargs):
        raise AssertionError("root_scalar should not be used in this test")

    monkeypatch.setattr(z0, "root_scalar", bomb_root_scalar)

    # fsolve succeeds
    def fake_fsolve(func, x0, full_output):
        assert full_output is True
        return (np.array([4321.0]), None, 1, "ok")

    monkeypatch.setattr(z0, "fsolve", fake_fsolve)

    out = z0._solve_x_unstable(ti, zeta0, zref, z0_limit)
    assert out == 4321.0


def test__solve_x_unstable_fsolve_failure_still_returns_xmax_when_clamp_applies(monkeypatch):
    ti = 0.06
    zeta0 = -6e-7
    zref = 70.0
    z0_limit = 1e-5
    x_max = zref / z0_limit

    # Make everything fail to bracket, and ensure clamp condition holds at x_max
    def fake_g(x, ti_, zeta0_):
        # Always negative, including at x_max => clamp applies
        return -1.0

    monkeypatch.setattr(z0, "_g_unstable", fake_g)

    # root_scalar should not be called
    def bomb_root_scalar(*args, **kwargs):
        raise AssertionError("root_scalar should not be used in this test")

    monkeypatch.setattr(z0, "root_scalar", bomb_root_scalar)

    # Fake fsolve non-convergence
    def fake_fsolve(func, x0, full_output):
        return (np.array([999.0]), None, 5, "no progress")

    monkeypatch.setattr(z0, "fsolve", fake_fsolve)

    out = z0._solve_x_unstable(ti, zeta0, zref, z0_limit)

    assert out == x_max


def test__solve_x_unstable_raises_when_no_root_and_no_clamp(monkeypatch):
    ti = 0.12
    zeta0 = -6e-7
    zref = 70.0
    z0_limit = 1e-5

    # No bracket (always positive), so no root in (0, x_max] in our fake world
    def fake_g(x, ti_, zeta0_):
        return +1.0

    monkeypatch.setattr(z0, "_g_unstable", fake_g)

    # root_scalar shouldn't help; if called, return non-converged
    class DummySol:
        converged = False
        root = np.nan

    monkeypatch.setattr(z0, "root_scalar", lambda *a, **k: DummySol())

    # fsolve fails
    monkeypatch.setattr(z0, "fsolve", lambda *a, **k: (np.array([1.0]), None, 5, "no progress"))

    with pytest.raises(RuntimeError, match="Unstable z0_from_TI solve failed"):
        z0._solve_x_unstable(ti, zeta0, zref, z0_limit)


def test__solve_x_unstable_final_clamp_check_returns_xmax(monkeypatch):
    ti = 0.12
    zeta0 = -6e-7
    zref = 70.0
    z0_limit = 1e-5
    x_max = zref / z0_limit
    x_hi_initial = min(x_max, max(10.0, zref / 1e-3))  # 70000.0

    calls_at_xmax = {"n": 0}

    def fake_g(x, ti_, zeta0_):
        assert ti_ == ti
        assert zeta0_ == zeta0

        # Make f_lo finite
        if x <= 1e-12:
            return -1.0

        # Make initial f_hi finite so we have x_hi_good/f_hi_good, but never a bracket
        if abs(x - x_hi_initial) < 1e-9:
            return -1.0

        # For all other x in (x_lo, x_max), keep it negative (no sign change)
        if x < x_max:
            return -1.0

        # x == x_max is queried at least twice:
        # - earlier "If we never bracketed" clamp check: must NOT clamp -> return +1
        # - final clamp check (after fsolve): must clamp -> return -1
        if abs(x - x_max) < 1e-6:
            calls_at_xmax["n"] += 1
            return +1.0 if calls_at_xmax["n"] == 1 else -1.0

        return -1.0

    monkeypatch.setattr(z0, "_g_unstable", fake_g)

    # Make root_scalar not return a usable solution
    class DummySol:
        converged = False
        root = np.nan

    monkeypatch.setattr(z0, "root_scalar", lambda *a, **k: DummySol())

    # Make fsolve fail to converge
    def fake_fsolve(func, x0, full_output):
        return (np.array([1.0]), None, 5, "no progress")

    monkeypatch.setattr(z0, "fsolve", fake_fsolve)

    out = z0._solve_x_unstable(ti, zeta0, zref, z0_limit)

    assert out == x_max
    assert calls_at_xmax["n"] >= 2  # sanity: we exercised both clamp checks
