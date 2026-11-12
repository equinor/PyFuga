from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from scipy import sparse

from pyfuga.file_readers import read_lut_dat_file, read_lut_file
from pyfuga.trafalgar import Trafalgar, _interpolate_levels, _spline_operator

from .test_files import tfp


def _scipy_contracts_fma() -> bool:
    a, b = 1.0 + 2.0**-30, 1.0 - 2.0**-30
    return bool((sparse.csr_matrix(np.array([[1.0, a]])) @ np.array([[-1.0], [b]]))[0, 0] != 0.0)


_FMA = _scipy_contracts_fma()


def _assert_same_bits(actual, expected):
    actual, expected = actual.view(np.float64), expected.view(np.float64)
    if not _FMA:
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        return
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[~finite], expected[~finite])
    atol = 16 * np.finfo(np.float64).eps * np.abs(expected[finite]).max()
    assert_allclose(actual[finite], expected[finite], rtol=0, atol=atol)


def test_trafalgar_make_luts_rejects_invalid_n_cpu() -> None:
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="None or >= 1"):
        Trafalgar(ds, nx=8, ny=8, verbose=False).make_luts(n_cpu=0)


@pytest.mark.parametrize("var", ["UL"])
def test_trafalgar_n_cpu_1_does_not_instantiate_pool(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    workers_seen: list[int] = []

    @contextmanager
    def spy_map(
        workers: int,
    ) -> Iterator[Callable[[Callable[..., object], Iterable[object]], Iterator[object]]]:
        workers_seen.append(workers)
        yield map

    monkeypatch.setattr("pyfuga.trafalgar.threading_map", spy_map)

    fourier_lut = read_lut_file(
        tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )
    Trafalgar(fourier_lut, nx=512, ny=128, verbose=False).make_luts(3, n_cpu=1)
    assert workers_seen == [1]


@pytest.mark.parametrize("var", ["UL"])
def test_trafalgar_default_none_uses_all_cpus(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pyfuga.utils.os.cpu_count", lambda: 3)
    workers_seen: list[int] = []

    @contextmanager
    def spy_map(
        workers: int,
    ) -> Iterator[Callable[[Callable[..., object], Iterable[object]], Iterator[object]]]:
        workers_seen.append(workers)
        yield map

    monkeypatch.setattr("pyfuga.trafalgar.threading_map", spy_map)

    fourier_lut = read_lut_file(
        tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )
    Trafalgar(fourier_lut, nx=512, ny=128, verbose=False).make_luts(3)
    assert workers_seen == [3]


@pytest.mark.parametrize("verbose", [True, False])
def test_trafalgar_progress_bars_follow_verbose(verbose: bool, capsys: pytest.CaptureFixture[str]) -> None:
    fourier_lut = read_lut_file(
        tfp + "D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )
    Trafalgar(fourier_lut, nx=512, ny=128, verbose=verbose).make_luts(3, n_cpu=1)
    err = capsys.readouterr().err
    assert ("LUTS: adding meshes" in err) is verbose
    assert ("Trafalgar:" in err) is verbose


@pytest.mark.parametrize("verbose", [True, False])
def test_trafalgar_parallel_shows_only_outer_progress_bar(verbose: bool, capfd: pytest.CaptureFixture[str]) -> None:
    fourier_lut = read_lut_file(
        tfp + "D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )
    Trafalgar(fourier_lut, nx=512, ny=128, verbose=verbose).make_luts(3, n_cpu=2)
    err = capfd.readouterr().err
    assert ("LUTS: adding meshes" in err) is verbose
    assert "Trafalgar:" not in err


@pytest.mark.parametrize("var", ["UL", "UT"])
def test_trafalgar(var):
    fourier_lut = read_lut_file(
        tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )

    fuga_luts = Trafalgar(fourier_lut, nx=512, ny=128, sigmax=80, sigmay=20, verbose=False).make_luts(
        3, legacy=True, n_cpu=1
    )

    fn = tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/D080.0000_zH070.0000_2_5_FIT0314{var}.dat"
    ref = read_lut_dat_file(fn, nx=512, ny=128, dx=20, dy=5)
    assert_allclose(fuga_luts[var].isel(z=0).T, ref, rtol=1e-7, atol=1e-7)
    if 0:
        fuga_luts[var].plot()
        plt.show()


@pytest.mark.parametrize("var", ["UL", "UT"])
def test_trafalgar_new_mode(var):
    """Smoke test for the new multi-tile FFIT (non-legacy path)."""
    fourier_lut = read_lut_file(
        tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
    )
    fuga_luts = Trafalgar(fourier_lut, nx=512, ny=128, verbose=False).make_luts(3, n_cpu=1)
    assert var in fuga_luts
    assert fuga_luts[var].shape == (1, 512, 64)
    assert np.isfinite(fuga_luts[var].values).all()


@pytest.mark.parametrize("var", ["UL", "UT"])
@pytest.mark.parametrize("legacy", [False, True])
def test_threaded_tiles_match_serial_without_mutating_input(var, legacy):
    fourier_lut = (
        read_lut_file(
            tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut",
            prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
        )
        .isel(level=[0, 0, 0])
        .assign_coords(level=[314, 315, 316])
    )
    original = fourier_lut.copy(deep=True)
    generator = Trafalgar(fourier_lut, nx=64, ny=32, verbose=False)
    serial = generator.make_luts(3, legacy=legacy, n_cpu=1)
    threaded = generator.make_luts(3, legacy=legacy, n_cpu=2)
    xr.testing.assert_identical(threaded, serial)
    xr.testing.assert_identical(fourier_lut, original)


@pytest.mark.parametrize(("nlevel", "kind"), [(1, "complex"), (3, "complex"), (3, "real"), (3, "nonfinite")])
def test_interpolate_levels_preserves_bits(nlevel, kind):
    rng = np.random.default_rng(42)
    x, y = np.linspace(0, 1, 6), np.linspace(-10, -2, 7)
    px, py = np.meshgrid(np.linspace(-0.2, 1.2, 8), np.linspace(-11, -1, 9))
    operator = _spline_operator(x, y, px, py, 3)
    assert operator is not None
    flut = rng.standard_normal((len(x), len(y), 2 * nlevel))
    if kind != "real":
        flut = flut + 1j * rng.standard_normal(flut.shape)
    if kind == "nonfinite":
        flut[0, 0, 0] = np.inf
    flut = flut[:, :, ::2]
    original = flut.copy()
    flut.flags.writeable = False

    w, cx, cy = operator.matrix(), operator.cx, operator.cy
    with np.errstate(invalid="ignore"):
        coef = np.einsum("ab,bcl,dc->adl", cx, flut, cy, optimize=True)
        expected = w @ coef.reshape(-1, nlevel)
        actual = _interpolate_levels(operator, flut)

    _assert_same_bits(actual, expected)
    np.testing.assert_array_equal(flut, original)


def test_spline_operator_matches_scipy_design_matrix_bits():
    from scipy.interpolate import BSpline, make_interp_spline

    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(-3, 3, 17))
    t = make_interp_spline(x, np.eye(len(x)), k=3).t
    q = np.concatenate([x, t, rng.uniform(x[0], x[-1], 500), np.nextafter(x, np.inf).clip(x[0], x[-1])])
    operator = _spline_operator(x, x[:5], q, np.full_like(q, x[0]), 3)
    assert operator is not None

    expected = BSpline.design_matrix(q, t, 3)
    _assert_same_bits(operator.bx.ravel(), expected.data)
    np.testing.assert_array_equal(operator.ox, expected.indices[::4])


def test_interpolate_scaled_matches_unfused_bits():
    from pyfuga.trafalgar import _interpolate_scaled

    rng = np.random.default_rng(7)
    x, y = np.linspace(0, 1, 6), np.linspace(-10, -2, 7)
    px, py = np.meshgrid(np.linspace(-0.2, 1.2, 8), np.linspace(-11, -1, 9))
    operator = _spline_operator(x, y, px, py, 3)
    assert operator is not None
    flut = rng.standard_normal((len(x), len(y), 3)) + 1j * rng.standard_normal((len(x), len(y), 3))
    scale = rng.standard_normal(px.shape) * np.exp(-rng.uniform(0, 30, px.shape))
    scale[0, :3] = [0.0, -0.0, 1e-30]

    expected = _interpolate_levels(operator, flut).reshape(*px.shape, -1) * scale[..., np.newaxis]
    actual = _interpolate_scaled(operator, flut, scale)

    assert actual is not None
    assert actual.dtype == expected.dtype
    _assert_same_bits(actual, np.ascontiguousarray(np.moveaxis(expected, -1, 0)))
    flut[0, 0, 0] = np.inf
    with np.errstate(invalid="ignore"):
        assert _interpolate_scaled(operator, flut, scale) is None


@pytest.mark.parametrize("order", [1, 3])
def test_x_transform_fused_matches_unfused_bits(order):
    from pyfuga.trafalgar import _fitfit_x, _interpolate_scaled, _spline_components, _x_transform_fused

    rng = np.random.default_rng(3)
    nx, ny, dx = 40, 45, 20.0  # ny not a multiple of the block width
    x, y = np.linspace(0, 1, 6), np.linspace(-10, -2, 7)
    px, py = np.meshgrid(np.linspace(-0.2, 1.2, ny), np.linspace(-11, -1, nx))
    operator = _spline_operator(x, y, px, py, order)
    assert operator is not None
    fluts = [rng.standard_normal((len(x), len(y), 3)) + 1j * rng.standard_normal((len(x), len(y), 3)) for _ in "ab"]
    scale = rng.standard_normal(px.shape)
    kx = np.linspace(1e-4, 1e-1, nx)

    components = [_spline_components(operator, f) for f in fluts]
    actual = _x_transform_fused(operator, components, scale, [1, -1], kx, nx, dx, 4)
    for j, (flut, sign) in enumerate(zip(fluts, [1, -1], strict=True)):
        scaled = _interpolate_scaled(operator, flut, scale)
        assert scaled is not None
        expected = np.stack([_fitfit_x(field, sign, kx, nx, dx, 4) for field in scaled])
        np.testing.assert_array_equal(actual[3 * j : 3 * (j + 1)], expected)


@pytest.mark.parametrize("sign", [1, -1])
def test_blocked_fitfit_matches_unblocked_bits(sign):
    from numpy import newaxis as na
    from scipy import fft as sfft

    from pyfuga.trafalgar import FITFIT, _fitfit_axis, doKvectorFIT_double

    nx, ny = 300, 45  # not multiples of the block sizes
    rng = np.random.default_rng(0)
    field = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    kx, ky = doKvectorFIT_double(nx, 20, NNN=4), doKvectorFIT_double(ny, 5, NNN=4)
    phix, psix, hhatx = _fitfit_axis(nx, 20, 4, float(kx[0]), -nx / 4.0)
    phiy, psiy, hhaty = _fitfit_axis(ny, 5, 4, float(ky[0]), 0.0)
    tmpx = sfft.ifft(hhatx[:, na] * sfft.fft(phix[:, na] * field, n=2 * nx, axis=0), axis=0)[:nx]
    field_x = 2.0 * (psix[:, na] * tmpx).real if sign == 1 else -2.0 * (psix[:, na] * tmpx).imag
    tmpy = sfft.ifft(hhaty[na, :] * sfft.fft(phiy[na, :] * field_x, n=2 * ny, axis=1), axis=1)[:, :ny]
    expected = 2.0 * (psiy[na, :] * tmpy).real if sign == 1 else 2.0 * (psiy[na, :] * tmpy).imag

    actual = FITFIT(field, sign, kx, ky, nx, 20, ny, 5)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_doKvector_tiles_edge_cases():
    from pyfuga.trafalgar import doKvector_tiles

    # Empty NNNs
    assert doKvector_tiles(512, 64, 20, 5, (), 0.1, 10.0) == []

    # kmax > kmin violation
    with pytest.raises(ValueError):
        doKvector_tiles(512, 64, 20, 5, (4,), 10.0, 0.1)

    # n_levels == 0: tile span exceeds kmax - kmin (small NNN = wide tile)
    assert doKvector_tiles(512, 64, 20, 5, (1,), 0.1, 0.2) == []
