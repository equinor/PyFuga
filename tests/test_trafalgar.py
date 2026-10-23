from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyfuga.file_readers import read_lut_dat_file, read_lut_file
from pyfuga.trafalgar import Trafalgar

from .test_files import tfp


def test_trafalgar_make_luts_rejects_invalid_n_cpu() -> None:
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="None or >= 1"):
        Trafalgar(ds, nx=8, ny=8, verbose=False).make_luts(n_cpu=0)


@pytest.mark.parametrize("var", ["UL"])
def test_trafalgar_n_cpu_1_does_not_instantiate_pool(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    workers_seen: list[int] = []

    @contextmanager
    def spy_map(
        workers: int, chunksize: int = 1
    ) -> Iterator[Callable[[Callable[..., object], Iterable[object]], Iterator[object]]]:
        _ = chunksize
        workers_seen.append(workers)
        yield map

    monkeypatch.setattr("pyfuga.trafalgar.multiprocessing_map", spy_map)

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
        workers: int, chunksize: int = 1
    ) -> Iterator[Callable[[Callable[..., object], Iterable[object]], Iterator[object]]]:
        _ = chunksize
        workers_seen.append(workers)
        yield map

    monkeypatch.setattr("pyfuga.trafalgar.multiprocessing_map", spy_map)

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
    # capfd (not capsys) so stderr from spawned worker processes is captured too
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


def test_doKvector_tiles_edge_cases():
    from pyfuga.trafalgar import doKvector_tiles

    # Empty NNNs
    assert doKvector_tiles(512, 64, 20, 5, (), 0.1, 10.0) == []

    # kmax > kmin violation
    with pytest.raises(ValueError):
        doKvector_tiles(512, 64, 20, 5, (4,), 10.0, 0.1)

    # n_levels == 0: tile span exceeds kmax - kmin (small NNN = wide tile)
    assert doKvector_tiles(512, 64, 20, 5, (1,), 0.1, 0.2) == []
