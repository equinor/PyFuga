import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfuga.file_readers import read_lut_dat_file, read_lut_file
from pyfuga.trafalgar import Trafalgar

from .test_files import tfp


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
