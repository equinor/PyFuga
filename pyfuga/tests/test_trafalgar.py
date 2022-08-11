from pyfuga.file_readers import read_lut_file, read_lut_dat_file
from pyfuga.tests.test_files import tfp
from pyfuga.trafalgar import Trafalgar
import matplotlib.pyplot as plt
from pyfuga.tests import npt
import pytest


@pytest.mark.parametrize('var', ['UL', 'UT'])
def test_trafalgar(var):
    fourier_lut = read_lut_file(tfp + f'D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0314.lut',
                                prelut_folder=tfp + 'preLUTs_Zeta0=0.00E+00_2_5')

    fuga_luts = Trafalgar(fourier_lut, nx=512, ny=128).make_luts(3)

    fn = tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/D080.0000_zH070.0000_2_5_FIT0314{var}.dat"
    ref = read_lut_dat_file(fn, nx=512, ny=128, dx=20, dy=5)
    npt.assert_allclose(fuga_luts[var].sel(level=314).T, ref, rtol=1e-7, atol=1e-7)
    if 0:
        fuga_luts[var].plot()
        plt.show()
