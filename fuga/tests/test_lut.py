from fuga.tests import npt
from fuga.file_readers import read_lut_file

from fuga.preluts import PreLUTs
from fuga.lut import FourierLUTGenerator
from fuga.tests.test_files import tfp
import pytest
from fuga.constants import UVW_LT
from fuga import utils
import time


@pytest.mark.parametrize('zeta0', [0, -1, 1])
def test_make_hubheight_luts(zeta0):
    ref = read_lut_file(tfp + f'D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0={zeta0}.00E+00/UL9999.lut',
                        prelut_folder=tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2')
    preluts = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400,).make_hubheight_luts(z0=0.00001, luts=['UL'])

    npt.assert_array_almost_equal(ref.UL.sel(kz0=luts.kz0), luts.sel(level=ref.level).UL)

    npt.assert_array_equal(ref.UL[:, len(luts.kz0):], 0)


def test_all_vars():
    preluts = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0=0.00E+00_2_5.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400,
                               ).make_lut(z0=0.00001, low_level_out=314, high_level_out=314)
    for v in UVW_LT:
        ref = read_lut_file(tfp + f'D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{v}0314.lut',
                            prelut_folder=tfp + f'preLUTs_Zeta0=0.00E+00_2_5')
        npt.assert_array_almost_equal(luts[v], ref[v][:, :len(luts.kz0)])
        npt.assert_array_equal(0, ref[v][:, len(luts.kz0):])


def test_rotor_luts():
    preluts = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_2_5.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400).make_rotor_luts(z0=0.00001, luts=['UL'])
    assert luts.z[0] < 70 - 40
    assert luts.z[-1 > 70 + 40]


def test_jit_luts():
    preluts = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_2_5.nc')
    for jit in [False, True]:
        utils.numba_jit = jit
        lut_generator = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400)
        lut_generator.verbose = False
        t = time.time()
        lut_generator.make_lut(z0=0.00001, low_level_out=315, high_level_out=315, luts=['UL'])
        print(f'Jit: {jit}: {time.time() - t}s')
