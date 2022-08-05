from ..lut import FourierLUTGenerator
from ..preluts import PreLUTs
from .test_files import tfp
from PyPreludium.pypreludium.file_readers import read_fourier_lut
from PyPreludium.pypreludium.tests import npt


def test_lut():
    preluts = PreLUTs.from_pre_files(tfp + 'ref/preLUTs_Zeta0=0.00E+00_1_1', zeta0=0)
    luts = FourierLUTGenerator(
        preluts,
        zhub=70,
        diameter=80,
        zi=400,
    ).make_lut(
        z0=0.00001,
        low_level_out=314,
        high_level_out=316,
        luts=['UL'])
    print(luts)
    ref = read_fourier_lut(tfp + 'ref/test_reference/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL0314.lut',
                           prelut_folder=tfp + 'ref/preLUTs_Zeta0=0.00E+00_1_1')
    npt.assert_array_almost_equal(ref.UL, luts.sel(level=ref.level).UL)
    print()


def test_lut_all():
    prelut = PreLUTs.from_pre_files(tfp + 'preLUTs_Zeta0=0.00E+00_1_2', zeta0=0)
    LUT(prelut, zhub=70, diameter=80, zi=400, LolevelOut=314, HilevelOut=316, WriteAllLuts=True).make_lut(z0=0.00001)
