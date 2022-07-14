from ..lut import LUT
from ..preluts import PreLUT
from .test_files import tfp


def test_lut():
    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/0.0000-06.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-6, kzmax=0, ds=0.05)
    LUT(prelut, zhub=70, diameter=80, zi=300, LolevelOut=314, HilevelOut=316, WriteAllLuts=False)


def test_lut_all():
    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/0.0000-06.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-6, kzmax=0, ds=0.05)
    LUT(prelut, zhub=70, diameter=80, zi=300, LolevelOut=314, HilevelOut=316, WriteAllLuts=True)
