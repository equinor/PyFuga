import numpy as np

from .preluts import PreLUT
from .constants import zminlevel


class LUT():
    def __init__(self, prelut, zhub, diameter, zi, LolevelOut, HilevelOut, WriteAllLuts=True):
        self.prelut = prelut
        self.zhub = zhub
        self.diameter = diameter
        self.zi = zi  # Domain height
        self.LolevelOut = LolevelOut  # Lowest z_level for output, 9999=zHub
        self.HilevelOut = HilevelOut  # Highest z_level for output, 9999=zHub
        self.WriteAllLuts = WriteAllLuts

    def make_lut(self, z0):
        zh = self.zhub
        R = self.diameter / 2
        ds = self.prelut.ds
        upperjf = int(np.floor(np.log((zh + R) / z0) / ds))  # ceiling->floor
        lowerjf = int(np.ceil(np.log((zh - R) / z0) / ds))  # floor->ceiling
        minlevel = int(np.floor(np.log(np.maximum(zminlevel / z0, 1)) / ds))
        maxlevel = int(np.ceil(np.log(self.zi / z0) / ds))

        ji = maxlevel

        if self.LolevelOut > minlevel:  # pragma: no cover
            print(f'LoLevelOut ({self.LoLevelOut}) raised to MinLevel ({minlevel}).')
            self.LolevelOut = minlevel

        if self.HilevelOut > maxlevel:  # pragma: no cover
            print(f'HiLevelOut ({self.HiLevelOut}) lowered to MaxLevel ({maxlevel}).')
            self.HilevelOut = maxlevel

        jmin = self.prelut.kz0.min()
        jmax = self.prelut.kz0.max()
        mkz0 = jmax - jmin + 1
        if self.WriteAllLuts:
            nout = 6
        else:
            nout = 3


if __name__ == '__main__':
    from PyPreludium.tests.test_files import tfp
    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/0.0000-09.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-6, kzmax=0, ds=0.05)
    LUT(prelut, zhub=70, diameter=80, zi=400, LolevelOut=314, HilevelOut=316).make_lut(0.00001)
