from PyPreludium.utils import get_beta
from PyPreludium.preluts import PreLUT
import numpy as np

from PyPreludium.tests.test_files import tfp
from numpy import newaxis as na
import numpy.testing as npt
import matplotlib.pyplot as plt
import pytest


def test_load_prelut_file():

    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/0.0000-09.0000.pre',
                                  zeta0=0, beta=0, kz0=0, kzmax=0, ds=0.05)

    npt.assert_array_almost_equal(prelut.Yleft[7][0, 3], -0.330350424728106 + 9.114133344026951e-012j, 10)
    npt.assert_array_almost_equal(prelut.Rleft[7][1, 4], -0.139000795854240 - 1.272631906082874E-010j, 10)
    npt.assert_array_almost_equal(prelut.Rright[7][0, 3], 3.768307078547423e-002 + 1.581796234292119e-010j, 10)
    npt.assert_array_almost_equal(prelut.dyxu0[7][1], -2.423313312692920e-011 - 1.267852505176752e-021j, 10)
    npt.assert_array_almost_equal(prelut.dyxu1[7][1], -3.528960438325056e-020 - 1.849660676125115e-030j, 10)
    npt.assert_array_almost_equal(prelut.dyxv0[7][3], 6.806655201915466e-011 - 7.883109477872014e-021, 10)
    npt.assert_array_almost_equal(prelut.dyxv1[7][3], 9.905991452339835e-020 - 1.147145744025741e-029, 10)
    npt.assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    npt.assert_array_almost_equal(prelut.dyxw1[7][2], 2.912345039036901e-039 - 3.093036809788905e-030, 10)
    npt.assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    npt.assert_array_almost_equal(prelut.sleft[7], 0.35, 10)
    npt.assert_array_almost_equal(prelut.sright[7], 0.4, 10)
    npt.assert_array_almost_equal(prelut.level[7], 7, 10)


@pytest.mark.parametrize('zeta0', [0, -1, 1])
def test_prelut(zeta0):
    res = PreLUT.make_prelut(zeta0=zeta0, kz0=1e-9, beta=get_beta(np.array([0]))[0],
                             kzmax=300, ds=0.05, accgoal=0.0001)
    prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre',
                                  zeta0=zeta0, beta=0, kz0=1e-9, kzmax=0, ds=0.05)
    npt.assert_array_equal(res.level, prelut.level)
    for k in prelut:
        # print(k)
        max_dims = ('j', 'k')[:len(prelut[k].shape) - 1]
        if 0 and len(prelut[k].shape):
            np.abs(prelut[k].real - res[k].real).max(max_dims).plot()
            np.abs(prelut[k].imag - res[k].imag).max(max_dims).plot()
            plt.show()
        if k[0] != 'd':
            npt.assert_allclose(res[k], prelut[k], atol=1e-10, rtol=1e-6, err_msg=k)
        else:
            npt.assert_allclose(res[k], prelut[k], atol=5e-3, rtol=1e-4, err_msg=k)


def test_prelut_with_substations():
    res = PreLUT.make_prelut(zeta0=0, kz0=1e-9, beta=get_beta(np.array([0]))[0],
                             kzmax=300, ds=0.05, accgoal=0.0001)
    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/0.0000-06.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-6, kzmax=0, ds=0.05)
    npt.assert_array_equal(res.level, prelut.level)
    for k in prelut:
        print(k)
        max_dims = ('j', 'k')[:len(prelut[k].shape) - 1]
        if 0 and len(prelut[k].shape):
            np.abs(prelut[k].real - res[k].real).max(max_dims).plot()
            np.abs(prelut[k].imag - res[k].imag).max(max_dims).plot()
            plt.show()
        if k[0] != 'd':
            npt.assert_allclose(res[k], prelut[k], atol=1e-10, rtol=1e-6, err_msg=k)
        else:
            npt.assert_allclose(res[k], prelut[k], atol=5e-3, rtol=1e-4, err_msg=k)
