import os

import pytest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from fuga.preluts import PreLUT, PreLUTs
from fuga.tests.test_files import tfp
from fuga.utils import get_beta


def test_load_prelut_file():

    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/0.0000-09.0000.pre',
                                  zeta0=0, beta=0, kz0=0, kzmax=0, ds=0.05)

    npt.assert_array_almost_equal(prelut.Yleft[7][0, 3], -0.330350424728106 + 9.114133411353441e-012j, 10)
    npt.assert_array_almost_equal(prelut.Rleft[7][0, 3], -4.045634236057177e-002 - 1.325470800079699e-010j, 10)
    npt.assert_array_almost_equal(prelut.Rright[7][0, 3], 3.768307078547423e-002 + 1.581796253544902e-010, 9)
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


def test_load_prelut_file_via_read_prelut_list():

    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/0.3333-07.0000.pre', zeta0=0)
    assert prelut.zeta0 == 0
    assert prelut.beta.item() == 0.829727913835271
    assert prelut.kz0 == 1e-7
    assert prelut.kzmax == 300
    assert prelut.ds == 0.05


def compare(res, ref, atol=1e-13, rtol=1e-9):
    npt.assert_array_equal(res.level, ref.level)
    for k in ref:
        # print(k)
        max_dims = ('j', 'k')[:len(ref[k].shape) - 1]
        max_idx = (0, 1, 2)[1:len(ref[k].shape)]
        try:
            npt.assert_allclose(res[k], ref[k], atol=atol, rtol=rtol, err_msg=k)
        except AssertionError:
            err = (ref[k] - res[k])

            rerr_real = np.where(np.abs(ref[k].real) > 1e-15, err.real / ref[k].real, np.nan)
            rerr_imag = np.where(np.abs(ref[k].imag) > 1e-15, err.imag / ref[k].imag, np.nan)
            ax1, ax2 = plt.subplots(2, 1)[1]
            plt.title(k)
            ax1.plot(np.abs(err.real).max(max_dims).values, label='Real, abs')
            ax2.plot(np.nanmax(np.abs(rerr_real), max_idx), label='Real, rel')
            # ax1.set_xlim([360, 375])

            ax1.plot(np.abs(err.imag).max(max_dims).values, label='Imag, abs')
            ax2.plot(np.nanmax(np.abs(rerr_imag), max_idx), label='Imag, rel')
            # ax1.set_xlim([360, 375])

            ax1.legend()
            ax2.legend()
            plt.ylabel = 'Error'
            plt.xlabel = 'Node'
            plt.show()
            raise


@pytest.mark.parametrize('zeta0', [0, -1, 1])
def test_prelut(zeta0):
    res = PreLUT.make_prelut(zeta0=zeta0, kz0=1e-9, beta=0,
                             kzmax=300, ds=0.05, accgoal=0.0001)
    prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre', zeta0=zeta0)
    assert res.ds == prelut.ds
    assert res.kzmax == prelut.kzmax
    compare(res, prelut)


def test_prelut_with_above_sm():
    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/0.0000-07.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-7, kzmax=0, ds=0.05)
    res = PreLUT.make_prelut(zeta0=0, kz0=1e-7, beta=get_beta(np.array([0]))[0],
                             kzmax=300, ds=0.05, accgoal=0.0001)
    compare(res, prelut)


def test_prelut_with_substations():
    # test will only pass if preludium compiled without optimization
    h_dict_debug = {18.15: 5.065748103785779E-003,
                    18.20: 9.834541888231090E-004,
                    18.25: 3.700851964819220E-003,
                    18.30: 3.782360731789152E-003,
                    18.35: 4.939093592985300E-003,
                    18.40: 5.628518632962573E-003}

    prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_1_2_debug/0.0000-06.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-6, kzmax=0, ds=0.05)

    res = PreLUT.make_prelut(zeta0=0, kz0=1e-6, beta=get_beta(np.array([0]))[0],
                             kzmax=300, ds=0.05, accgoal=0.0001, h_dict=h_dict_debug)
    compare(res, prelut, rtol=1e-7)


def test_prelut_save_load():

    if os.path.isfile('tmp.nc'):
        os.remove('tmp.nc')
    PreLUT.make_prelut(zeta0=0, kz0=1e-9, beta=get_beta(np.array([0]))[0],
                       kzmax=300, ds=0.05, accgoal=0.0001).save('tmp.nc')
    res = PreLUT.from_netcdf('tmp.nc')
    prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0=0.00E+00_2_5/0.0000-09.0000.pre',
                                  zeta0=0, beta=0, kz0=1e-9, kzmax=0, ds=0.05)
    compare(res, prelut)


def test_preluts():
    preluts = PreLUTs.from_pre_files(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/', zeta0=0, all_vars=False)
    ref = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_1_2.nc')
    ref.drop(['sleft', 'sright', 'dyxw0', 'dyxw1']).equals(preluts)

    prelut = preluts.isel(beta=1, kz0=1, i=7)

    npt.assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    npt.assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    npt.assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    npt.assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    npt.assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    npt.assert_array_equal(prelut.sleft, 0.35)
    npt.assert_array_almost_equal(prelut.sright, 0.4, 15)
    npt.assert_array_equal(prelut.level, 7)


def test_preluts_from_pre_files():
    zeta0 = -1
    preluts = PreLUTs.from_pre_files(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/', zeta0=zeta0)
    preluts_nc = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2.nc')
    assert preluts.equals(preluts_nc)
    ref = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre', zeta0)
    prelut = preluts.isel(beta=0, kz0=0)
    for k in prelut:
        v = prelut[k]
        if 'i' in v.dims:
            v = v[:ref[k].shape[0]]
        npt.assert_array_equal(v, ref[k])


def test_make_preluts():
    preluts_ref = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_1_2.nc')
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-9, 1e-8], beta_lst=preluts_ref.beta[:2],
                                   kzmax=0.0000001, ds=0.05, accgoal=0.0001)
    prelut = preluts.isel(beta=1, kz0=1, i=7)

    npt.assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    npt.assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    npt.assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    npt.assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    npt.assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    npt.assert_array_equal(prelut.sleft, 0.35)
    npt.assert_array_almost_equal(prelut.sright, 0.4, 15)
    npt.assert_array_equal(prelut.level, 7)
