import os

import pytest

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pyfuga.preluts import PreLUT, PreLUTs
from .test_files import tfp
from pyfuga.utils import get_beta, get_beta_lst, get_kz0_lst, ComplexXRDataset, compile
from pyfuga.flut import FourierLUTGenerator
from pyfuga.file_readers import read_lut_file
from pyfuga.constants import UVW_LT
from pyfuga import utils
from pyfuga.preluts_generator import PrelutNode, PreLUTGenerator
from pyfuga.profiling import timeit
import xarray as xr
import warnings
import time


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    compile(jit=True)


def teardown_module(module):
    compile(jit=False)


def test_prelut_neutral_all_vars():
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-9, 1e-8], beta_lst=get_beta_lst(1),
                                   kzmax=300, ds=0.05, accgoal=0.00001, verbose=False)
    ref_prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0=0.00E+00_2_5/0.0000-09.0000.pre', zeta0=0)

    assert ref_prelut.ds == preluts.ds
    assert ref_prelut.kzmax == preluts.kzmax
    # assert ref_prelut.accgoal == preluts.accgoal

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0=0.00001, low_level_out=315, high_level_out=315)

    for var in UVW_LT:
        ref = read_lut_file(tfp + f'D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0315.lut',
                            prelut_folder=tfp + f'preLUTs_Zeta0=0.00E+00_2_5')

        assert_allclose(
            ref.sel(kz0=flut.kz0, beta=flut.beta, method='nearest')[var].real, flut[var].real, rtol=1e-5, atol=1e-6
        )
        assert_allclose(
            ref.sel(kz0=flut.kz0, beta=flut.beta, method='nearest')[var].imag, flut[var].imag, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize('zeta0', [-1, 1])
def test_prelut_stable_and_unstable(zeta0):
    preluts = PreLUTs.make_preluts(zeta0=zeta0, kz0_lst=[1e-9], beta_lst=[0],
                                   kzmax=300, ds=0.05, accgoal=0.0001, verbose=False)
    ref_prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre', zeta0=zeta0)

    assert ref_prelut.ds == preluts.ds
    assert ref_prelut.kzmax == preluts.kzmax
    assert ref_prelut.accgoal == preluts.accgoal

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=0.00001, luts=['UL'])

    ref = read_lut_file(tfp + f'D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0={zeta0}.00E+00/UL9999.lut',
                        prelut_folder=tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2')

    assert_allclose(ref.sel(kz0=1e-9, beta=0).UL, flut.UL.item(), rtol=2e-5, atol=1e-10)


def test_next_node():
    node = PrelutNode()

    prelut = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0=0.00E+00_1_2/0.0000-09.0000.pre', zeta0=0)
    i = 1
    node.Yright = prelut.Yleft[i].values @ np.conj(prelut.Rleft[i].T).values
    next_node = node.get_next(.05, .1)
    assert next_node.sleft == 0.05
    assert next_node.sright == .1

    # print(np.abs(next_node.Yleft @ np.conj(next_node.Rleft.T) - node.Yright).max())
    assert_allclose(next_node.Yleft @ np.conj(next_node.Rleft.T), node.Yright, rtol=1e-6, atol=1e-15)
    # print(np.abs(node.Rright @ next_node.Rleft - np.eye(6)).max())
    assert_allclose(node.Rright @ next_node.Rleft, np.eye(6), atol=1e-16)


def test_prelut_with_above_sm():
    # prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/0.0000-07.0000.pre',
    #                               zeta0=0, beta=0, kz0=1e-7, kzmax=0, ds=0.05)
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-7], beta_lst=get_beta(np.array([0]))[:1],
                                   kzmax=300, ds=0.05, accgoal=0.00001, verbose=False)
    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_hubheight_luts(z0=0.00001, luts=['UL'])
    ref = read_lut_file(tfp + f'D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL9999.lut',
                        prelut_folder=tfp + f'preLUTs_Zeta0=0.00E+00_1_2')

    assert_allclose(flut.UL.item(), ref.sel(kz0=1e-7, beta=0).UL)


def test_prelut_with_substations():
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-6], beta_lst=get_beta(np.array([0]))[:1],
                                   kzmax=300, ds=0.05, accgoal=0.0001, verbose=False)
    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_hubheight_luts(z0=0.00001, luts=['UL'])
    ref = read_lut_file(tfp + f'D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL9999.lut',
                        prelut_folder=tfp + f'preLUTs_Zeta0=0.00E+00_1_2')

    assert_array_almost_equal(ref.sel(kz0=1e-6, beta=0).UL, flut.UL.item())


def test_preluts():
    preluts = PreLUTs.from_pre_files(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/', zeta0=0, verbose=False)
    ref = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_1_2.nc')
    ref.equals(preluts)
    prelut = preluts.isel(beta=1, kz0=1, i=7)

    assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    assert_array_equal(prelut.sleft, 0.35)
    assert_array_almost_equal(prelut.sright, 0.4, 15)
    assert_array_equal(prelut.level, 7)


def test_preluts_from_pre_files():
    zeta0 = -1
    preluts = PreLUTs.from_pre_files(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/', zeta0=zeta0, all_vars=False,
                                     verbose=False)
    preluts_nc = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2.nc')
    assert preluts_nc.drop_vars(['sleft', 'sright', 'dyxw0', 'dyxw1']).equals(preluts)
    ref = PreLUT.from_pre_file(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre', zeta0)
    prelut = preluts.isel(beta=0, kz0=0)
    for k in prelut:
        v = prelut[k]
        if 'i' in v.dims:
            v = v[:ref[k].shape[0]]
        assert_array_equal(v, ref[k])


def test_make_preluts():
    preluts_ref = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_1_2.nc')
    kz0_lst = [1e-9, 1e-8]
    beta_lst = preluts_ref.beta[:2].values
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=kz0_lst, beta_lst=beta_lst,
                                   kzmax=0.0000001, ds=0.05, accgoal=0.0001, verbose=False)
    prelut = preluts.sel(beta=beta_lst[1], kz0=kz0_lst[1], i=7)

    assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    # not equal due to new ortogonalization
    # assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    assert_array_equal(prelut.sleft, 0.35)
    assert_array_almost_equal(prelut.sright, 0.4, 15)
    assert_array_equal(prelut.level, 7)


def test_make_preluts_jit():

    kwargs = dict(zeta0=0, kz0=1e-1, beta=0, kzmax=300, ds=0.05, accgoal=0.0001)
    if 1:  # not os.path.isfile('flut1.nc'):
        utils.compile(jit=False)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from pyfuga import preluts_generator_nojit
        r1, t1 = timeit(preluts_generator_nojit.PreLUTGenerator(**kwargs).make_prelut, min_runs=1)()
        r1 = xr.combine_by_coords([ds.assign_coords(i=ds.i, kz0=ds.kz0, beta=ds.beta).expand_dims(('kz0', 'beta'))
                                   for ds in [r1]])
        flut1 = FourierLUTGenerator(r1, zhub=70, diameter=80, zi=400, verbose=False).make_lut(z0=0.1,
                                                                                              low_level_out=47,
                                                                                              high_level_out=165)
        flut1.to_netcdf('flut1.nc')
    else:
        flut1 = ComplexXRDataset.from_netcdf('flut1.nc')

    utils.compile(jit=True)
    r2, t2 = timeit(PreLUTGenerator(**kwargs).make_prelut, min_runs=2)()

    # timeit(PreLUTGenerator(**kwargs).make_prelut, verbose=1, line_profile=True, profile_funcs=[
    #     PreLUTGenerator.make_prelut,
    # ])()

    r2 = xr.combine_by_coords([ds.assign_coords(i=ds.i, kz0=ds.kz0, beta=ds.beta).expand_dims(('kz0', 'beta'))
                               for ds in [r2]])
    flut2 = FourierLUTGenerator(r2, zhub=70, diameter=80, zi=400, verbose=False).make_lut(z0=0.1,
                                                                                          low_level_out=47,
                                                                                          high_level_out=165)

    assert t2 < t1
    for k in flut1:
        # print(k, flut1[k].shape)
        assert_array_almost_equal(flut1[k], flut2[k], 10)


def test_compile():
    # second compile should be fast
    t = time.time()
    utils.compile(True)
    assert time.time() - t < .01


def test_make_preluts_jit_parallel():

    # r1, t1 = timeit(PreLUTs.make_preluts)(zeta0=0, kz0_lst=get_kz0_lst(1, 1e-9, 1e-1), beta_lst=get_beta_lst(1),
    #                                       kzmax=300, ds=0.05, accgoal=0.0001)
    # print(t1)
    # utils.compile(jit=True)
    # r2, t2 = timeit(PreLUTs.make_preluts)(zeta0=0, kz0_lst=get_kz0_lst(1, 1e-9, 1e-1), beta_lst=get_beta_lst(1),
    #                                       kzmax=300, ds=0.05, accgoal=0.0001)
    # print(t2)

    r3, t3 = timeit(PreLUTs.make_preluts)(zeta0=0, kz0_lst=get_kz0_lst(1, 1e-9, 1e-1), beta_lst=get_beta_lst(1),
                                          kzmax=300, ds=0.05, accgoal=0.0001, n_cpu=None)
    print(t3)
