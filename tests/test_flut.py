import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pyfuga.constants import UVW_LT
from pyfuga.file_readers import read_lut_file
from pyfuga.flut import FourierLUTGenerator
from pyfuga.preluts import PreLUTs
from pyfuga.profiling import timeit
from pyfuga.utils import compile, get_beta_lst

from .test_files import tfp


@pytest.mark.parametrize('zeta0', [0, -1, 1])
def test_make_hubheight_luts(zeta0):
    ref = read_lut_file(tfp + f'D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0={zeta0}.00E+00/UL9999.lut',
                        prelut_folder=tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2')
    preluts = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0={zeta0}.00E+00_1_2.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_hubheight_luts(z0=0.00001, luts=['UL'])

    assert_array_almost_equal(ref.UL.sel(kz0=luts.kz0), luts.sel(level=ref.level).UL)

    assert_array_equal(ref.UL[:, len(luts.kz0):], 0)


def test_make_hubheight_luts_lo_eq_hi():
    preluts = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0=0.00E+00_1_2.nc')
    z0 = 70 / np.exp(315 * 0.05)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_hubheight_luts(z0=z0, luts=['UL'])
    ref = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                              ).make_lut(z0, low_level_out=315, high_level_out=315, luts=['UL'])

    assert_array_almost_equal(ref.UL, luts.UL)


def test_all_vars():
    preluts = PreLUTs.from_netcdf(tfp + f'preLUTs_Zeta0=0.00E+00_2_5.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_lut(z0=0.00001, low_level_out=314, high_level_out=314)
    for v in UVW_LT:
        ref = read_lut_file(tfp + f'D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{v}0314.lut',
                            prelut_folder=tfp + f'preLUTs_Zeta0=0.00E+00_2_5')
        assert_array_almost_equal(luts[v], ref[v][:, :len(luts.kz0)])
        assert_array_equal(0, ref[v][:, len(luts.kz0):])


def test_compact_preluts():
    preluts_compact = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-9, 1e-8], beta_lst=get_beta_lst(1),
                                           kzmax=300, ds=0.05, accgoal=0.00001, jit=False, verbose=False, compact=True)
    preluts = PreLUTs.make_preluts(zeta0=0, kz0_lst=[1e-9, 1e-8], beta_lst=get_beta_lst(1),
                                   kzmax=300, ds=0.05, accgoal=0.00001, jit=False, verbose=False, compact=False)
    luts_c = FourierLUTGenerator(preluts_compact, zhub=70, diameter=80, zi=400, verbose=False
                                 ).make_lut(z0=0.00001, low_level_out=314, high_level_out=314)
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_lut(z0=0.00001, low_level_out=314, high_level_out=314)

    for beta in luts.beta.values:
        for kz0 in luts.kz0.values:
            assert preluts.sel(beta=beta, kz0=kz0).equals(preluts_compact.sel(beta=beta, kz0=kz0))
    for k in UVW_LT:
        assert_array_almost_equal(luts[k], luts_c[k])


def test_rotor_luts():
    preluts = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_2_5.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_rotor_luts(z0=0.00001, luts=['UL'])
    assert luts.z[0] < 70 - 40
    assert luts.z[-1] > 70 + 40


def test_fluts_ncpu():
    preluts = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_2_5.nc')
    luts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                               ).make_rotor_luts(z0=0.00001, luts=['UL'])
    pluts = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False
                                ).make_rotor_luts(z0=0.00001, luts=['UL'], n_cpu=None)
    assert luts.equals(pluts)


def test_jit_luts():
    preluts = PreLUTs.from_netcdf(tfp + 'preLUTs_Zeta0=0.00E+00_2_5.nc')
    lut_generator = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False)
    r1, t1 = timeit(lut_generator.make_lut)(z0=0.00001, low_level_out=315, high_level_out=315, luts=['UL'])
    compile(jit=True)
    r2, t2 = timeit(lut_generator.make_lut, min_runs=2)(z0=0.00001, low_level_out=315, high_level_out=315, luts=['UL'])
    # print(t1, t2)
    assert t2[1] < t1[0]
    for k in r1:
        # print(k, r1[k].shape)
        assert_array_almost_equal(r1[k], r2[k], 10)
