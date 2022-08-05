from .test_files import tfp
from ..file_readers import read_prelut_list, Parameters
from PyPreludium.pypreludium.file_readers import read_fourier_lut, CaseData
from PyPreludium.pypreludium.tests import npt
import numpy as np


def test_parameters():
    p = Parameters(tfp + 'preLUTs_Zeta0=0.00E+00_1_2/')

    assert p.prelutname == 'preLUTs_Zeta0=0.00E+00_1_2'
    assert p.closure == 2
    assert p.kz0min == 1e-9
    assert p.kz0max == 1e-5
    assert p.nkz0 == 1
    assert p.jmin == -9
    assert p.jmax == -5
    assert p.nbeta == 2
    assert p.mbeta == 0
    assert p.ds == 0.05
    npt.assert_array_almost_equal(p.beta_lst, [1.469367938527859e-039,
                                               1.44768407985792,
                                               1.57079632679490])
    npt.assert_array_equal(p.kz0_lst, 10.**np.arange(-9, -4))


def test_casedata():
    c = CaseData(tfp + r'ref\test_reference\Z0=0.00001000Zi=00400Zeta0=0.00E+00')

    assert c.case_name == "Z0=0.00001000Zi=00400Zeta0=0.00E+00"
    assert c.radius == 40
    assert c.zhub == 70
    assert c.low_level_out == 314
    assert c.high_level_out == 316
    assert c.z0 == 1e-5
    assert c.zi == 400
    assert c.ds == 0.05
    assert c.closure == 2


def test_read_prelut_list():
    lst = read_prelut_list(tfp + 'preLUTs_Zeta0=0.00E+00_1_2', dict=False)
    filename, ds, smaxx, kz0, beta, kzmax, accgoal = lst[9]
    assert filename == '0.8000-05.0000.pre'
    assert ds == 0.05
    assert smaxx == 17.25000000000011
    assert kz0 == 1e-05
    assert beta == 1.4476840798579238
    assert kzmax == 300.0
    assert accgoal == 0.0001
    assert len(lst) == 15


def test_read_prelut_list_dict():
    d = read_prelut_list(tfp + 'preLUTs_Zeta0=0.00E+00_1_2')
    ds, smaxx, kz0, beta, kzmax, accgoal = d['0.8000-05.0000.pre']
    assert ds == 0.05
    assert smaxx == 17.25000000000011
    assert kz0 == 1e-05
    assert beta == 1.4476840798579238
    assert kzmax == 300.0
    assert accgoal == 0.0001


def test_read_fourier_lut():
    lut = read_fourier_lut(tfp + "ref/test_reference/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL0314.lut",
                           prelut_folder=tfp + "ref/preLUTs_Zeta0=0.00E+00_1_1")
