from .test_files import tfp
from ..file_readers import read_prelut_list, Parameters


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
