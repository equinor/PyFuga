from pyfuga.file_readers import CaseData, read_lut_file, read_prelut_list, Parameters
from pyfuga.tests import npt
import numpy as np
from pyfuga.file_readers import read_lut_dat_file
import matplotlib.pyplot as plt
from pyfuga.tests.test_files import tfp


def test_parameters():
    p = Parameters(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/')

    assert p.prelutname == 'preLUTs_Zeta0=0.00E+00_2_5'
    assert p.closure == 2
    assert p.kz0min == 1e-9
    assert p.kz0max == 1e-1
    assert p.nkz0 == 2
    assert p.jmin == -18
    assert p.jmax == -2
    assert p.nbeta == 5
    assert p.mbeta == 0
    assert p.ds == 0.05
    npt.assert_array_almost_equal(p.beta_lst, [0., 0.829728, 1.295116, 1.515176, 1.552644, 1.570796])
    npt.assert_array_equal(p.kz0_lst, 10.**(np.arange(p.jmin, p.jmax + 1) / p.nkz0))


def test_casedata():
    c = CaseData(tfp + r'D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00')

    assert c.case_name == "Z0=0.00001000Zi=00400Zeta0=0.00E+00"
    assert c.radius == 40
    assert c.zhub == 70
    assert c.low_level_out == 314
    assert c.high_level_out == 315
    assert c.z0 == 1e-5
    assert c.zi == 400
    assert c.ds == 0.05
    assert c.closure == 2


def test_read_prelut_list():
    lst = read_prelut_list(tfp + 'preLUTs_Zeta0=0.00E+00_2_5', dict=False)
    filename, ds, smaxx, kz0, beta, kzmax, accgoal = lst[25]
    assert filename == '0.3333-05.0000.pre'
    assert ds == 0.05
    assert smaxx == 17.25000000000011
    assert kz0 == 1e-05
    assert beta == 0.829727913835271
    assert kzmax == 300.0
    assert accgoal == 0.0001
    assert len(lst) == 17 * 6


def test_read_prelut_list_smaxx():
    lst = read_prelut_list(tfp + 'preLUTs_Zeta0=0.00E+00_1_2', dict=False)
    npt.assert_array_almost_equal([l[2] for l in lst],
                                  [18.4, 18.4, 18.45, 18.45, 17.25, 18.4,
                                   18.4, 18.45, 18.45, 17.25, 18.4,
                                   18.4, 18.45, 18.45, 17.25])


def test_read_prelut_list_dict():
    d = read_prelut_list(tfp + 'preLUTs_Zeta0=0.00E+00_2_5', dict=True)
    ds, smaxx, kz0, beta, kzmax, accgoal = d['0.3333-05.0000.pre']
    assert ds == 0.05
    assert smaxx == 17.25000000000011
    assert kz0 == 1e-05
    assert beta == 0.829727913835271
    assert kzmax == 300.0
    assert accgoal == 0.0001
    assert len(d) == 17 * 6


def test_read_fourier_lut():
    lut = read_lut_file(tfp + "D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL0314.lut",
                        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5")
    npt.assert_almost_equal(lut.UL.isel(beta=0, kz0=0, level=0), -1642.12402757 + 569.48780155j)


def test_read_fourier_9999():
    lut = read_lut_file(tfp + "D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0=1.00E+00/UL9999.lut",
                        prelut_folder=tfp + "preLUTs_Zeta0=1.00E+00_1_2")
    assert lut.z == 70


def test_read_lut_dat():
    fn = tfp + "D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/D080.0000_zH070.0000_2_5_FIT0314UL.dat"
    lut = read_lut_dat_file(fn, nx=512, ny=128, dx=20, dy=5)
    assert lut.shape == (64, 512)
    if 0:
        plt.contourf(lut)
        plt.show()
