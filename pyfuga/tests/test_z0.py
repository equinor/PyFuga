import matplotlib.pyplot as plt
import numpy as np
from pyfuga.tests import npt
from pyfuga.z0 import phi, psi, z0_from_TI


def test_phi():
    zeta = np.linspace(-.1, .1, 11)
    print(list(np.round(phi(zeta), 3)))
    if 0:
        plt.plot(zeta, phi(zeta))
        plt.show()
    npt.assert_array_almost_equal(phi(zeta), [0.764, 0.792, 0.825, 0.867, 0.922, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 3)


def test_psi():
    zeta = np.linspace(-.1, .1, 51)
    print(list(np.round(psi(zeta[::5]), 3)))
    print(list(np.round(psi(zeta[::5], 'Wilson'), 3)))
    if 0:
        plt.plot(zeta, psi(zeta))
        plt.plot(zeta, psi(zeta, 'Wilson'), label='Wilson')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal(psi(zeta[::5]),
                                  [0.326, 0.276, 0.221, 0.159, 0.087, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)
    npt.assert_array_almost_equal(psi(zeta[::5], 'Wilson'),
                                  [2.541, 2.488, 2.427, 2.355, 2.261, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5], 3)


def test_z0_from_TI():
    if 0:
        TI = np.linspace(3, 10, 100) / 100
        zref = 70
        plt.figure()

        z0_stable = z0_from_TI(TI, zref, 6e-7)
        z0_neutral = z0_from_TI(TI, zref, 0.0)
        z0_unstable = z0_from_TI(TI, zref, -6e-7)
        plt.plot(TI * 100, z0_stable, label='stable')
        plt.plot(TI * 100, z0_neutral, label='neutral')
        plt.plot(TI * 100, z0_unstable, label='unstable')

        z0_unstable = z0_from_TI(TI, zref, -6e-7)
        plt.plot(TI * 100, z0_unstable, '--', label='unstable')

        plt.xlabel('ti%')
        plt.ylabel('z0')
        plt.legend()
        plt.show()
    npt.assert_array_almost_equal([z0_from_TI([.06, .12], 70, zeta0) for zeta0 in [-6e-7, 0, 6e-7]],
                                  [[1e-05, 1.663e-02], [1e-05, 1.683e-02], [7.269e-05, 1.703e-02]], 5)
