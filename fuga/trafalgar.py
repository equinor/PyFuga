import numpy as np
from numpy import newaxis as na
from scipy.interpolate import RectBivariateSpline
from fuga.constants import kappa
import xarray as xr
from tqdm import tqdm
M_PI = 3.141592653589793


class Trafalgar():
    def __init__(self, fourier_luts, nx=2048, ny=512, dx=20, dy=5, sigmax=80, sigmay=20, verbose=True):

        self.fourier_luts = fourier_luts
        self.nx = nx
        self.ny = ny // 2  #
        self.dx = dx
        self.dy = dy
        self.sigmax = sigmax
        self.sigmay = sigmay

        self.x_lst = np.arange(-self.nx // 4, self.nx * 3 / 4) * self.dx  # rotor is located 1/4 downstream
        self.y_lst = np.arange(self.ny) * self.dy
        self.verbose = verbose

        # self.load_input(input_filename)
        # self.initializeLUT()

    # def load_input(self, input_filename):
    #     with open(input_filename) as fid:
    #         lines = fid.readlines()
    #     self.label = lines[0].strip()
    #     self.FIToption, self.nx, self.ny = map(int, lines[1:4])
    #     if self.FIToption != 2:
    #         raise NotImplementedError
    #     if self.FIToption == 2:
    #         self.ny //= 2
    #     self.dx, self.dy, self.sigmaX, self.sigmaY, self.z0, self.zi, self.zh = map(float, lines[4:11])
    #     self.levelLow, self.levelHigh = map(int, lines[11:13])
    #     self.parentDirectory, self.LUTDirectory, self.prelutFile, self.var, self.LUTFormat = map(
    #         lambda s: s.strip(), lines[13:18])
    #     assert self.LUTFormat.lower() == 'binary'

    # def initializeLUT(self):
    #     with open(self.parentDirectory + self.prelutFile, 'rb') as fid:
    #         fid.read(127 + 4 + 8 + 8 + 4)  # str,i,d,d,i
    #         downE = struct.unpack('i', fid.read(4))[0]
    #         upE = struct.unpack('i', fid.read(4))[0]
    #         self.nkz0 = (upE - downE) + 1
    #         self.nbeta = struct.unpack('i', fid.read(4))[0] + struct.unpack('i', fid.read(4))[0] + 1
    #         fid.read(8)  # double
    #         self.ds = struct.unpack('d', fid.read(8))[0]
    #         fid.read(8 + 8)  # double, double
    #         self.betaList = np.fromfile(fid, float, self.nbeta)
    #         # plt.plot(self.betaList)
    #         self.kz0List = np.fromfile(fid, float, self.nkz0)
    #         # plt.plot(self.kz0List)
    #
    #     # Layer information
    #     lowerLevel = 0
    #     upperLevel = int(np.ceil(np.log(self.zi / self.z0) / self.ds))
    #
    #     if self.levelLow == self.levelHigh == 9999:
    #         print("Hub height chosen...")
    #     else:
    #         if self.levelLow < lowerLevel or self.levelLow > upperLevel:
    #             print("ERROR!!! - %s is not in range. Limits are %s and %s" % (self.levelLow, lowerLevel, upperLevel))
    #             sys.exit()
    #         print("Levels of interest are: ")
    #         for i in range(self.levelLow, self.levelHigh + 1):
    #             print("%d corresponding to height" % i, self.z0 * np.exp(self.ds * i))

    def make_luts(self, interpolation_order=3):
        # if self.levelLow < 0 or self.levelHigh > 10000:
        #     print("ERROR - level not legal!!!")
        #     sys.exit()
        fluts = self.fourier_luts
        beta_lst = fluts.beta.values
        kz0_lst = fluts.kz0.values

        sign_dict = {"UL": 1, "VL": -1, "WL": 1, "PL": 1, "UT": -1, "VT": 1, "WT": -1, "PT": -1}
        D, zhub, z0 = fluts.diameter.item(), fluts.hubheight.item(), fluts.z0.item()
        luts_dict = {}
        for var, sign in tqdm(sign_dict.items(), desc='Trafalgar', disable=(not self.verbose)):
            if var in fluts:

                kx = self.doKvectorFIT(self.nx, self.dx, 4)
                ky = self.doKvectorFIT(self.ny, self.dy, 4)

                # set forcing

                bx = M_PI / (self.dx * self.nx) / 4
                by = M_PI / (self.dy * self.ny) / 4
                U = np.log(zhub / z0) / kappa
                fuzz = U * bx * by * self.nx * self.ny / (M_PI * M_PI)
                force = np.exp(-0.5 * (self.sigmay * ky[na])**2) * np.exp(-0.5 * (self.sigmax * kx[:, na])**2)

                klen = np.sqrt(kx[:, na]**2 + ky[na]**2)

                angleTab = np.arctan2(ky[na], kx[:, na])
                kz0Tab = np.log(z0 * klen)
                order = interpolation_order
                flut = fluts[var].values * sign
                luts = []
                for i in range(len(fluts.level)):
                    re, im = flut[:, :, i].real, flut[:, :, i].imag

                    field = (RectBivariateSpline(beta_lst, np.log(kz0_lst), re, kx=order, ky=order).ev(angleTab, kz0Tab) +
                             1j * RectBivariateSpline(beta_lst, np.log(kz0_lst), im, kx=order, ky=order).ev(angleTab, kz0Tab))
                    field *= force * fuzz

                    k0 = 0  # ??? TODO: Ask Søren if this is true
                    field[0, 0] = k0 * force[0, 0] * fuzz

                    luts.append(self.FITFIT(field, sign, kx, ky).real)
                luts_dict[var] = (('level', 'x', 'y'), luts)
                self.ny0 = self.ny // 2
        return xr.Dataset({**luts_dict, 'z': fluts.z}, coords={'x': self.x_lst, 'y': self.y_lst, 'level': fluts.level},
                          attrs=self.fourier_luts.attrs)

    def doKvectorFIT(self, n, delta, NNN):
        c = M_PI / (n * delta) / NNN  # n*delta =nx*dx
        return c * (np.arange(n) + .5)

    def FITFIT(self, field, sign, kx, ky):
        nx, dx, ny, dy = self.nx, self.dx, self.ny, self.dy

        bx = M_PI / (dx * nx) / 4
        by = M_PI / (dy * ny) / 4
        abx = dx * bx
        aby = dy * by

        i0 = -nx / 4.

        i = np.arange(nx)
        phix = np.exp(1j * abx * (i * (i / 2. + i0)))
        psix = np.exp(1j * abx / 2. * (i + i * i + i0))

        j = np.arange(ny)
        phiy = np.exp(1j * aby * (j * j / 2.))
        psiy = np.exp(1j * aby / 2. * (j + j * j))

        i = np.arange(nx + 1)

        auxx = np.exp(-1j * abx / 2. * i * i)
        hhatxList = np.fft.ifft(np.r_[auxx, auxx[::-1][1:-1]])

        j = np.arange(ny + 1)
        auxy = np.exp(-1j * aby / 2. * j * j)
        hhatyList = np.fft.ifft(np.r_[auxy, auxy[::-1][1:-1]])

        # Over X
        ghatx = np.roll(np.fft.ifft(phix[:, na] * field, n=2 * nx, axis=0)[::-1], 1, axis=0)

        # fft here
        tmpx = np.fft.ifft(hhatxList[:, na] * ghatx, axis=0) * 2 * nx

        if(sign == 1):
            field = 2. * (psix[:, na] * tmpx[:nx]).real
        else:  # (sign==-1)
            field = -2. * (psix[:, na] * tmpx[:nx]).imag

        # Over Y
        ghaty = np.roll(np.fft.ifft(phiy[na, :] * field, n=2 * ny, axis=1)[:, ::-1], 1, 1)

        # fft here
        tmpy = np.fft.ifft(hhatyList[na, :] * ghaty, axis=1) * 2 * ny

        if(sign == 1):
            field = 2. * (psiy[na, :] * tmpy[:, :ny]).real
        else:  # (sign==-1)
            # minus here
            field = 2. * (psiy[na, :] * tmpy[:, :ny]).imag

        # plt.contourf(field.T)

        return field
