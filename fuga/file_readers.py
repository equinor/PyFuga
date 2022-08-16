from pathlib import Path
import struct
import numpy as np
import os
import xarray as xr
from numpy import newaxis as na


class BinaryReader():
    def s(self, n):
        return self.fid.read(n).decode().strip()

    def i(self):
        return struct.unpack('i', self.fid.read(4))[0]

    def d(self):
        return struct.unpack('d', self.fid.read(8))[0]


class Parameters(BinaryReader):
    def __init__(self, folder):
        i, d, s = self.i, self.d, self.s
        with open(Path(folder) / "parameters.bin", 'rb') as self.fid:
            self.prelutname = s(127)
            self.closure = i()
            self.kz0min = d()
            self.kz0max = d()
            self.nkz0 = i()
            self.jmin = i()
            self.jmax = i()
            self.nbeta = i()
            self.mbeta = i()
            self.dummy2 = d()
            self.ds = d()
            self.kzmax = d()
            self.accgoal = d()
            self.beta_lst = np.fromfile(self.fid, float, self.nbeta + self.mbeta + 1)
            self.kz0_lst = np.fromfile(self.fid, float, self.jmax - self.jmin + 1)


class CaseData(BinaryReader):
    def __init__(self, folder):
        i, d, s = self.i, self.d, self.s

        with open(Path(folder) / "CaseData.bin", 'rb') as self.fid:
            self.case_name = s(127)
            self.radius = d()
            self.zhub = d()
            self.low_level_out = i()
            self.high_level_out = i()
            self.z0 = d()
            self.zi = d()
            self.ds = d()
            self.closure = i()


def eof(fid):
    fid.seek(-1, 2)     # go to the file end.
    eof = fid.tell() + 1   # get the end of file location
    fid.seek(0, 0)      # go back to file beginning
    return eof


def read_prelut_list(folder, dict=True):

    folder = Path(folder)
    with open(folder / 'prelut_list.lst', 'rb') as fid:
        N = int(eof(fid) / 72)

        # list of filename (24, ignore last 6), ds, smaxx, kz0, beta, kzmax, accgoal
        if dict:
            return {fid.read(24)[:18].decode():
                    struct.unpack('d' * 6, fid.read(8 * 6)) for _ in range(N)}
        else:
            return [(fid.read(24)[:18].decode(),) +
                    struct.unpack('d' * 6, fid.read(8 * 6)) for _ in range(N)]


def read_pre_file(filename):
    with open(filename, 'rb') as fid:

        # def read_complex(shape):
        #     n = np.prod(shape)
        #     v = np.reshape(struct.unpack('d' * 2 * n, fid.read(16 * n)), shape + (2,))
        #     return np.sum(v * np.array([1, 1j]), -1)
        #
        # def read_level():
        #     r = ([read_complex((6, 6)) for _ in range(3)] +   # Yleft, Rleft, Rright
        #          [read_complex((6,)) for _ in range(6)] +   # dyxu0, dyxu1, dyxv0, dyxv1, dyxw0, dyxw1
        #          list(struct.unpack('ddi', fid.read(20))))  # sleft, sright, level
        #     struct.unpack('i', fid.read(4))
        #     return r
        #
        # def read_level2():
        #     # Yleft,Rleft,Rright,(dyxu0, dyxu1, dyxv0, dyxv1, dyxw0, dyxw1),sleft, sright, level
        #     r = np.fromfile(fid, np.complex128, 144).reshape((4, 6, 6)), list(struct.unpack('ddi', fid.read(20)))
        #     struct.unpack('i', fid.read(4))
        #     return r
        # r = []
        # n = eof(fid)
        # # while fid.tell() < n:
        # #     r.append(read_level())

        d = np.fromfile(fid, float, -1).reshape((-1, 291))
        Yleft, Rleft, Rright, dy = np.moveaxis((d[:, :288:2] + d[:, 1:288:2] * 1j).reshape((-1, 4, 6, 6)), 1, 0)
        sleft, sright = d[:, 288:290].T
        fid.seek(0, 0)
        level = np.fromfile(fid, int, -1).reshape((-1, 291 * 2))[:, -2]
    return {'Yleft': (['i', 'j', 'k'], Yleft),
            'Rleft': (['i', 'j', 'k'], Rleft),
            'Rright': (['i', 'j', 'k'], Rright),
            'dyxu0': (['i', 'j'], dy[:, 0]),
            'dyxu1': (['i', 'j'], dy[:, 1]),
            'dyxv0': (['i', 'j'], dy[:, 2]),
            'dyxv1': (['i', 'j'], dy[:, 3]),
            'dyxw0': (['i', 'j'], dy[:, 4]),
            'dyxw1': (['i', 'j'], dy[:, 5]),
            'sleft': (['i'], sleft),
            'sright': (['i'], sright),
            'level': (['i'], level),
            }
    # return r
    #
    # return {k: (dims, np.array(v)) for (k, dims), v in zip([
    #     ('Yleft', ['i', 'j', 'k']),
    #     ('Rleft', ['i', 'j', 'k']),
    #     ('Rright', ['i', 'j', 'k']),
    #     ('dyxu0', ['i', 'j']),
    #     ('dyxu1', ['i', 'j']),
    #     ('dyxv0', ['i', 'j']),
    #     ('dyxv1', ['i', 'j']),
    #     ('dyxw0', ['i', 'j']),
    #     ('dyxw1', ['i', 'j']),
    #     ('sleft', ['i']),
    #     ('sright', ['i']),
    #     ('level', ['i'])],
    #     zip(*r))}


def read_lut_file(filename, prelut_folder):
    parameters = Parameters(prelut_folder)
    casedata = CaseData(os.path.dirname(filename))
    var = os.path.basename(filename)[:2]
    level = int(filename[-8:-4])
    if level == 9999:
        z = casedata.zhub
    else:
        z = casedata.z0 * np.exp(casedata.ds * level)
    # if var in ["UL", "VT", "WL", "PL"]:
    #     sign = 1
    # elif var in ["UT", "VL", "WT", "PT"]:
    #     sign = -1
    # else:
    #     print("ERROR - illegal variable ")
    #     sys.exit()

    # lut = sign * np.fromfile(filename, complex, -1).reshape((len(parameters.kz0_lst) + 1, len(parameters.beta_lst)))
    lut = np.fromfile(filename, complex, -1).reshape((len(parameters.kz0_lst) + 1, len(parameters.beta_lst)))
    lut = lut[1:].T
    return xr.Dataset({var: (('beta', 'kz0', 'level'), lut[:, :, na]), 'z': (('level'), [z]),
                       'diameter': casedata.radius * 2, 'hubheight': casedata.zhub, 'z0': casedata.z0},
                      coords={'kz0': parameters.kz0_lst, 'beta': parameters.beta_lst, 'level': [level]},
                      )


def read_lut_dat_file(filename, nx, ny, dx, dy):
    dat = np.fromfile(filename, np.dtype('<f'), -1).astype(float).reshape((ny // 2, nx))
    x_lst = np.arange(-nx // 4, nx * 3 / 4) * dx  # rotor is located 1/4 downstream

    y_lst = np.arange(ny // 2) * dy
    return xr.DataArray(dat, dims=('y', 'x'), coords={'x': x_lst, 'y': y_lst})
