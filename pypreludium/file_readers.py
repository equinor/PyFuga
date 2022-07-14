from pathlib import Path
import struct
import numpy as np


class Parameters():
    def __init__(self, folder):
        folder = Path(folder)

        def i():
            return struct.unpack('i', fid.read(4))[0]

        def d():
            return struct.unpack('d', fid.read(8))[0]

        with open(folder / "parameters.bin", 'rb') as fid:
            self.prelutname = fid.read(127).decode().strip()
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
        def read_complex(shape):
            n = np.prod(shape)
            v = np.reshape(struct.unpack('d' * 2 * n, fid.read(16 * n)), shape + (2,))
            return np.sum(v * np.array([1, 1j]), -1)

        def read_level():
            r = ([read_complex((6, 6)) for _ in range(3)] +   # Yleft, Rleft, Rright
                 [read_complex((6,)) for _ in range(6)] +   # dyxu0, dyxu1, dyxv0, dyxv1, dyxw0, dyxw1
                 list(struct.unpack('ddi', fid.read(20))))  # sleft, sright, level
            struct.unpack('i', fid.read(4))
            return r
        r = []
        n = eof(fid)
        while fid.tell() < n:
            r.append(read_level())

    return {k: (dims, np.array(v)) for (k, dims), v in zip([
        ('Yleft', ['i', 'j', 'k']),
        ('Rleft', ['i', 'j', 'k']),
        ('Rright', ['i', 'j', 'k']),
        ('dyxu0', ['i', 'j']),
        ('dyxu1', ['i', 'j']),
        ('dyxv0', ['i', 'j']),
        ('dyxv1', ['i', 'j']),
        ('dyxw0', ['i', 'j']),
        ('dyxw1', ['i', 'j']),
        ('sleft', ['i']),
        ('sright', ['i']),
        ('level', ['i'])],
        zip(*r))}
