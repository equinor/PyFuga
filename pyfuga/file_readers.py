"""file_readers.py
================
Read Fortran binary output files from the original Fuga code.

This module provides readers for legacy .bin output files containing
parameters, case data, and preLUT tables saved by the Fortran Fuga implementation.
Used for initialising preLUTs from existing Fuga runs and for compatibility
with legacy simulation data.
"""

import os
import struct
from io import BufferedReader
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import xarray as xr
from numpy import newaxis as na

PathLike: TypeAlias = str | os.PathLike[str]
PrelutTuple: TypeAlias = tuple[float, float, float, float, float, float]
PrelutListEntry: TypeAlias = tuple[str, float, float, float, float, float, float]


class BinaryReader:
    fid: BufferedReader

    def s(self, n: int) -> str:
        return self.fid.read(n).decode().strip()

    def i(self) -> int:
        return int(struct.unpack("i", self.fid.read(4))[0])

    def d(self) -> float:
        return float(struct.unpack("d", self.fid.read(8))[0])


class Parameters(BinaryReader):
    def __init__(self, folder: PathLike) -> None:
        i, d, s = self.i, self.d, self.s
        with open(Path(folder) / "parameters.bin", "rb") as self.fid:
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
    def __init__(self, folder: PathLike) -> None:
        i, d, s = self.i, self.d, self.s

        with open(Path(folder) / "CaseData.bin", "rb") as self.fid:
            self.case_name = s(127)
            self.radius = d()
            self.zhub = d()
            self.low_level_out = i()
            self.high_level_out = i()
            self.z0 = d()
            self.zi = d()
            self.ds = d()
            self.closure = i()


def eof(fid: BufferedReader) -> int:
    fid.seek(-1, 2)  # go to the file end.
    eof = fid.tell() + 1  # get the end of file location
    fid.seek(0, 0)  # go back to file beginning
    return eof


def read_prelut_list(folder: PathLike, dict: bool = True) -> dict[str, PrelutTuple] | list[PrelutListEntry]:

    folder = Path(folder)
    with open(folder / "prelut_list.lst", "rb") as fid:
        fid.seek(18)
        n = 24 if fid.read(6) == b"\x00\x00\x00\x00\x00\x00" else 18

        n_rows = eof(fid) // (n + 48)

        # list of filename (18 or 18+6xNull), ds, smaxx, kz0, beta, kzmax, accgoal
        if dict:
            return {fid.read(n)[:18].decode(): struct.unpack("d" * 6, fid.read(8 * 6)) for _ in range(n_rows)}

        else:
            return [(fid.read(n)[:18].decode(),) + struct.unpack("d" * 6, fid.read(8 * 6)) for _ in range(n_rows)]


def read_pre_file(filename: PathLike) -> dict[str, tuple[list[str], Any]]:
    with open(filename, "rb") as fid:
        d = np.fromfile(fid, float, -1).reshape((-1, 291))
        Yleft, Rleft, Rright, dy = np.moveaxis((d[:, :288:2] + d[:, 1:288:2] * 1j).reshape((-1, 4, 6, 6)), 1, 0)
        sleft, sright = d[:, 288:290].T
        fid.seek(0, 0)
        level = np.fromfile(fid, np.int32, -1).reshape((-1, 291 * 2))[:, -2]
    return {
        "Yleft": (["i", "j", "k"], Yleft),
        "Rleft": (["i", "j", "k"], Rleft),
        "Rright": (["i", "j", "k"], Rright),
        "dyxu0": (["i", "j"], dy[:, 0]),
        "dyxu1": (["i", "j"], dy[:, 1]),
        "dyxv0": (["i", "j"], dy[:, 2]),
        "dyxv1": (["i", "j"], dy[:, 3]),
        "dyxw0": (["i", "j"], dy[:, 4]),
        "dyxw1": (["i", "j"], dy[:, 5]),
        "sleft": (["i"], sleft),
        "sright": (["i"], sright),
        "level": (["i"], level),
    }


def read_lut_file(filename: PathLike, prelut_folder: PathLike) -> xr.Dataset:
    filename_path = Path(filename)
    parameters = Parameters(prelut_folder)
    casedata = CaseData(filename_path.parent)
    var = filename_path.name[:2]
    level = int(filename_path.name[-8:-4])
    z = casedata.zhub if level == 9999 else casedata.z0 * np.exp(casedata.ds * level)

    lut = np.fromfile(filename, complex, -1).reshape((len(parameters.kz0_lst) + 1, len(parameters.beta_lst)))
    lut = lut[1:].T
    return xr.Dataset(
        {
            var: (("beta", "kz0", "level"), lut[:, :, na]),
            "z": (("level"), [z]),
            "diameter": casedata.radius * 2,
            "hubheight": casedata.zhub,
            "z0": casedata.z0,
        },
        coords={"kz0": parameters.kz0_lst, "beta": parameters.beta_lst, "level": [level]},
    )


def read_lut_dat_file(filename: PathLike, nx: int, ny: int, dx: float, dy: float) -> xr.DataArray:
    dat = np.fromfile(filename, np.dtype("<f"), -1).astype(float).reshape((ny // 2, nx))
    x_lst = np.arange(-nx // 4, nx * 3 / 4) * dx  # rotor is located 1/4 downstream

    y_lst = np.arange(ny // 2) * dy
    return xr.DataArray(dat, dims=("y", "x"), coords={"x": x_lst, "y": y_lst})
