from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyfuga.constants import ACCGOAL, DS

PathLike = str | Path


def get_level_range(zlow: float, zhigh: float, zhub: float, z0: float) -> tuple[int, int]:
    if zlow == zhigh == zhub:
        return 9999, 9999
    low_level = int(np.floor(np.log(zlow / z0) / DS))
    high_level = int(np.ceil(np.log(zhigh / z0) / DS))
    return low_level, high_level


def get_luts_path(
    folder: PathLike,
    zeta0: float,
    nkz0: int,
    nbeta: int,
    diameter: float,
    zhub: float,
    z0: float,
    zi: float,
    zlow: float,
    zhigh: float,
    lut_vars: Sequence[str],
    nx: int,
    ny: int,
    dx: float | None,
    dy: float | None,
    *,
    single_precision: bool = False,
    accgoal: float = ACCGOAL,
) -> Path:
    fluts_id = get_fluts_path(
        folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh, lut_vars, accgoal=accgoal
    ).stem.removeprefix("fLUTs_")
    luts_id = fluts_id + f"_nx{nx}_ny{ny}_dx{dx}_dy{dy}" + ("_sp" if single_precision else "")
    return Path(folder) / f"LUTs_{luts_id}.nc"


def get_fluts_path(
    folder: PathLike,
    zeta0: float,
    nkz0: int,
    nbeta: int,
    diameter: float,
    zhub: float,
    z0: float,
    zi: float,
    zlow: float,
    zhigh: float,
    lut_vars: Sequence[str],
    *,
    accgoal: float = ACCGOAL,
) -> Path:
    preluts_id = get_preluts_path(folder, zeta0, nkz0, nbeta, accgoal=accgoal).stem.removeprefix("preLUTs_")

    L_vars = [v[0] for v in lut_vars if v[1] == "L"]
    T_vars = [v[0] for v in lut_vars if v[1] == "T"]
    lut_vars_id = ""
    if L_vars:
        lut_vars_id += f"_{''.join(L_vars)}L"
    if T_vars:
        lut_vars_id += f"_{''.join(T_vars)}T"

    low_level, high_level = get_level_range(zlow, zhigh, zhub, z0)
    if low_level == high_level == 9999:
        z_id = f"z{zhub:.1f}"
    else:
        zlow_out = z0 * np.exp(low_level * DS)
        zhigh_out = z0 * np.exp(high_level * DS)
        z_id = f"z{zlow_out:.1f}-{zhigh_out:.1f}"

    fluts_id = preluts_id + f"_D{diameter}_zhub{zhub}_zi{zi}_z0={z0:.8f}_{z_id}{lut_vars_id}"
    return Path(folder) / f"fLUTs_{fluts_id}.nc"


def get_preluts_path(folder: PathLike, zeta0: float, nkz0: int, nbeta: int, *, accgoal: float = ACCGOAL) -> Path:
    acc_id = "" if accgoal == ACCGOAL else f"_acc{accgoal:g}"
    return Path(folder) / f"preLUTs_Zeta0={zeta0:3.2e}_{nkz0}_{nbeta}{acc_id}.nc"
