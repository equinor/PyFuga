from pathlib import Path

import numpy as np

from pyfuga.constants import DS


def get_level_range(zlow, zhigh, zhub, z0):
    if zlow == zhigh == zhub:
        return 9999, 9999
    low_level = int(np.floor(np.log(zlow / z0) / DS))
    high_level = int(np.ceil(np.log(zhigh / z0) / DS))
    return low_level, high_level


def get_luts_path(folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh, lut_vars, nx, ny, dx, dy):
    fluts_id = get_fluts_path(
        folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh, lut_vars
    ).stem.removeprefix("fLUTs_")
    luts_id = fluts_id + f"_nx{nx}_ny{ny}_dx{dx}_dy{dy}"
    return Path(folder) / f"LUTs_{luts_id}.nc"


def get_fluts_path(folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh, lut_vars):
    preluts_id = get_preluts_path(folder, zeta0, nkz0, nbeta).stem.removeprefix("preLUTs_")

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


def get_preluts_path(folder, zeta0, nkz0, nbeta):
    return Path(folder) / f"preLUTs_Zeta0={zeta0:3.2e}_{nkz0}_{nbeta}.nc"
