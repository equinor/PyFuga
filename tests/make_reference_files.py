import os
import sys
from pathlib import Path

from pyfuga.preluts import PreLUTs

from .test_files import tfp

sys.path.append(r"C:\mmpe\programming\python\Topfarm\CuttingEdge\Fuga\Easylut")  # nopep8
from run import run

os.chdir(r"C:\mmpe\programming\python\Topfarm\CuttingEdge\Fuga\Easylut")
Path("lut_path.txt").write_text(tfp)


# neutral, stable and unstable, level 9999, nbeta=2, nkz0=5
run(
    **{
        "D_zH_z0_unique": [[80, 70, 1e-05]],
        "nbeta": 2,
        "mbeta": 0,
        "nkz0": 1,
        "out_zhigh": 9999,
        "out_zlow": 9999,
        "writeall": False,
        "zetas": [0, 1, -1],
        "zi": 400,
        "kz0min": 1e-9,  # Lowest value of k*z0
        "kz0max": 1e-5,  # Highest value of k*z0
        "lut_path": tfp,
        "turbine": "D080.0000_zH070.0000_1_2_9999",
        "interpolation_order": 1,
        "Nxout": 512,
        "Nyout": 128,
    }
)
for zeta0 in [0, 1, -1]:
    preluts = PreLUTs.from_pre_files(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2/", zeta0=zeta0)
    preluts.to_netcdf(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2.nc")

# neutral, level 314-315, nbeta=5, nkz=17
run(
    **{
        "D_zH_z0_unique": [[80, 70, 1e-05]],
        "nbeta": 5,
        "mbeta": 0,
        "nkz0": 2,
        "out_zhigh": 67,
        "out_zlow": 67,
        "writeall": True,
        "zetas": [0],
        "zi": 400,
        "kz0min": 1e-9,  # Lowest value of k*z0
        "kz0max": 1e-1,  # Highest value of k*z0
        "lut_path": tfp,
        "turbine": "D080.0000_zH070.0000_2_5",
        "Nxout": 512,
        "Nyout": 128,
    }
)


preluts = PreLUTs.from_pre_files(tfp + "preLUTs_Zeta0=0.00E+00_2_5/", zeta0=0, all_vars=False)
preluts.to_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
