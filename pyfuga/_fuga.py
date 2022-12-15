import numpy as np
import xarray as xr

from pyfuga.flut import FourierLUTGenerator
from pyfuga.preluts import PreLUTs
from pyfuga.trafalgar import Trafalgar
from pyfuga.utils import get_kz0_lst, get_beta_lst, ComplexXRDataset
from pathlib import Path
from pyfuga.constants import UVW_LT
from pyfuga import utils
import os


def get_luts(folder, zeta0, nkz0, nbeta, diameter, zhub, z0, zi, zlow, zhigh,
             lut_vars=UVW_LT, nx=2048, ny=512, dx=None, dy=None,
             jit=True, n_cpu=1):  # pragma: no cover
    """Generate and save (or load if exists) Fuga look-up tables. This function performs the full path from
    input via preluts, fourier LUTs and LUTs to the final LUTs netcdf dataset

    Parameters
    ----------
    folder : string or Path
        Path where all files (intermediate and final) are stored
    zeta0 : float
        Stability parameter
    nkz0 : int
        Number of kz0 per decade. Total number of kz0 = 9 * nkz0 - (nkz0-1)
    nbeta : int
        Number of beta angles. Total number of beta angles = nbeta + 1
    diameter : float
        Wind turbine diameter
    zhub : float
        Wind turbine hub height
    z0 : float
        Roughness length
    zi : float
        Inversion height
    zlow : float
        Lower height of output domain. If zlow=zhigh=zhub, the output will only contain one layer at the hub height
    zhigh : float
        Upper height of output domain. If zlow=zhigh=zhub, the output will only contain one layer at the hub height
    nx : int, optional
        Number of points in LUT (U direction), default is 2048
        The wind turbine is located 1/4 inside the domain
    ny : int, optional
        Number of points in LUT (V direction). Note only one half of the domain is store,
        i.e. ny is the number of points from the wind turbine to the side boundary of the domain, default is 512
    dx : int, float or None, optional
        Distance between points on the x-axis (U direction)
        If None (default), dx is set to diameter / 4
    dy : int, float or None, optional
        Distance between points on the y-axis (V direction)
        If None (default), dy is set to diameter / 16
    jit : boolean
        If True (default), some slow functions are just-in-time compiled
    n_cpu : int or None
        If >1, the preluts are generated in parallel on <n_cpu> cpus
        If None, the maximum available number of cpus are used

    Returns
    -------
    luts : xarray Dataset
    """
    utils.compile(jit)
    folder = Path(folder)
    os.makedirs(folder, exist_ok=True)
    dx = dx or diameter / 4
    dy = dy or diameter / 16

    preluts_id = f'Zeta0={zeta0:3.2f}_{nkz0}_{nbeta}'

    L_vars = [v[0] for v in lut_vars if v[1] == 'L']
    T_vars = [v[0] for v in lut_vars if v[1] == 'T']
    lut_vars_id = ""
    if L_vars:
        lut_vars_id += f"_{''.join(L_vars)}L"
    if T_vars:
        lut_vars_id += f"_{''.join(T_vars)}T"

    ds = 0.05
    if zlow == zhigh == zhub:
        low_level_out = high_level_out = 9999
        z_id = f"z{zhub:.1f}"
    else:
        low_level_out = int(np.floor(np.log(zlow / z0) / ds))
        high_level_out = int(np.ceil(np.log(zhigh / z0) / ds))
        zlow = z0 * np.exp(low_level_out * ds)
        zhigh = z0 * np.exp(high_level_out * ds)
        z_id = f"z{zlow:.1f}-{zhigh:.1f}"

    fluts_id = preluts_id + f'_D{diameter}_zhub{zhub}_zi{zi}_z0={z0:.8f}_{z_id}{lut_vars_id}'

    luts_id = fluts_id + f'_nx{nx}_ny{ny}_dx{dx}_dy{dy}'

    luts_path = folder / f'LUTs_{luts_id}.nc'
    if not luts_path.exists():
        # LUTs are missing (run Trafalgar)
        fluts_path = folder / f'fLUTs_{fluts_id}.nc'
        if not fluts_path.exists():
            # FourierLUTs are missing (run lut)

            preluts_path = folder / f'preLUTs_Zeta0={zeta0:3.2f}_{nkz0}_{nbeta}.nc'
            if not preluts_path.exists():
                # Preluts are missing (run prelut)
                preluts = PreLUTs.make_preluts(zeta0=0,
                                               kz0_lst=get_kz0_lst(nkz0, 1e-9, 1e-1),
                                               beta_lst=get_beta_lst(nbeta),
                                               kzmax=300, ds=ds, accgoal=0.0001, jit=jit, n_cpu=n_cpu)
                preluts.attrs['nkz0'] = nkz0
                preluts.attrs['nbeta'] = nbeta
                preluts.save(preluts_path)
            else:
                preluts = PreLUTs.from_netcdf(preluts_path)

            # preluts loaded make fourier luts
            flut_generator = FourierLUTGenerator(preluts, zhub, diameter, zi)
            if low_level_out == high_level_out == 9999:
                fluts = flut_generator.make_hubheight_luts(z0, lut_vars)
            else:
                fluts = flut_generator.make_lut(z0, low_level_out, high_level_out, lut_vars)

            fluts.save(fluts_path)
        else:
            fluts = ComplexXRDataset.from_netcdf(fluts_path)

        # fourier luts loaded make luts
        luts = Trafalgar(fluts, nx, ny, dx, dy).make_luts(3)
        luts.to_netcdf(luts_path)
    else:
        luts = xr.load_dataset(luts_path)
    luts.attrs['name'] = luts_path.stem
    return luts


def main():
    if __name__ == '__main__':  # pragma: no cover
        luts = get_luts(folder=Path(__file__).parent.parent, zeta0=0, nkz0=16, nbeta=32,
                        diameter=80, zhub=70, z0=0.00001, zi=400, zlow=70, zhigh=70,
                        lut_vars=['UL'])
        import matplotlib.pyplot as plt
        luts.UL.sel(z=70)[502:542, :40].plot(x='x')
        plt.axis('equal')
        plt.figure()
        luts = get_luts(folder=Path(__file__).parent.parent, zeta0=0, nkz0=16, nbeta=32,
                        diameter=80, zhub=70, z0=0.00001, zi=400, zlow=30, zhigh=110,
                        lut_vars=['UL'])

        luts.UL.sel(x=500)[:, :50].plot(x='y')
        plt.axis('equal')
        plt.show()


main()
