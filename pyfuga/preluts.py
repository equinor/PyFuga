from pathlib import Path

from tqdm import tqdm
import xarray as xr
import numpy as np
from pyfuga.file_readers import read_prelut_list, read_pre_file
from pyfuga.utils import ComplexXRDataset


class PreLUT(ComplexXRDataset):
    __slots__ = []

    @staticmethod
    def from_pre_file(filename, zeta0, kz0=None, beta=None, kzmax=None, ds=None, accgoal=None):
        filename = Path(filename)
        if None in [kz0, beta, kzmax, ds, accgoal]:
            ds, smaxx, kz0, beta, kzmax, accgoal = read_prelut_list(filename.parent)[filename.name]

        pre_file = read_pre_file(filename)
        return PreLUT({**pre_file, 'beta': beta, 'kz0': kz0},
                      attrs={'ds': ds, 'kzmax': kzmax, 'zeta0': zeta0, 'accgoal': accgoal})

    @staticmethod
    def make_prelut(zeta0, kz0, beta, kzmax, ds, accgoal):
        from pyfuga.preluts_generator import PreLUTGenerator
        return PreLUTGenerator(zeta0, kz0, beta, kzmax, ds, accgoal).make_prelut()


class PreLUTs(ComplexXRDataset):
    __slots__ = []

    @staticmethod
    def from_pre_files(folder, zeta0, all_vars=True, verbose=True):

        folder = Path(folder)

        fn, ds, smaxx, kz0, beta, kzmax, accgoal = list(zip(*read_prelut_list(folder, dict=False)))
        assert all(ds[0] == np.array(ds))
        assert all(kzmax[0] == np.array(kzmax))
        dat_lst = [read_pre_file(folder / f) for f in tqdm(fn, disable=(not verbose))]

        # ds, smaxx, kz0, beta, kzmax, accgoal
        nkz0 = len(np.unique(kz0))
        kz0_lst = np.array(kz0)[:nkz0]
        beta_lst = np.array(beta)[::nkz0]
        f = dat_lst[0]  # first
        n_node = np.array([len(d['level'][1]) for d in dat_lst])
        max_nodes = n_node.max()

        def get_var(k):
            dim = ['beta', 'kz0'] + f[k][0]
            shape = f[k][1].shape
            data = np.reshape([np.r_[d[k][1], np.full((max_nodes - d[k][1].shape[0],) + shape[1:], np.nan)] for d in dat_lst],
                              (len(beta_lst), nkz0, max_nodes) + shape[1:])
            return (dim, data)
        if all_vars:
            var_name_lst = f.keys()
        else:
            var_name_lst = ['Yleft', 'Rleft', 'Rright', 'dyxu0', 'dyxu1', 'dyxv0', 'dyxv1', 'level']
        dim_dat_dict = {k: get_var(k) for k in var_name_lst}
        return PreLUTs({**dim_dat_dict},
                       coords={'kz0': kz0_lst, 'beta': beta_lst, 'i': range(max_nodes)},
                       attrs={'ds': ds[0], 'kzmax': kzmax[0], 'zeta0': zeta0, 'accgoal': accgoal})

    @staticmethod
    def make_preluts(zeta0, kz0_lst, beta_lst, kzmax, ds, accgoal, verbose=True):
        kz0_beta_lst = [(kz0, beta) for kz0 in kz0_lst for beta in beta_lst]
        ds_lst = [PreLUT.make_prelut(zeta0, kz0, beta, kzmax, ds, accgoal)
                  for kz0, beta in tqdm(kz0_beta_lst, disable=(not verbose))]
        preluts = xr.merge([ds.assign_coords(i=ds.i, kz0=ds.kz0, beta=ds.beta).expand_dims(('kz0', 'beta'))
                            for ds in ds_lst])
        f = ds_lst[0]  # first
        for k in ['ds', 'kzmax', 'zeta0']:
            assert all(f.attrs[k] == np.array([ds.attrs[k] for ds in ds_lst]))
        return PreLUTs(preluts, attrs={'ds': f.ds, 'kzmax': f.kzmax, 'zeta0': f.zeta0, 'accgoal': f.accgoal})
