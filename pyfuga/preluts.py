import multiprocessing as mp
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from pyfuga.file_readers import read_pre_file, read_prelut_list
from pyfuga.utils import ComplexXRDataset, read_complex


class PreLUT(ComplexXRDataset):
    __slots__ = []

    @staticmethod
    def from_pre_file(filename, zeta0, kz0=None, beta=None, kzmax=None, ds=None, accgoal=None):
        filename = Path(filename)
        if None in [kz0, beta, kzmax, ds, accgoal]:
            ds, smaxx, kz0, beta, kzmax, accgoal = read_prelut_list(filename.parent)[filename.name]

        pre_file = read_pre_file(filename)
        return PreLUT(
            {**pre_file, "beta": beta, "kz0": kz0}, attrs={"ds": ds, "kzmax": kzmax, "zeta0": zeta0, "accgoal": accgoal}
        )

    @staticmethod
    def make_prelut(zeta0, kz0, beta, kzmax, ds, accgoal):
        from pyfuga.preluts_generator import PreLUTGenerator

        return PreLUTGenerator(zeta0, kz0, beta, kzmax, ds, accgoal).make_prelut()

    @staticmethod
    def make_prelut_args(args):
        from pyfuga import utils

        jit = args[-1]
        args = args[:-1]
        utils.compile(jit)
        return PreLUT.make_prelut(*args)


class PreLUTs(ComplexXRDataset):
    __slots__ = []

    @staticmethod
    def from_pre_files(folder, zeta0, all_vars=True, verbose=True):

        folder = Path(folder)

        fn, ds, smaxx, kz0, beta, kzmax, accgoal = list(zip(*read_prelut_list(folder, dict=False), strict=True))
        assert all(ds[0] == np.array(ds))
        assert all(kzmax[0] == np.array(kzmax))
        dat_lst = [read_pre_file(folder / f) for f in tqdm(fn, disable=(not verbose))]

        # ds, smaxx, kz0, beta, kzmax, accgoal
        nkz0 = len(np.unique(kz0))
        kz0_lst = np.array(kz0)[:nkz0]
        beta_lst = np.array(beta)[::nkz0]
        f = dat_lst[0]  # first
        n_node = np.array([len(d["level"][1]) for d in dat_lst])
        max_nodes = n_node.max()

        def get_var(k):
            dim = ["beta", "kz0"] + f[k][0]
            shape = f[k][1].shape
            data = np.reshape(
                [np.r_[d[k][1], np.full((max_nodes - d[k][1].shape[0],) + shape[1:], np.nan)] for d in dat_lst],
                (len(beta_lst), nkz0, max_nodes) + shape[1:],
            )
            return (dim, data)

        if all_vars:
            var_name_lst = f.keys()
        else:
            var_name_lst = ["Yleft", "Rleft", "Rright", "dyxu0", "dyxu1", "dyxv0", "dyxv1", "level"]
        dim_dat_dict = {k: get_var(k) for k in var_name_lst}
        return PreLUTs(
            {**dim_dat_dict},
            coords={"kz0": kz0_lst, "beta": beta_lst, "i": range(max_nodes)},
            attrs={"ds": ds[0], "kzmax": kzmax[0], "zeta0": zeta0, "accgoal": accgoal},
        )

    @staticmethod
    def make_preluts(zeta0, kz0_lst, beta_lst, kzmax, ds, accgoal, jit=True, n_cpu=1, verbose=True, compact=True):

        args_lst = [(zeta0, kz0, beta, kzmax, ds, accgoal, jit) for beta in beta_lst for kz0 in kz0_lst]

        if n_cpu in (None, 1):
            iterator = map(PreLUT.make_prelut_args, args_lst)
            ds_lst = list(
                tqdm(
                    iterator,
                    total=len(args_lst),
                    disable=(not verbose),
                    desc="Generating preluts",
                )
            )
        else:
            ctx = mp.get_context("spawn")  # or "forkserver"
            with ctx.Pool(processes=n_cpu) as pool:
                iterator = pool.imap(PreLUT.make_prelut_args, args_lst, chunksize=1)
                ds_lst = list(
                    tqdm(
                        iterator,
                        total=len(args_lst),
                        disable=(not verbose),
                        desc="Generating preluts",
                    )
                )

        f = ds_lst[0]  # first
        for k in ["ds", "kzmax", "zeta0"]:
            assert all(f.attrs[k] == np.array([ds.attrs[k] for ds in ds_lst]))

        if compact:

            data_dict = {}

            def append(ds):
                for k, v in ds.items():
                    if k not in data_dict:
                        data_dict[k] = []
                    if len(v.dims) and v.dims[0] == "i":
                        data_dict[k].append(v.values)
                    else:
                        data_dict[k].append(np.full(len(ds.i), v))

            while len(ds_lst):
                append(ds_lst.pop())

            for k, v in data_dict.items():
                data_dict[k] = (("kz0_beta_i",) + f[k].dims[1:], np.concatenate(v, 0))

            return CompactPreLUTs(
                data_dict,
                attrs={
                    "ds": f.ds,
                    "kzmax": f.kzmax,
                    "zeta0": f.zeta0,
                    "accgoal": f.accgoal,
                    "kz0_lst": kz0_lst,
                    "beta_lst": beta_lst,
                },
            )
        else:
            preluts = xr.combine_by_coords(
                [ds.assign_coords(i=ds.i, kz0=ds.kz0, beta=ds.beta).expand_dims(("kz0", "beta")) for ds in ds_lst]
            )

            return PreLUTs(
                preluts.data_vars,
                coords=preluts.coords,
                attrs={
                    "ds": f.ds,
                    "kzmax": f.kzmax,
                    "zeta0": f.zeta0,
                    "accgoal": f.accgoal,
                },
            )

    @classmethod
    def from_netcdf(cls, filename):
        ds = read_complex(filename)
        if "kz0_beta_i" in ds.dims:
            return CompactPreLUTs(ds, attrs=ds.attrs)
        else:
            return PreLUTs(ds, attrs=ds.attrs)


class CompactPreLUTs(PreLUTs):
    __slots__ = []

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=False,
        kz0=None,
        beta=None,
        **indexers_kwargs,
    ):
        m = (self.beta.values == beta) & (self.kz0.values == kz0)
        ds = PreLUTs(
            {k: (("i",) + v.dims[1:], v.values[m]) for k, v in self.items()},
            attrs={k: v for k, v in self.attrs.items() if k not in ["kz0_lst", "beta_lst"]},
        )
        ds = ds.assign_coords(i=ds.i, beta=beta, kz0=kz0)
        return PreLUTs.sel(ds, indexers=indexers, method=method, tolerance=tolerance, drop=drop, **indexers_kwargs)
