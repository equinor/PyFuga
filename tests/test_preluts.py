import importlib
import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from pyfuga import utils
from pyfuga.constants import UVW_LT
from pyfuga.file_readers import read_lut_file
from pyfuga.flut import FourierLUTGenerator
from pyfuga.preluts import PreLUT, PreLUTs
from pyfuga.preluts_generator import PreLUTGenerator, PrelutNode
from pyfuga.utils import compile, get_beta, get_beta_lst, get_kz0_lst

from .helpers import expose_old_names
from .test_files import tfp


def setup_module(module):
    """setup any state specific to the execution of the given module."""
    compile(jit=False)


def test_get_beta_lst():
    ref = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    assert_allclose(get_beta_lst(nbeta=5), ref.beta)


def test_get_kz0_lst():
    ref = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_2_5.nc")
    assert_allclose(get_kz0_lst(nkz0=2), ref.kz0)


def test_load_prelut_file():

    prelut = PreLUT.from_pre_file(
        tfp + "preLUTs_Zeta0=0.00E+00_2_5/0.0000-09.0000.pre", zeta0=0, beta=0, kz0=0, kzmax=0, ds=0.05
    )

    assert_array_almost_equal(prelut.Yleft[7][0, 3], -0.330350424728106 + 9.114133411353441e-012j, 10)
    assert_array_almost_equal(prelut.Rleft[7][0, 3], -4.045634236057177e-002 - 1.325470800079699e-010j, 10)
    assert_array_almost_equal(prelut.Rright[7][0, 3], 3.768307078547423e-002 + 1.581796253544902e-010, 9)
    assert_array_almost_equal(prelut.dyxu0[7][1], -2.423313312692920e-011 - 1.267852505176752e-021j, 10)
    assert_array_almost_equal(prelut.dyxu1[7][1], -3.528960438325056e-020 - 1.849660676125115e-030j, 10)
    assert_array_almost_equal(prelut.dyxv0[7][3], 6.806655201915466e-011 - 7.883109477872014e-021, 10)
    assert_array_almost_equal(prelut.dyxv1[7][3], 9.905991452339835e-020 - 1.147145744025741e-029, 10)
    assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    assert_array_almost_equal(prelut.dyxw1[7][2], 2.912345039036901e-039 - 3.093036809788905e-030, 10)
    assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    assert_array_almost_equal(prelut.sleft[7], 0.35, 10)
    assert_array_almost_equal(prelut.sright[7], 0.4, 10)
    assert_array_almost_equal(prelut.level[7], 7, 10)


def test_load_prelut_file_via_read_prelut_list():

    prelut = PreLUT.from_pre_file(tfp + "preLUTs_Zeta0=0.00E+00_2_5/0.3333-07.0000.pre", zeta0=0)
    assert prelut.zeta0 == 0
    assert prelut.beta.item() == 0.829727913835271
    assert prelut.kz0 == 1e-7
    assert prelut.kzmax == 300
    assert prelut.ds == 0.05


def compare(res, ref, atol=1e-13, rtol=1e-9):
    assert_array_equal(res.level, ref.level)
    for k in ref:
        # print(k)
        max_dims = ("j", "k")[: len(ref[k].shape) - 1]
        max_idx = (0, 1, 2)[1 : len(ref[k].shape)]
        try:
            assert_allclose(res[k], ref[k], atol=atol, rtol=rtol, err_msg=k)
        except AssertionError:
            err = ref[k] - res[k]

            rerr_real = np.where(np.abs(ref[k].real) > 1e-15, err.real / ref[k].real, np.nan)
            rerr_imag = np.where(np.abs(ref[k].imag) > 1e-15, err.imag / ref[k].imag, np.nan)
            ax1, ax2 = plt.subplots(2, 1)[1]
            plt.title(k)
            ax1.plot(np.abs(err.real).max(max_dims).values, label="Real, abs")
            ax2.plot(np.nanmax(np.abs(rerr_real), max_idx), label="Real, rel")
            # ax1.set_xlim([360, 375])

            ax1.plot(np.abs(err.imag).max(max_dims).values, label="Imag, abs")
            ax2.plot(np.nanmax(np.abs(rerr_imag), max_idx), label="Imag, rel")
            # ax1.set_xlim([360, 375])

            ax1.legend()
            ax2.legend()
            plt.ylabel = "Error"
            plt.xlabel = "Node"
            plt.show()
            raise


def test_compact():
    preluts = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=get_beta_lst(1),
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
        compact=False,
    )
    preluts.save(tfp + "tmp_preluts.nc")
    preluts = PreLUTs.from_netcdf(tfp + "tmp_preluts.nc")

    preluts_compact = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=get_beta_lst(1),
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
    )
    preluts_compact.save(tfp + "tmp_preluts_compact.nc")
    preluts_compact = PreLUTs.from_netcdf(tfp + "tmp_preluts_compact.nc")

    for beta in preluts.beta.values:
        for kz0 in preluts.kz0.values:
            p_c = preluts_compact.sel(beta=beta, kz0=kz0)
            p = preluts.sel(beta=beta, kz0=kz0)
            for k, v in p.items():
                assert_array_equal(v, p_c[k])


def test_prelut_neutral_all_vars():
    preluts = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-9, 1e-8],
        beta_lst=get_beta_lst(1),
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
    )

    # QUICK FIX: expose old variable names for compatibility with reference files
    preluts = expose_old_names(preluts)

    ref_prelut = PreLUT.from_pre_file(tfp + "preLUTs_Zeta0=0.00E+00_2_5/0.0000-09.0000.pre", zeta0=0)

    assert ref_prelut.ds == preluts.ds
    assert ref_prelut.kzmax == preluts.kzmax
    # assert ref_prelut.accgoal == preluts.accgoal

    # compare(res, prelut)

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_lut(
        z0=0.00001, low_level_out=315, high_level_out=315
    )

    for var in UVW_LT:
        ref = read_lut_file(
            tfp + f"D080.0000_zH070.0000_2_5/Z0=0.00001000Zi=00400Zeta0=0.00E+00/{var}0315.lut",
            prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_2_5",
        )

        assert_allclose(
            ref.sel(kz0=flut.kz0, beta=flut.beta, method="nearest")[var].real, flut[var].real, rtol=1e-5, atol=1e-6
        )
        assert_allclose(
            ref.sel(kz0=flut.kz0, beta=flut.beta, method="nearest")[var].imag, flut[var].imag, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("zeta0", [-1, 1])
def test_prelut_stable_and_unstable(zeta0):
    preluts = PreLUTs.make_preluts(
        zeta0=zeta0, kz0_lst=[1e-9], beta_lst=[0], kzmax=300, ds=0.05, accgoal=0.0001, jit=False, verbose=False
    )

    # QUICK FIX: expose old variable names for compatibility with reference files
    preluts = expose_old_names(preluts)

    ref_prelut = PreLUT.from_pre_file(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre", zeta0=zeta0)

    assert ref_prelut.ds == preluts.ds
    assert ref_prelut.kzmax == preluts.kzmax
    assert ref_prelut.accgoal == preluts.accgoal

    # compare(res, prelut)

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=0.00001, luts=["UL"]
    )

    ref = read_lut_file(
        tfp + f"D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0={zeta0}.00E+00/UL9999.lut",
        prelut_folder=tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2",
    )

    assert_allclose(ref.sel(kz0=1e-9, beta=0).UL, flut.UL.item(), rtol=2e-5, atol=1e-10)


def test_prelut_stable():

    utils.preludium_equivalent = True
    module = inspect.getmodule(PreLUTGenerator)
    if module is None:
        raise RuntimeError("Could not determine the module for PreLUTGenerator.")
    importlib.reload(module)
    zeta0 = 3.85e-7
    kzmax = 1
    prelut_generator = PreLUTGenerator(
        zeta0=zeta0, kz0=1e-6, beta=1.469367938527859e-039, kzmax=kzmax, ds=0.05, accgoal=0.0001
    )
    prelut = prelut_generator.make_prelut()

    # QUICK FIX: expose old variable names for compatibility with reference files
    prelut = expose_old_names(prelut)

    utils.preludium_equivalent = False
    module = inspect.getmodule(PreLUTGenerator)
    if module is None:
        raise RuntimeError("Could not determine the module for PreLUTGenerator.")
    importlib.reload(module)

    ref_prelut = PreLUT.from_pre_file(tfp + "preLUTs_Zeta0=3.85E-07_1_2/0.0000-06.0000.pre", zeta0=zeta0)
    compare(prelut, ref_prelut)


def test_next_node():
    node = PrelutNode()

    prelut = PreLUT.from_pre_file(tfp + "preLUTs_Zeta0=0.00E+00_1_2/0.0000-09.0000.pre", zeta0=0)
    i = 1
    node.Yright = prelut.Yleft[i].values @ np.conj(prelut.Rleft[i].T).values
    next_node = node.get_next(0.05, 0.1)
    assert next_node.sleft == 0.05
    assert next_node.sright == 0.1

    assert next_node.Rleft is not None
    # print(np.abs(next_node.Yleft @ np.conj(next_node.Rleft.T) - node.Yright).max())
    assert_allclose(next_node.Yleft @ np.conj(next_node.Rleft.T), node.Yright, rtol=1e-6, atol=1e-15)
    # print(np.abs(node.Rright @ next_node.Rleft - np.eye(6)).max())
    assert_allclose(node.Rright @ next_node.Rleft, np.eye(6), atol=1e-16)


def test_prelut_with_above_sm():
    # prelut = PreLUT.from_pre_file(tfp + 'preLUTs_Zeta0=0.00E+00_2_5/0.0000-07.0000.pre',
    #                               zeta0=0, beta=0, kz0=1e-7, kzmax=0, ds=0.05)
    preluts = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-7],
        beta_lst=get_beta(np.array([0]))[:1],
        kzmax=300,
        ds=0.05,
        accgoal=0.00001,
        jit=False,
        verbose=False,
    )

    # QUICK FIX: expose old variable names for compatibility with reference files
    preluts = expose_old_names(preluts)

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=0.00001, luts=["UL"]
    )
    ref = read_lut_file(
        tfp + "D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL9999.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_1_2",
    )

    assert_allclose(flut.UL.item(), ref.sel(kz0=1e-7, beta=0).UL)


def test_prelut_with_substations():
    preluts = PreLUTs.make_preluts(
        zeta0=0,
        kz0_lst=[1e-6],
        beta_lst=get_beta(np.array([0]))[:1],
        kzmax=300,
        ds=0.05,
        accgoal=0.0001,
        jit=False,
        verbose=False,
    )

    # QUICK FIX: expose old variable names for compatibility with reference files
    preluts = expose_old_names(preluts)

    flut = FourierLUTGenerator(preluts, zhub=70, diameter=80, zi=400, verbose=False).make_hubheight_luts(
        z0=0.00001, luts=["UL"]
    )
    ref = read_lut_file(
        tfp + "D080.0000_zH070.0000_1_2_9999/Z0=0.00001000Zi=00400Zeta0=0.00E+00/UL9999.lut",
        prelut_folder=tfp + "preLUTs_Zeta0=0.00E+00_1_2",
    )

    assert_array_almost_equal(ref.sel(kz0=1e-6, beta=0).UL, flut.UL.item())


def test_prelut_save_load():

    if os.path.isfile("tmp.nc"):
        os.remove("tmp.nc")
    ref = PreLUT.make_prelut(zeta0=0, kz0=1e-9, beta=get_beta(np.array([0]))[0], kzmax=300, ds=0.05, accgoal=0.0001)
    ref.save("tmp.nc")

    res = PreLUT.from_netcdf("tmp.nc")
    compare(res, ref)


def test_preluts():
    preluts = PreLUTs.from_pre_files(tfp + "preLUTs_Zeta0=0.00E+00_1_2/", zeta0=0, verbose=False)
    ref = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    ref.equals(preluts)
    prelut = preluts.isel(beta=1, kz0=1, i=7)

    assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    assert_array_equal(prelut.sleft, 0.35)
    assert_array_almost_equal(prelut.sright, 0.4, 15)
    assert_array_equal(prelut.level, 7)


def test_preluts_from_pre_files():
    zeta0 = -1
    preluts = PreLUTs.from_pre_files(
        tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2/", zeta0=zeta0, all_vars=False, verbose=False
    )
    preluts_nc = PreLUTs.from_netcdf(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2.nc")
    assert preluts_nc.drop_vars(["sleft", "sright", "dyxw0", "dyxw1"]).equals(preluts)
    ref = PreLUT.from_pre_file(tfp + f"preLUTs_Zeta0={zeta0}.00E+00_1_2/0.0000-09.0000.pre", zeta0)
    prelut = preluts.isel(beta=0, kz0=0)
    for k in prelut:
        v = prelut[k]
        if "i" in v.dims:
            v = v[: ref[k].shape[0]]
        assert_array_equal(v, ref[k])


def test_make_preluts():
    preluts_ref = PreLUTs.from_netcdf(tfp + "preLUTs_Zeta0=0.00E+00_1_2.nc")
    kz0_lst = [1e-9, 1e-8]
    beta_lst = preluts_ref.beta[:2].values
    preluts = PreLUTs.make_preluts(
        zeta0=0, kz0_lst=kz0_lst, beta_lst=beta_lst, kzmax=0.0000001, ds=0.05, accgoal=0.0001, jit=False, verbose=False
    )

    # QUICK FIX: expose old variable names for compatibility with reference files
    preluts = expose_old_names(preluts)

    prelut = preluts.sel(beta=beta_lst[1], kz0=kz0_lst[1], i=7)

    assert_array_almost_equal(prelut.Yleft[0, 3], -3.818665046221538e-002 - 4.092601925385363e-011j, 10)
    assert_array_almost_equal(prelut.Rleft[1, 4], -4.185413090968856e-002 - 1.707412114366788e-010j, 10)
    # not equal due to new ortogonalization
    # assert_array_almost_equal(prelut.Rright[0, 3], 0.133639331140749 + 1.683220576275511e-010j, 10)
    assert_array_almost_equal(prelut.dyxu0[1], -2.801213847640760e-011 - 1.811265791216862e-019j, 10)
    assert_array_almost_equal(prelut.dyxu1[1], -4.079279718323910e-019 - 2.637711084697475e-027j, 10)
    assert_array_equal(prelut.sleft, 0.35)
    assert_array_almost_equal(prelut.sright, 0.4, 15)
    assert_array_equal(prelut.level, 7)
