import numpy as np
from numpy import newaxis as na, linalg
import xarray as xr
from tqdm import tqdm
from pyfuga.constants import zminlevel, kappa, UVW_LT
from pyfuga.utils import ComplexXRDataset, jit
import multiprocessing
import warnings


class FourierLUTGenerator():
    def __init__(self, preluts, zhub, diameter, zi, verbose=True):
        self.preluts = preluts
        self.zhub = zhub
        self.radius = diameter / 2
        self.zi = zi  # Domain height
        self.verbose = verbose

    def make_rotor_luts(self, z0, luts=UVW_LT, n_cpu=1):
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log((self.zhub - self.radius) / z0) / ds))
        high_level_out = int(np.ceil(np.log((self.zhub + self.radius) / z0) / ds))
        return self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)

    def make_hubheight_luts(self, z0, luts=UVW_LT, n_cpu=1):
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log((self.zhub) / z0) / ds))
        high_level_out = int(np.ceil(np.log((self.zhub) / z0) / ds))

        luts = self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)
        if low_level_out == high_level_out:
            lut_vars = luts.isel(level=0)
        else:
            a = np.log(self.zhub / z0) / ds - low_level_out
            lut_vars = luts.isel(level=0) * (1 - a) + luts.isel(level=1) * (a)
        lut_vars['level'] = 9999

        return ComplexXRDataset(data_vars={'z': (('level'), [self.zhub]),
                                           'k': luts.k,
                                           'diameter': luts.diameter,
                                           'hubheight': self.zhub,
                                           'z0': z0,
                                           # fourier luts, UL, UT, ...
                                           ** {k: lut_vars[k].expand_dims('level', 2)
                                               for k in lut_vars if lut_vars[k].dims == ('beta', 'kz0')}},
                                coords={'beta': luts.beta, 'kz0': luts.kz0, 'level': [9999]},
                                attrs=luts.attrs)

    def make_lut(self, z0, low_level_out, high_level_out,
                 luts=UVW_LT, n_cpu=1):
        assert all([lut in UVW_LT for lut in luts])
        zh = self.zhub
        R = self.radius
        ds, kzmax, zeta0 = [self.preluts.attrs[k] for k in ['ds', 'kzmax', 'zeta0']]

        kz0_lst = np.sort(np.unique(self.preluts.kz0.values))
        if len(kz0_lst) > 1:
            nkz0 = int(np.round(1 / np.diff(np.log10(kz0_lst))[0]))
        else:
            nkz0 = 1
        aux = z0 * np.pi * 8.0 / R
        kz0limit = 10.0**(np.ceil(np.log10(aux) * nkz0) / nkz0)

        kz0_lst = kz0_lst[kz0_lst <= kz0limit]

        upperjf = int(np.floor(np.log((zh + R) / z0) / ds))  # ceiling->floor
        lowerjf = int(np.ceil(np.log((zh - R) / z0) / ds))  # floor->ceiling
        minlevel = int(np.floor(np.log(np.maximum(zminlevel / z0, 1)) / ds))
        maxlevel = int(np.ceil(np.log(self.zi / z0) / ds))

        if low_level_out < minlevel + 1:  # pragma: no cover
            print(f'LoLevelOut ({low_level_out}) raised to MinLevel ({minlevel + 1}).')
            low_level_out = minlevel + 1

        if high_level_out > maxlevel - 1:  # pragma: no cover
            print(f'HiLevelOut ({high_level_out}) lowered to MaxLevel ({maxlevel - 1}).')
            high_level_out = maxlevel - 1

        logZiZ0 = np.log(self.zi / z0)
        beta_lst = np.sort(np.unique(self.preluts.beta.values))
        ktab = kz0_lst / z0

        # assert smaxx>self.preluts.ds * upperjf # TODO: smaxx not available here

        smax = np.minimum(np.log((16 * np.pi * (1 + zh / R) + 120.0) / kz0_lst), logZiZ0)  # neutral
        if np.abs(zeta0) > 0:
            # Stable and unstable
            smax = np.minimum(smax, np.log(np.abs(15 / zeta0)))

        if n_cpu == 1:
            map_func = map
        else:
            map_func = multiprocessing.Pool(n_cpu).imap

        beta_kz0_lst = [(beta, kz0) for beta in beta_lst for kz0 in kz0_lst]
        if any([lut[1] == 'L' for lut in luts]):
            args_lst = ((self.preluts.sel(beta=beta, kz0=kz0), beta, kz0, z0, self.zhub, self.radius, 'L',
                         lowerjf, upperjf, minlevel, maxlevel, low_level_out, high_level_out)
                        for beta, kz0 in beta_kz0_lst)

            xL = list(tqdm(map_func(solve_layers, args_lst), total=len(beta_kz0_lst), disable=(not self.verbose),
                           desc='Fourier LUTS: Solving for longitudinal forcing'))
        if any([lut[1] == 'T' for lut in luts]):
            args_lst = ((self.preluts.sel(beta=beta, kz0=kz0), beta, kz0, z0, self.zhub, self.radius, 'T',
                         lowerjf, upperjf, minlevel, maxlevel, low_level_out, high_level_out)
                        for beta, kz0 in beta_kz0_lst)

            xT = list(tqdm(map_func(solve_layers, args_lst), total=len(beta_kz0_lst), disable=(not self.verbose),
                           desc='Fourier LUTS: Solving for transversal forcing'))

        def get_var(s):
            if s[1] == 'L':
                x = xL
            else:
                x = xT
            i = {'U': 0, 'V': 2, 'W': 4, 'P': 5}[s[0]]
            return np.reshape(x, (len(beta_lst), len(kz0_lst)) + np.shape(x)[1:])[..., i]

        level = np.arange(low_level_out, high_level_out + 1)
        z = z0 * np.exp(ds * level)
        return ComplexXRDataset(data_vars={'z': (('level'), z), 'k': (('kz0'), ktab),
                                           'diameter': R * 2, 'hubheight': zh, 'z0': z0,

                                           ** {k: (('beta', 'kz0', 'level'), get_var(k))
                                               for k in luts}}, coords={'beta': beta_lst, 'kz0': kz0_lst, 'level': level},
                                attrs={'zi': self.zi,
                                       **self.preluts.attrs})


def solve_layers(args):
    prelut, beta, kz0, z0, zhub, radius, forcing, lowerjf, upperjf, minlevel, maxlevel, low_level_out, high_level_out = args
    assert forcing in 'LT'
    forcing = forcing.replace('L', 'u').replace('T', 'v')
    ds = prelut.ds
    jf_l = np.arange(lowerjf + 1, upperjf)

    z = z0 * np.exp(ds * (jf_l + np.array([-1, 0, 1])[:, na]))  # height blow, at and above current layer
    k = kz0 / z0

    max_table_level = prelut.level.max().item()

    if max_table_level < minlevel:
        return np.zeros((high_level_out - low_level_out + 1, 6), dtype=np.complex128)
    imin, imax = np.searchsorted(prelut.level, [minlevel, min(maxlevel, max_table_level)])
    prelut = prelut.sel(i=slice(imin, imax))

    zero_pad_levels = int(max(0, maxlevel - max_table_level))

    ijf0_l, ijf1_l, ijf2_l = np.searchsorted(prelut.level, [jf_l - 1, jf_l, jf_l + 1])

    YL = prelut.Yleft.values
    Rright = prelut.Rright.values
    Rleft = prelut.Rleft.values
    dYx_0 = prelut[f'dyx{forcing}0'].values
    dYx_1 = prelut[f'dyx{forcing}1'].values
    levels = prelut.level.values.astype(int)

    fac0_1_l = z[0] / (kappa * k * (z[1] - z[0]))
    fac1_1_l = 1 / (kappa * (z[1] - z[0]) * k**2)
    fac0_2_l = z[2] / (kappa * k * (z[1] - z[2]))
    fac1_2_l = 1 / (kappa * (z[1] - z[2]) * k**2)
    output = [np.concatenate([solve_layer(Rright, Rleft, YL, levels, dYx_0, dYx_1,
                                          fac0_1, fac1_1, fac0_2, fac1_2,
                                          ijf0, ijf1, ijf2),
                              np.zeros((zero_pad_levels, 6))])
              for ijf0, ijf1, ijf2, fac0_1, fac1_1, fac0_2, fac1_2 in zip(ijf0_l, ijf1_l, ijf2_l,
                                                                          fac0_1_l, fac1_1_l, fac0_2_l, fac1_2_l)]

    zf = z0 * np.exp(ds * np.arange(lowerjf, upperjf + 1))
    layer_halfwidth = np.sqrt(radius**2 - (zf - zhub)**2)
    ky = k * np.sin(beta)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
        warnings.filterwarnings('ignore', r'invalid value encountered in divide')
        fac = np.where(ky == 0, layer_halfwidth, np.sin(ky * layer_halfwidth) / ky)

    area_err_fac = radius**2 * np.pi / \
        np.sum(np.sqrt(radius**2 - (zf - zhub)**2) * zf * (np.exp(ds) - np.exp(-ds)))
    s = slice(low_level_out - minlevel - 1, high_level_out - minlevel)
    output = np.array(output)[:, s]  # dim: (from_level, to_level, uu'vv'wp)
    return np.sum(fac[1:-1][:, na, na] * output, 0) * area_err_fac


# c2 = 'complex128[:,:],'
# c3 = 'complex128[:,:,:],'
# d = 'double,'
# i = 'int32,'
# i1 = 'int32[:],'


@jit  # (f'complex128[:,:]({c3}{c3}{c3}{i1}{c2}{c2}{d}{d}{d}{d}{i}{i}{i})')
def solve_layer(Rright, Rleft, YL, levels, dYx_0, dYx_1,
                fac0_1, fac1_1, fac0_2, fac1_2,
                icl_m1, icl, icl_p1, ):  # pragma: no cover
    """
    maxlevel (z_inversion or table_max_level)
    cl+1
    cl (current level)
    cl-1
    minlevel (1m)

    'i'-prefix, e.g. icl is the node number of cl (current layer)

    Bottom to top
    -------------
    We need to calculate for each layer, cl=lowerjf+1.. upperjf-1
    YxL[j, :3] =
    minlevel..cl-1: 0
    cl-1..cl: RR @ YxL[j-1] - dYxL0 * fac01 + dYxL1 * fac11
    cl..cl+1: RR @ YxL[j-1] - dYxL0 * fac02 + dYxL1 * fac12
    cl+1..maxlevel: RR @ YxL[j-1]

    top to bottom
    -------------
    We need to calculate for each layer, cl=upperjf-1.. lowerjf+1
    YxL[j, 33] =
    maxlevel..cl+1: RR @ YxL[j-1]
    cl..cl+1: RR @ YxL[j-1] - dYxL0 * fac02 + dYxL1 * fac12
    cl-1..cl: RR @ YxL[j-1] - dYxL0 * fac01 + dYxL1 * fac11
    minlevel..cl-1: 0
    """
    icl_m1, icl, icl_p1 = icl_m1.item(), icl.item(), icl_p1.item()
    Yx_3 = [np.zeros(3, dtype=np.complex128)] * (icl_m1 + 1)  # minlevel(z=1m) to cl-1

    Ux_step_lst = np.concatenate(((-dYx_0[icl_m1:icl, :3] * fac0_1 + dYx_1[icl_m1:icl, :3] * fac1_1),  # cl-1 to cl
                                  (-dYx_0[icl:icl_p1, :3] * fac0_2 + dYx_1[icl:icl_p1, :3] * fac1_2)))  # cl to cl+1

    for Ux_step, RR in zip(Ux_step_lst, Rright[icl_m1:icl_p1, :3, :3]):
        Yx_3.append(np.dot(np.ascontiguousarray(RR.T), Yx_3[-1] + Ux_step))

    # cl+1 to max level
    for RR in Rright[icl_p1:-1, :3, :3]:
        Yx_3.append(np.dot(np.ascontiguousarray(RR.T), Yx_3[-1]))

    M = np.concatenate((np.conj(YL[-1, :3]),
                        np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=np.complex128)))
    b = np.concatenate((Yx_3[-1][:3], np.asarray([0 + 0j, 0, 0])))
    x_ = linalg.solve(M, b)
    Yct = np.conj(YL[-1, 3:])
    Yx_6 = [np.concatenate((Yx_3.pop(), Yct @ x_))]  # YxL row 1-6 in reverse order

    # max level to cl+1
    for RL in Rleft[-1:icl_p1:-1, :, 3:]:
        Yx_6.append(np.concatenate((Yx_3.pop(), np.dot(np.ascontiguousarray(RL.T), Yx_6[-1]))))

    icl_m1, icl, icl_p1 = icl_m1 - 1, icl - 1, icl_p1 - 1
    Ux_step_lst = np.concatenate(((+dYx_0[icl_p1:icl:-1, 3:] * fac0_2 - dYx_1[icl_p1:icl:-1, 3:] * fac1_2),  # cl+1 to cl
                                  (+dYx_0[icl:icl_m1:-1, 3:] * fac0_1 - dYx_1[icl:icl_m1:-1, 3:] * fac1_1)))  # cl to cl-1
    for Ux_step, RL in zip(Ux_step_lst, Rleft[icl_p1 + 1:icl_m1 + 1:-1, :, 3:]):
        Yx_6.append(np.concatenate((Yx_3.pop(), np.dot(np.ascontiguousarray(RL.T), Yx_6[-1]) + Ux_step)))

    # cl-1 to min_level
    for RL in Rleft[icl_m1 + 1:0:-1, :, 3:]:
        Yx_6.append(np.concatenate((np.zeros(3, dtype=np.complex128), np.dot(np.ascontiguousarray(RL.T), Yx_6[-1]))))

    new_level = np.concatenate((np.array([True]), levels[1:-1] > levels[:-2]))

    return [(np.ascontiguousarray(YL.T) @ Yx_6)
            for YL, Yx_6 in list(zip(

                # YL[1:][new_level],
                [v for v, nl in zip(YL[1:], new_level) if nl],
                # same as np.array(Yx_6[::-1][1:])[new_level] which is not working with
                # numba jit
                [v for v, nl in zip(Yx_6[::-1][1:], new_level) if nl]))]
