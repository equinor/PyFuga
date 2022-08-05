import numpy as np
from numpy import newaxis as na, linalg
import xarray as xr
from .constants import zminlevel
from PyPreludium.pypreludium.constants import kappa

np.set_printoptions(precision=3, linewidth=200)


class FourierLUTGenerator():
    def __init__(self, preluts, zhub, diameter, zi):
        self.preluts = preluts
        self.zhub = zhub
        self.radius = diameter / 2
        self.zi = zi  # Domain height

    def make_rotor_luts(self, z0, luts=['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']):
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log(self.zhub - self.radius) / ds))
        high_level_out = int(np.ceil(np.log(self.zhub + self.radius) / ds))
        return self.make_lut(z0, low_level_out, high_level_out, luts)

    def make_hubheight_luts(self, z0, luts=['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']):
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log(self.zhub) / ds))
        high_level_out = int(np.ceil(np.log(self.zhub) / ds))
        luts = self.make_lut(z0, low_level_out, high_level_out, luts)
        return luts

    def make_lut(self, z0, low_level_out, high_level_out,
                 luts=['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT']):
        assert all([l in ['UL', 'UT', 'VL', 'VT', 'WL', 'WT', 'PL', 'PT'] for l in luts])
        zh = self.zhub
        R = self.radius
        ds = self.preluts.ds
        kz0_lst = self.preluts.kz0.values
        nkz0 = int(np.round(1 / np.diff(np.log10(kz0_lst))[0]))
        upperjf = int(np.floor(np.log((zh + R) / z0) / ds))  # ceiling->floor
        lowerjf = int(np.ceil(np.log((zh - R) / z0) / ds))  # floor->ceiling
        minlevel = int(np.floor(np.log(np.maximum(zminlevel / z0, 1)) / ds))
        maxlevel = int(np.ceil(np.log(self.zi / z0) / ds))

        if low_level_out < minlevel:  # pragma: no cover
            print(f'LoLevelOut ({self.low_level_out}) raised to MinLevel ({minlevel}).')
            low_level_out = minlevel

        if high_level_out > maxlevel:  # pragma: no cover
            print(f'HiLevelOut ({self.high_level_out}) lowered to MaxLevel ({maxlevel}).')
            high_level_out = maxlevel

        aux = z0 * np.pi * 8.0 / R
        kz0limit = 10.0**(np.ceil(np.log10(aux) * nkz0) / nkz0)
        logZiZ0 = np.log(self.zi / z0)
        beta_lst = self.preluts.beta.values
        ktab = kz0_lst / z0
        assert all(kz0_lst <= kz0limit)
        # assert smaxx>self.preluts.ds * upperjf # TODO: smaxx not available here

        smax = np.minimum(np.log((16 * np.pi * (1 + zh / R) + 120.0) / kz0_lst), logZiZ0)  # neutral
        if np.abs(self.preluts.zeta0) > 0:
            # Stable and unstable
            smax = np.minimum(smax, np.log(np.abs(15 / self.preluts.zeta0)))

        if any([l[1] == 'L' for l in luts]):
            xL = np.array([[self.solve_layers(beta, kz0, z0, 'L', lowerjf, upperjf,
                                              minlevel, maxlevel, low_level_out, high_level_out)
                            for kz0 in kz0_lst]
                           for beta in beta_lst])
        if any([l[1] == 'T' for l in luts]):
            xT = np.array([[self.solve_layers(beta, kz0, z0, 'T', lowerjf, upperjf,
                                              minlevel, maxlevel, low_level_out, high_level_out)
                            for kz0 in kz0_lst]
                           for beta in beta_lst])

        def get_var(s):
            if s[1] == 'L':
                x = xL
            else:
                x = xT
            i = {'U': 0, 'V': 2, 'W': 4, 'P': 5}[s[0]]
            return x[..., i]
        if low_level_out == 9999:
            raise NotImplementedError()
        else:
            level = np.arange(low_level_out, high_level_out + 1)
            z = z0 * np.exp(ds * level)
        return xr.Dataset({'z': (('level'), z), **{k: (('beta', 'kz0', 'level'), get_var(k))
                                                   for k in luts}}, coords={'beta': beta_lst, 'k': ktab, 'level': level})

    def solve_layers(self, beta, kz0, z0, forcing, lowerjf, upperjf, minlevel, maxlevel, low_level_out, high_level_out):
        print(beta, kz0)
        assert forcing in 'LT'
        forcing = forcing.replace('L', 'u').replace('T', 'v')
        ds = self.preluts.ds
        jf_l = np.arange(lowerjf + 1, upperjf)

        z = z0 * np.exp(ds * (jf_l + np.array([-1, 0, 1])[:, na]))  # height blow, at and above current layer
        k = kz0 / z0
        prelut = self.preluts.sel(beta=beta, kz0=kz0)
        max_table_level = prelut.level.max().item()
        imin, imax = np.searchsorted(prelut.level, [minlevel, min(maxlevel, max_table_level)])
        zero_pad_levels = int(max(0, maxlevel - max_table_level))

        ijf0_l, ijf1_l, ijf2_l = np.searchsorted(prelut.level, [jf_l - 1, jf_l, jf_l + 1])

        YL = prelut.Yleft.values
        Rright = prelut.Rright.values
        Rleft = prelut.Rleft.values
        dYx_0 = prelut[f'dyx{forcing}0'].values
        dYx_1 = prelut[f'dyx{forcing}1'].values
        levels = prelut.level.values

        fac0_1_l = z[0] / (kappa * k * (z[1] - z[0]))
        fac1_1_l = 1 / (kappa * (z[1] - z[0]) * k**2)
        fac0_2_l = z[2] / (kappa * k * (z[1] - z[2]))
        fac1_2_l = 1 / (kappa * (z[1] - z[2]) * k**2)
        output = [np.r_[self.solve_layer(Rright, Rleft, YL, levels, dYx_0, dYx_1,
                                         fac0_1, fac1_1, fac0_2, fac1_2,
                                         imin, imax, ijf0, ijf1, ijf2),
                        np.zeros((zero_pad_levels, 6))]
                  for ijf0, ijf1, ijf2, fac0_1, fac1_1, fac0_2, fac1_2 in zip(ijf0_l, ijf1_l, ijf2_l,
                                                                              fac0_1_l, fac1_1_l, fac0_2_l, fac1_2_l)]

        zf = z0 * np.exp(ds * np.arange(lowerjf, upperjf + 1))
        layer_halfwidth = np.sqrt(self.radius**2 - (zf - self.zhub)**2)
        ky = k * np.sin(beta)
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered in divide')
            fac = np.where(ky == 0, layer_halfwidth, np.sin(ky * layer_halfwidth) / ky)

        area_err_fac = self.radius**2 * np.pi / \
            np.sum(np.sqrt(self.radius**2 - (zf - self.zhub)**2) * zf * (np.exp(ds) - np.exp(-ds)))
        s = slice(low_level_out - imin - 1, high_level_out - imin)
        output = np.array(output)[:, s]
        return np.sum(fac[1:-1][:, na, na] * output, 0) * area_err_fac

    def solve_layer(self, Rright, Rleft, YL, levels, dYx_0, dYx_1,
                    fac0_1, fac1_1, fac0_2, fac1_2,
                    imin, imax, ijf0, ijf1, ijf2, ):
        # print(ijf1)

        # Bottom to top
        # -------------
        # We need to calculate for each layer, jf=lowerjf+1.. upperjf-1
        # YxL[j, :3] =
        # minlevel..jf-1: 0
        # jf-1..jf: RR @ YxL[j-1] - dYxL0 * fac01 + dYxL1 * fac11
        # jf..jf+1: RR @ YxL[j-1] - dYxL0 * fac02 + dYxL1 * fac12
        # jf+1..maxlevel: RR @ YxL[j-1]

        # top to bottom
        # -------------
        # We need to calculate for each layer, jf=upperjf-1.. lowerjf+1
        # YxL[j, 33] =
        # maxlevel..jf+1: RR @ YxL[j-1]
        # jf..jf+1: RR @ YxL[j-1] - dYxL0 * fac02 + dYxL1 * fac12
        # jf-1..jf: RR @ YxL[j-1] - dYxL0 * fac01 + dYxL1 * fac11
        # minlevel..jf-1: 0
        Yx_3 = [np.zeros(3, dtype=np.complex128)] * (ijf0 - imin + 1)
        Ux_step_lst = [*(-dYx_0[ijf0:ijf1, :3] * fac0_1 + dYx_1[ijf0:ijf1, :3] * fac1_1),
                       *(-dYx_0[ijf1:ijf2, :3] * fac0_2 + dYx_1[ijf1:ijf2, :3] * fac1_2)]
        for Ux_step, RR in zip(Ux_step_lst, Rright[ijf0:ijf2, :3, :3]):
            Yx_3.append(np.dot(RR.T, Yx_3[-1] + Ux_step))
        for RR in Rright[ijf2:imax, :3, :3]:
            Yx_3.append(np.dot(RR.T, Yx_3[-1]))

        M = np.r_[np.conj(YL[imax, :3]), [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]]
        b = np.r_[Yx_3[-1][:3], 0, 0, 0]
        x_ = linalg.solve(M, b)
        Yct = np.conj(YL[imax, 3:])
        Yx_6 = [np.r_[Yx_3.pop(), Yct @ x_]]  # YxL row 1-6 in reverse order

        for RL in Rleft[imax:ijf2:-1, :, 3:]:
            Yx_6.append(np.r_[Yx_3.pop(), np.dot(RL.T, Yx_6[-1])])

        ijf0, ijf1, ijf2 = ijf0 - 1, ijf1 - 1, ijf2 - 1
        Ux_step_lst = [*(+dYx_0[ijf2:ijf1:-1, 3:] * fac0_2 - dYx_1[ijf2:ijf1:-1, 3:] * fac1_2),
                       *(+dYx_0[ijf1:ijf0:-1, 3:] * fac0_1 - dYx_1[ijf1:ijf0:-1, 3:] * fac1_1)]
        for Ux_step, RL in zip(Ux_step_lst, Rleft[ijf2 + 1:ijf0 + 1:-1, :, 3:]):
            Yx_6.append(np.r_[Yx_3.pop(), np.dot(RL.T, Yx_6[-1]) + Ux_step])

        for RL in Rleft[ijf0 + 1:imin:-1, :, 3:]:
            Yx_6.append(np.r_[np.zeros(3, dtype=np.complex128), np.dot(RL.T, Yx_6[-1])])

        new_level = np.r_[True, levels[imin + 1:imax] == levels[imin:imax - 1] + 1]

        # np.einsum('...ij,...j') = [YL[i].T@Yx_6[i] for i in ...]
        return np.einsum('...ji,...j', YL[imin + 1:imax + 1][new_level], np.array(Yx_6[::-1][1:])[new_level])
