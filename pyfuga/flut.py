"""flut.py
========
Generate Fourier-space look-up tables from atmospheric boundary layer solutions.

This module implements the second stage of the PyFuga pipeline: the
Fourier-space perturbation solution derived from the adjoint-problem
solution stored in the preliminary LUTs.

**Input**
    Preliminary LUTs (:class:`pyfuga.preluts.PreLUT`) from
    :mod:`pyfuga.preluts_generator`. These encode the state profile
    (velocity, pressure) as a function of height and horizontal wavenumber.

**Output**
    Fourier LUTs (:class:`xarray.Dataset`) with variables ``UL``, ``VL``, ``WL``,
    ``PL`` (L = longitudinal forcing) and ``UT``, ``VT``, ``WT``, ``PT``
    (T = transverse forcing), each with dimensions ``(beta, kz0, level)``.
    These are passed to :mod:`pyfuga.trafalgar` for inverse Fourier transform.

**Method**
    Solves the superposition integral equation for each ``(beta, kz0)`` pair
    (polar angle and wavenumber), ranging over all levels in the rotor plane
    or at hub height. Uses parallel processing where applicable.
"""

import multiprocessing as mp
import warnings
from contextlib import nullcontext

import numpy as np
from numpy import linalg
from numpy import newaxis as na
from tqdm import tqdm

from pyfuga.constants import FIRST_Z_OUTPUT, KAPPA, UVW_LT
from pyfuga.utils import ComplexXRDataset, jit


class FourierLUTGenerator:
    def __init__(self, preluts, zhub, diameter, zi, verbose=True):
        self.preluts = preluts
        self.zhub = zhub
        self.radius = diameter / 2
        self.zi = zi  # Domain height
        self.verbose = verbose

    def make_rotor_luts(self, z0, luts=UVW_LT, n_cpu=1):  # for wake solution at all levels within rotor plane
        # note: can change z0 --> modify T.I. through z=z0*exp(1/T.I.)
        ds = self.preluts.ds
        # Calculate j levels within rotor plane, including points at ceiling (zh+R) and floor (zh-R).
        low_level_out = int(np.floor(np.log((self.zhub - self.radius) / z0) / ds))
        high_level_out = int(np.ceil(np.log((self.zhub + self.radius) / z0) / ds))
        return self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)

    def make_hubheight_luts(self, z0, luts=UVW_LT, n_cpu=1):  # for wake solution only at hub height
        ds = self.preluts.ds
        low_level_out = int(np.floor(np.log((self.zhub) / z0) / ds))
        high_level_out = int(np.ceil(np.log((self.zhub) / z0) / ds))

        luts = self.make_lut(z0, low_level_out, high_level_out, luts, n_cpu=n_cpu)
        if low_level_out == high_level_out:
            lut_vars = luts.isel(level=0)
        else:
            a = np.log(self.zhub / z0) / ds - low_level_out
            lut_vars = luts.isel(level=0) * (1 - a) + luts.isel(level=1) * (a)
        lut_vars["level"] = 9999

        return ComplexXRDataset(
            data_vars={
                "z": (("level"), [self.zhub]),
                "k": luts.k,
                "diameter": luts.diameter,
                "hubheight": self.zhub,
                "z0": z0,
                # fourier luts, UL, UT, ...
                **{k: lut_vars[k].expand_dims("level", 2) for k in lut_vars if lut_vars[k].dims == ("beta", "kz0")},
            },
            coords={"beta": luts.beta, "kz0": luts.kz0, "level": [9999]},
            attrs=luts.attrs,
        )

    def make_lut(self, z0, low_level_out, high_level_out, luts=UVW_LT, n_cpu=1):
        """
        Generate Fourier-space wake look-up tables for specified height range.

        This is the core method of the second stage in the PyFuga pipeline. It
        takes preliminary LUTs containing the adjoint-problem solution and solves
        for spectral wake response across all ``(beta, kz0)`` wavenumber pairs
        for the specified problem geometry and turbine characteristics. Output is
        suitable for inverse Fourier transformation by
        :class:`pyfuga.trafalgar.Trafalgar`.

        Parameters
        ----------
        z0 : float
            Roughness length (m). Defines the vertical grid spacing via logarithmic
            height increments (s = log(z/z0)).
        low_level_out : int
            Lowest output level index. Defines minimum height on the vertical grid.
        high_level_out : int
            Highest output level index. Defines maximum height on the vertical grid.
        luts : sequence of str, optional
            Variable names to generate. Default is all 8 variables: 'UL', 'UT', 'VL',
            'VT', 'WL', 'WT', 'PL', 'PT' (L = longitudinal forcing, T = transverse
            forcing). Variables are u, v, w (velocity components) and p (pressure).
        n_cpu : int, optional
            Number of CPU cores for parallel (beta, kz0) computation. Default is 1
            (serial). Pass None to use all available cores.

        Returns
        -------
        xarray.Dataset
            Fourier-space wake fields with variables (as requested), each with
            dimensions (beta, kz0, level). Dimensions correspond to:
            - beta: azimuthal angles (radians)
            - kz0: wavenumber magnitude scaled to domain height
            - level: vertical index on the log-height grid
            Also includes scalar coordinates: diameter, hubheight, z0.

        See Also
        --------
        pyfuga.trafalgar.Trafalgar.make_luts : Next pipeline stage; inverse Fourier transform.
        pyfuga.preluts_generator.PreLUTGenerator.make_prelut : Previous stage; generates input.

        Notes
        -----
        The method validates that output levels fit within the precomputed preLUT range.
        Wavenumbers larger than those with wavelengths < R/4 (rotor radius/4) are filtered
        out as physically irrelevant. The computation is embarrassingly parallel over
        (beta, kz0) pairs and uses multiprocessing when n_cpu > 1.
        """
        assert all(lut in UVW_LT for lut in luts)
        zh = self.zhub
        R = self.radius
        ds, kzmax, zeta0 = [self.preluts.attrs[k] for k in ["ds", "kzmax", "zeta0"]]

        # Get all unique kz0 values from the prelut, sorted in ascending order
        kz0_lst = np.sort(np.unique(self.preluts.kz0.values))

        # Filter out wavenumbers that are too large (wavelengths smaller than R/4)
        nkz0 = int(np.round(1 / np.diff(np.log10(kz0_lst))[0])) if len(kz0_lst) > 1 else 1
        aux = z0 * np.pi * 8.0 / R
        kz0limit = 10.0 ** (np.ceil(np.log10(aux) * nkz0) / nkz0)
        kz0_lst = kz0_lst[kz0_lst <= kz0limit]  # keep only physically relevant wavenumbers

        upperjf = int(np.floor(np.log((zh + R) / z0) / ds))  # top of rotor, excluding point at ceiling (zh+R).
        lowerjf = int(np.ceil(np.log((zh - R) / z0) / ds))  # bottom of rotor, excluding point at floor (zh-R).
        minlevel = int(np.floor(np.log(np.maximum(FIRST_Z_OUTPUT / z0, 1)) / ds))  # ground
        maxlevel = int(np.ceil(np.log(self.zi / z0) / ds))  # inversion height

        # Validate that output levels are within the range of prelut levels
        if low_level_out < minlevel + 1:
            print(f"LoLevelOut ({low_level_out}) raised to MinLevel ({minlevel + 1}).")
            low_level_out = minlevel + 1

        if high_level_out > maxlevel - 1:
            print(f"HiLevelOut ({high_level_out}) lowered to MaxLevel ({maxlevel - 1}).")
            high_level_out = maxlevel - 1

        logZiZ0 = np.log(self.zi / z0)

        # Calculate maximum meaningful vertical wake extent
        # assert smaxx>self.preluts.ds * upperjf # TODO: smaxx not available here
        smax = np.minimum(np.log((16 * np.pi * (1 + zh / R) + 120.0) / kz0_lst), logZiZ0)  # Neutral
        if np.abs(zeta0) > 0:
            smax = np.minimum(smax, np.log(np.abs(15 / zeta0)))  # Stable and unstable

        # Create list of (beta, kz0) combinations for which to solve the layers
        beta_lst = np.sort(np.unique(self.preluts.beta.values))
        ktab = kz0_lst / z0
        beta_kz0_lst = [(beta, kz0) for beta in beta_lst for kz0 in kz0_lst]

        # Choose serial vs parallel mapping
        if n_cpu in (None, 1):
            pool_cm = nullcontext()
            map_func = map
        else:
            ctx = mp.get_context("spawn")  # or "forkserver"
            pool_cm = ctx.Pool(processes=n_cpu)
            map_func = pool_cm.imap

        xL = []  # longitudinal forcing
        xT = []  # transversal forcing

        with pool_cm as pool:
            # If parallel, map_func should come from the pool
            if n_cpu not in (None, 1) and pool is not None:
                map_func = pool.imap

            if any(lut[1] == "L" for lut in luts):
                args_lst = (
                    (
                        self.preluts.sel(beta=beta, kz0=kz0),
                        beta,
                        kz0,
                        z0,
                        self.zhub,
                        self.radius,
                        "L",
                        lowerjf,
                        upperjf,
                        minlevel,
                        maxlevel,
                        low_level_out,
                        high_level_out,
                    )
                    for beta, kz0 in beta_kz0_lst
                )

                xL = list(
                    tqdm(
                        map_func(solve_layers, args_lst),
                        total=len(beta_kz0_lst),
                        disable=(not self.verbose),
                        desc="Fourier LUTS: Solving for longitudinal forcing",
                    )
                )

            if any(lut[1] == "T" for lut in luts):
                args_lst = (
                    (
                        self.preluts.sel(beta=beta, kz0=kz0),
                        beta,
                        kz0,
                        z0,
                        self.zhub,
                        self.radius,
                        "T",
                        lowerjf,
                        upperjf,
                        minlevel,
                        maxlevel,
                        low_level_out,
                        high_level_out,
                    )
                    for beta, kz0 in beta_kz0_lst
                )

                xT = list(
                    tqdm(
                        map_func(solve_layers, args_lst),
                        total=len(beta_kz0_lst),
                        disable=(not self.verbose),
                        desc="Fourier LUTS: Solving for transversal forcing",
                    )
                )

        def get_var(s):
            """Extract specific component and forcing direction from results."""
            x = xL if s[1] == "L" else xT
            i = {"U": 0, "V": 2, "W": 4, "P": 5}[s[0]]
            # Extract: (n_beta, n_kz0, n_levels)
            return np.reshape(x, (len(beta_lst), len(kz0_lst)) + np.shape(x)[1:])[..., i]

        level = np.arange(low_level_out, high_level_out + 1)
        z = z0 * np.exp(ds * level)  # defining j levels / stations

        return ComplexXRDataset(
            data_vars={
                "z": (("level"), z),
                "k": (("kz0"), ktab),
                "diameter": R * 2,
                "hubheight": zh,
                "z0": z0,
                **{k: (("beta", "kz0", "level"), get_var(k)) for k in luts},
            },
            coords={"beta": beta_lst, "kz0": kz0_lst, "level": level},
            attrs={"zi": self.zi, **self.preluts.attrs},
        )


def solve_layers(args):
    (
        prelut,
        beta,
        kz0,
        z0,
        zhub,
        radius,
        forcing,
        lowerjf,
        upperjf,
        minlevel,
        maxlevel,
        low_level_out,
        high_level_out,
    ) = args
    """
    Solves for a single (beta, kz0, forcing direction) combination.
    Discards pre-calculated levels outside of min/max level range for memory optimisation.
    Returns: X(z) at all output levels for that specific (beta, kz0, forcing direction) combination.
    """

    if forcing not in ("L", "T"):
        raise ValueError(f"forcing must be 'L' or 'T', got {forcing!r}")

    axis = forcing.replace("L", "x").replace("T", "y")
    ds = prelut.ds
    jf_l = np.arange(lowerjf + 1, upperjf)  # layers within rotor

    z = z0 * np.exp(ds * (jf_l + np.array([-1, 0, 1])[:, na]))  # height blow, at and above current layer
    k = kz0 / z0

    max_table_level = prelut.level.max().item()

    # Memory optimisation: discard irrelevant prelut data
    if max_table_level < minlevel:
        return np.zeros((high_level_out - low_level_out + 1, 6), dtype=np.complex128)  # Discard if below min level

    imin, imax = np.searchsorted(prelut.level, [minlevel, min(maxlevel, max_table_level)])
    prelut = prelut.sel(i=slice(imin, imax))  # discard unused levels (outside min/max range)

    zero_pad_levels = int(max(0, maxlevel - max_table_level))

    ijf0_l, ijf1_l, ijf2_l = np.searchsorted(prelut.level, [jf_l - 1, jf_l, jf_l + 1])  # the three levels per layer
    # e.g. ijf0_l[0], ijf1_l[0], ijf2_l[0] makes up the first layer

    Q = prelut.Y_lower.values
    R_upper = prelut.R_upper.values
    R_lower = prelut.R_lower.values
    db_const = prelut[f"db{axis}_const"].values
    db_lin = prelut[f"db{axis}_lin"].values
    levels = prelut.level.values.astype(int)

    # Triangular/Chapeau function weights
    phi_const_low_lst = z[0] / (KAPPA * k * (z[1] - z[0]))  # constant part of Chapeau function for lowest part of layer
    phi_lin_low_lst = 1 / (KAPPA * (z[1] - z[0]) * k**2)  # prop. part of Chapeau function for lowest part of layer
    phi_const_high_lst = z[2] / (KAPPA * k * (z[1] - z[2]))  # const. -- for upper part of layer
    phi_lin_high_lst = 1 / (KAPPA * (z[1] - z[2]) * k**2)

    # Solve for each layer - returns (n_levels, 6) for each layer
    output = [
        np.concatenate(
            [
                solve_layer(
                    R_upper,
                    R_lower,
                    Q,
                    levels,
                    db_const,
                    db_lin,
                    phi_const_low,
                    phi_lin_low,
                    phi_const_high,
                    phi_lin_high,
                    ijf0,
                    ijf1,
                    ijf2,
                ),
                np.zeros((zero_pad_levels, 6)),
            ]
        )
        for ijf0, ijf1, ijf2, phi_const_low, phi_lin_low, phi_const_high, phi_lin_high in zip(
            ijf0_l,
            ijf1_l,
            ijf2_l,
            phi_const_low_lst,
            phi_lin_low_lst,
            phi_const_high_lst,
            phi_lin_high_lst,
            strict=True,
        )
    ]

    # Apply geometric weighting (Fourier transform of circular rotor)
    zf = z0 * np.exp(ds * np.arange(lowerjf, upperjf + 1))
    layer_halfwidth = np.sqrt(radius**2 - (zf - zhub) ** 2)
    ky = k * np.sin(beta)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", r"invalid value encountered in divide")
        fac = np.where(ky == 0, layer_halfwidth, np.sin(ky * layer_halfwidth) / ky)

    area_err_fac = radius**2 * np.pi / np.sum(np.sqrt(radius**2 - (zf - zhub) ** 2) * zf * (np.exp(ds) - np.exp(-ds)))

    # Extract output levels
    s = slice(low_level_out - minlevel - 1, high_level_out - minlevel)
    output = np.array(output)[:, s]  # dim: (from_level, to_level, uu'vv'wp)

    # Sum over layers with geometric weights
    return np.sum(fac[1:-1][:, na, na] * output, 0) * area_err_fac


@jit  # (f'complex128[:,:]({c3}{c3}{c3}{i1}{c2}{c2}{d}{d}{d}{d}{i}{i}{i})')
def solve_layer(
    R_upper,
    R_lower,
    Q,
    levels,  # excluding substations
    db_const,
    db_lin,
    phi_const_low,  # 4 Triangular/Chapeau function weights
    phi_lin_low,
    phi_const_high,
    phi_lin_high,
    icl_m1,  # icl-1
    icl,  # i_current level
    icl_p1,  # icl+1
):
    """
    Loops over a single (beta, kz0, layer, forcing direction) combination.
    Returns: X(z) at all output levels for that specific layer and forcing.
    NB: Forcing contributions are calculated using all (sub-)stations, but the output only includes the main stations.

    maxlevel (z_inversion or table_max_level)
    cl+1
    cl (current level)
    cl-1
    minlevel (1m)

    'i'-prefix, e.g. icl is the node number of cl (current layer)
    """
    icl_m1, icl, icl_p1 = icl_m1.item(), icl.item(), icl_p1.item()  # icl-1/icl/icl+1 as int
    b_lower_3 = [np.zeros(3, dtype=np.complex128)] * (icl_m1 + 1)  # minlevel(z=1m) to cl-1 -- b for lower BC

    forcing_increments_up = np.concatenate(
        (
            (-db_const[icl_m1:icl, :3] * phi_const_low + db_lin[icl_m1:icl, :3] * phi_lin_low),  # cl-1 to cl
            (-db_const[icl:icl_p1, :3] * phi_const_high + db_lin[icl:icl_p1, :3] * phi_lin_high),  # cl to cl+1
        )
    )

    # cl-1 to cl+1
    R_upper_slice = R_upper[icl_m1:icl_p1, :3, :3]
    if len(forcing_increments_up) != len(R_upper_slice):
        raise ValueError("Length mismatch between forcing_increments_up and R_upper slice")
    for F_increment, RR in zip(forcing_increments_up, R_upper_slice):  # noqa: B905
        b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1] + F_increment))

    # cl+1 to max level
    for RR in R_upper[icl_p1:-1, :3, :3]:
        b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1]))

    # at max level
    Y_tilde = np.concatenate(
        (
            np.conj(Q[-1, :3]),
            np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=np.complex128),
        )
    )
    b_tilde = np.concatenate((b_lower_3[-1][:3], np.asarray([0 + 0j, 0, 0])))
    x_ih = linalg.solve(Y_tilde, b_tilde)
    Y_ih_last_3 = np.conj(Q[-1, 3:])
    b_full_6 = [np.concatenate((b_lower_3.pop(), Y_ih_last_3 @ x_ih))]

    # max level to cl+1
    for RL in R_lower[-1:icl_p1:-1, :, 3:]:
        b_full_6.append(np.concatenate((b_lower_3.pop(), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1]))))

    icl_m1, icl, icl_p1 = icl_m1 - 1, icl - 1, icl_p1 - 1
    forcing_increments_down = np.concatenate(
        (
            (+db_const[icl_p1:icl:-1, 3:] * phi_const_high - db_lin[icl_p1:icl:-1, 3:] * phi_lin_high),  # cl+1 to cl
            (+db_const[icl:icl_m1:-1, 3:] * phi_const_low - db_lin[icl:icl_m1:-1, 3:] * phi_lin_low),
        )
    )

    # cl+1 to cl-1
    R_lower_slice = R_lower[icl_p1 + 1 : icl_m1 + 1 : -1, :, 3:]
    if len(forcing_increments_down) != len(R_lower_slice):
        raise ValueError("Length mismatch between forcing_increments_down and R_lower slice")
    for F_increment, RL in zip(forcing_increments_down, R_lower_slice):  # noqa: B905
        b_full_6.append(
            np.concatenate((b_lower_3.pop(), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1]) + F_increment))
        )

    # cl-1 to min_level
    for RL in R_lower[icl_m1 + 1 : 0 : -1, :, 3:]:
        b_full_6.append(
            np.concatenate((np.zeros(3, dtype=np.complex128), np.dot(np.ascontiguousarray(RL.T), b_full_6[-1])))
        )

    # Only keep the levels corresponding to the original prelut levels, excluding substations.
    new_level = np.concatenate((np.array([True]), levels[1:-1] > levels[:-2]))

    filtered_q_levels = [v for v, nl in zip(Q[1:], new_level) if nl]  # noqa: B905
    filtered_b_levels = [v for v, nl in zip(b_full_6[::-1][1:], new_level) if nl]  # noqa: B905
    if len(filtered_q_levels) != len(filtered_b_levels):
        raise ValueError("Length mismatch between filtered Q and b levels")

    paired_levels = zip(filtered_q_levels, filtered_b_levels)  # noqa: B905

    return [(np.ascontiguousarray(Q_level.T) @ b_level) for Q_level, b_level in paired_levels]
