"""
Generates preliminary look-up tables (preLUTs) for atmospheric boundary layer simulations. Includes:

- The PreLUTGenerator class, which orchestrates the generation of preLUTs. It integrates
  the state between stations using second-order Runge-Kutta (RK2) methods, adaptive step-size
  control, and transformation routines for the physical variables (u, v, w, p). It collects
  these states into a structured PreLUT used by other components of the PyFuga package.
"""

import importlib
import math

import numpy as np
from tqdm import tqdm

from pyfuga.common import get_cdivkL, get_psi
from pyfuga.constants import COORD_S, COORD_T, MAX_NODES
from pyfuga.preluts import PreLUT

# ---------------------------------------------------------------------------
# IMPORTANT: Dynamic preludium-equivalent switching
# ---------------------------------------------------------------------------
# The unit tests toggle `utils.preludium_equivalent` *at runtime* and then
# reload the module that defines `PreLUTGenerator`. Historically, the entire
# preluts_generator lived in one file, so reloading that file redefined:
#
#   - PrelutNode and PrelutNodeFirst (different base class)
#   - get_new_h2 (normal vs. Preludium version)
#
# After splitting into multiple modules (nodes.py, integration.py, generator.py)
# we must preserve that behaviour.
#
# Therefore:
#   1. We re-import and reload the submodules (`nodes`, `integration`)
#      whenever this module is imported.
#   2. We import PrelutNodeFirst and get_new_h2 only *after* reload.
#
# DO NOT "simplify" this import logic: tests depend on it, and the
# Preludium-vs-standard behaviour must be switchable at module-load time.
# ---------------------------------------------------------------------------
from . import integration as _integration
from . import nodes as _nodes

importlib.reload(_nodes)
importlib.reload(_integration)

from .integration import get_new_h2, integrate_between_stations, modified_midpoint_integration_step  # noqa: E402
from .nodes import PreLUTNodeFirst  # noqa: E402


class PreLUTGenerator:
    """
    Orchestrates the generation of preliminary look-up tables (preLUTs).

    Integrates the system state between (sub)stations using a second-order Runge-Kutta
    (RK2) method with adaptive step-size control. It applies relevant transformations
    for the physical variables and aggregates the resulting nodes into a structured
    preLUT dataset.

    Internally uses variables:
        u,v,w,p and t = u/kappa for kz<kzm
        u,v,w,p and kz          for kz>kzm
        kzm = phi(zm/L)

    - The preLUTs are formulated in terms of u,v,w,p and s = log(z/z0).
    - Stations are located at s = j*ds and at s = log(kzm/kz0) with optional extra stations.
    - Level increments at stations with s = j*ds.
    - Orthonormalisation is performed at each station and the results are appended to a list.
    - Uses second-order Runge-Kutta methods.
    """

    def __init__(self, zeta0, kz0, beta, kzmax, ds, accgoal):
        """
        Initialise the preLUT generator with simulation parameters.

        Args:
            zeta0: The stability parameter (z0/L)
            kz0: Base s-coordinate
            beta: Phase angle
            kzmax: Maximum s-coordinate
            ds: Station spacing
            accgoal: Accuracy goal for integration
        """
        self.zeta0 = float(zeta0)
        self.kz0 = kz0
        self.beta = beta
        self.ds = ds
        self.cosbeta = np.cos(beta)
        self.sinbeta = np.sin(beta)
        self.accgoal = accgoal
        self.kzmax = kzmax
        self.counter = 0

        if zeta0 > 0:
            # Stable
            # log_smaxx is reduced for very stable conditions
            self.log_smaxx = np.log(np.min([kzmax / kz0, 1.0e8, 10 / zeta0]))
        else:
            # Neutral and unstable
            self.log_smaxx = np.log(np.minimum(kzmax / kz0, 1e8))
        self.acc = accgoal / self.log_smaxx
        self.cdivkL = get_cdivkL(zeta0, kz0)
        self.psi0 = get_psi(zeta0, kz0, self.cdivkL)

    def calculate_s_transition(self) -> float:
        """
        Calculate the transitional value of s/t for the current stability conditions.

        s_transition is the value of s (kz) at which the system transitions from the independent variable s to the
        independent variable t. This transition occurs where d/dt = d/ds (dt/ds = 1) and depends on the stability.
        """
        max_iterations = 1000
        tolerance = 1.0e-14
        iterations = 0

        if self.zeta0 < 0:
            # Unstable
            initial_guess = 1 if self.cdivkL > 1 else self.cdivkL**0.2

            # Use Newton-Raphson method to solve for s
            s = initial_guess
            while iterations < max_iterations:
                ds = (s**4 + self.cdivkL * s**5 - 1.0) / (4.0 * s**3 + self.cdivkL * 5.0 * s**4)
                s_new = s - ds
                if math.isclose(s_new, s, rel_tol=tolerance, abs_tol=tolerance):
                    s = s_new
                    break
                s = s_new
                iterations += 1
                if iterations == max_iterations:
                    raise RuntimeError("Maximum iterations reached without convergence.")

            log_s_tr = np.log(s / self.kz0)
        else:
            # Stable and neutral
            if self.cdivkL < 1:
                kzm = 1 / (1 - self.cdivkL)
                log_s_tr = np.log(kzm / self.kz0)
            else:
                log_s_tr = self.log_smaxx
                # kzm = self.kz0 * np.exp(sm)

        log_s_tr = min(self.log_smaxx, log_s_tr)
        return log_s_tr

    def make_prelut(self):
        """
        Generate the preLUT dataset by integrating the state between stations.

        Returns:
            PreLUT: A structured dataset containing the preLUT and associated metadata.
        """
        self.nodes = []
        first = PreLUTNodeFirst(self.beta, self.ds)
        h = np.sqrt(self.acc * 6 / 3.125)
        self.lastkz = self.kz0

        yerr = self.modified_midpoint_integration_step(first, first.Y_lower, 0.0, h, COORD_T)
        first.reset_forcing_accumulators()
        h = get_new_h2(h, self.acc, yerr, first.Y_upper)
        s_tr = self.calculate_s_transition()

        # cumsum gives slightly different results than arange (more equal to fortran implementation)
        log_s_lst = np.sort(np.r_[0, np.cumsum(np.full(int(self.log_smaxx // self.ds) + 1, self.ds)), s_tr])

        # equal(first.Y_lower, f'yleft{0:6.3f}')
        segment, h = self.integrate_between_stations(first, h, yerr, self.acc, COORD_T)
        # equal(segment.Y_upper, f'yright{0:6.3f}')
        for log_s1, log_s2 in tqdm(list(zip(log_s_lst[1:], log_s_lst[2:])), disable=True):  # noqa: B905
            self.nodes.append(segment)
            segment = segment.generate_next_node(log_s1, log_s2)

            # Before s_tr: integrate in t, after s_tr: integrate in s
            coordsys = COORD_T if log_s1 < s_tr else COORD_S

            segment, h = self.integrate_between_stations(segment, h, yerr, self.acc, coordsys)
            # equal(segment.Y_upper, f'yright{s1:6.3f}')
            if len(self.nodes) > MAX_NODES:  # pragma: no cover
                break
        else:  # max_recs not reached
            if s_tr < self.log_smaxx:
                self.nodes.append(segment)
                segment.generate_next_node(log_s_lst[-1], log_s_lst[-1])
                segment.Y_upper = segment.Y_lower
            # compare(segment.Y_upper, 'yright%6.3f' % s1)
            # if s2 < self.smaxx:
            #     self.nodes.append(segment)

            # allocate(segment%next)
            # segment%next%prev=>segment
            # call GMRES(segment)
            # segment=>segment%next

        # if s2 == self.smaxx:
        #     self.nodes.pop(-1)
        # else:
        #     last.s_upper = last.s_lower
        #     last.yright = last.Y_lower
        #     last.R_upper = self.nodes[0].R_lower
        #

        # segment.dat.level = i
        # segment.dat.s_lower = s1
        # segment.dat.s_upper = segment.dat.s_lower = s1
        # segment.Y_upper = segment.dat.Y_lower

        # segment.dat.R_upper = np.zeros_like(segment.dat.Y_lower)
        # self.last = segment

        # def get_res(node, k):
        #     if node.next is None:
        #         return [getattr(node.dat, k)]
        #     else:
        #         return [getattr(node.dat, k)] + get_res(node.next, k)
        # print(self.counter)
        var_names = [
            "Y_lower",
            "R_lower",
            "R_upper",
            "dbx_const",
            "dbx_lin",
            "dby_const",
            "dby_lin",
            "dbz_const",
            "dbz_lin",
            "log_s_lower",
            "log_s_upper",
        ]
        var_values = [np.moveaxis(np.array([getattr(n, k) for n in self.nodes]), 1, 2) for k in var_names[:3]] + [
            np.array([getattr(n, k) for n in self.nodes]) for k in var_names[3:]
        ]
        return PreLUT(
            {
                **{n: (("i", "j", "k")[: len(v.shape)], v) for n, v in zip(var_names, var_values, strict=True)},
                "beta": self.beta,
                "kz0": self.kz0,
                **{"level": (("i",), np.round(var_values[-2] / self.ds, 3).astype(int))},
            },
            attrs={"ds": self.ds, "kzmax": self.kzmax, "zeta0": self.zeta0, "accgoal": self.accgoal},
        )

    def modified_midpoint_integration_step(self, node, y, x, h, j):
        """
        A wrapper for the jit-compilable rk2 function to allow it to be called from within a class.

        Args:
            ...
            j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        """
        Y_upper, yerr = modified_midpoint_integration_step(
            y,
            x,
            h,
            j,
            self.kz0,
            self.psi0,
            self.lastkz,
            self.zeta0,
            self.cdivkL,
            self.cosbeta,
            self.sinbeta,
            node.dbx_const,
            node.dbx_lin,
            node.dby_const,
            node.dby_lin,
            node.dbz_const,
            node.dbz_lin,
        )
        node.Y_upper = Y_upper
        return yerr

    def integrate_between_stations(self, p, h, yerr, acc, j):
        """
        A wrapper for the jit-compilable integrate_between_stations function to allow it to be called from within a
        class.

        Adjust the integration step size until the accuracy requirement is met.

        This function repeatedly applies RK2 integration steps to update the state until the
        accumulated error is within the specified tolerance.

        Args:
            p: The current node in the preLUT generation process.
            h: The current integration step size.
            yerr: The accumulated error in the state.
            acc: The accuracy goal for the integration.
            j: Coordinate system indicator (COORD_T for t, COORD_S for s).
        """
        # return self.integrate_between_stations_old(p, h, yerr, acc, j)
        while True:
            log_s_upper = p.log_s_upper
            Y_upper, h, s2, lastkz = integrate_between_stations(
                p.Y_lower,
                p.log_s_lower,
                log_s_upper,
                p.dbx_const,
                p.dby_const,
                p.dbz_const,
                p.dbx_lin,
                p.dby_lin,
                p.dbz_lin,
                h,
                yerr,
                acc,
                j,
                self.kz0,
                self.lastkz,
                self.zeta0,
                self.cdivkL,
                self.psi0,
                self.cosbeta,
                self.sinbeta,
            )
            self.lastkz = lastkz
            p.Y_upper = Y_upper

            if s2 >= p.log_s_upper:
                break

            p.log_s_upper = s2
            self.nodes.append(p)
            p = p.generate_next_node(s2, log_s_upper)

        return p, h
