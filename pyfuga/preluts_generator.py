import numpy as np
from tqdm import tqdm

from pyfuga import utils
from pyfuga.common import cdivkL, get_new_h2, phi, phi_inverse, psi
from pyfuga.constants import COORD_S, COORD_T, Cm1, Cm2, Ythreshold, kappa, kappa2, max_recs, n_eq
from pyfuga.preluts import PreLUT
from pyfuga.utils import jit

# np.set_printoptions(precision=2, linewidth=200)
# left: (sub)station (lower height)
# right: (sub)station + 1 (higher height)
# (Y+).x = b is a set of boundary condition equations.
# R is a Gram-Smidt (or QR) transformation of Y
# dbx_const = Delta_b for constant longitudinal forcing
# dbx_lin = Delta_b for longitudinal forcing proportional to kz
# Additional if transversal and/or vertical forcing is present:
#  dby_const = Delta_b for constant transversal forcing
#  dby_lin = Delta_b for transversal forcing proportional to kz
#  dbz_const = Delta_b for constant vertical forcing
#  dbz_lin = Delta_b for vertical forcing proportional to kz
# Delta_b = integral of (Y+).f between two  (sub)stations
# where f is the wind turbine forcing.


class PrelutNode:
    """
    Represents a node in the preLUT generation process.

    Encapsulates state variables and error accumulators at a station and provides
    methods to reset errors and advance to the next node using a QR decomposition.
    """

    def __init__(self):
        self.reset_dyx()

    def set_s(self, sleft, sright):
        """Set the lower and upper bounds for the segment in s."""
        self.sleft = sleft
        self.sright = sright

    def reset_dyx(self):
        """Reset the differential error accumulators for forcing."""
        self.dbx_const = np.zeros(n_eq, dtype=np.complex128)
        self.dby_const = np.zeros(n_eq, dtype=np.complex128)
        self.dbz_const = np.zeros(n_eq, dtype=np.complex128)
        self.dbx_lin = np.zeros(n_eq, dtype=np.complex128)
        self.dby_lin = np.zeros(n_eq, dtype=np.complex128)
        self.dbz_lin = np.zeros(n_eq, dtype=np.complex128)

    def get_next(self, sleft, sright):
        """Generate and return the next node via QR decomposition."""
        next_node = self.GMRES()
        next_node.set_s(sleft, sright)
        return next_node

    def GMRES(self):
        """
        Perform QR decomposition of the final state matrix of a segment, Yright, to
        get the first statement matrix of the next segment, Yleft. Store the
        decomposition results (Rleft, Yleft, Rright) in the new node.
        """
        node = PrelutNode()
        Rleft, Yleft, Rright = gmres(self.Yright)
        node.Rleft = Rleft
        node.Yleft = Yleft
        self.Rright = Rright
        return node


if utils.preludium_equivalent:
    from pyfuga.preludium_eq_routines import PrelutNodePreludium as PrelutNode
    from pyfuga.preludium_eq_routines import get_new_h2_Preludium as get_new_h2


@jit("Tuple((complex128[:,:],complex128[:,:],complex128[:,:]))(complex128[:,:])")
def gmres(Yright):
    """
    Perform a QR decomposition of an input matrix with columns representing linearly independent vectors.

    Returns a tuple containing:
        1) the lower triangular matrix, Rleft,
        2) the unitary, orthonormal basis matrix, Yleft, and
        3) Rright, which is the inverse of the Hermitian adjoint (conjugate transpose) of Rleft.
    """
    Yleft, Rleft = np.linalg.qr(Yright)
    Rleft = np.conj(Rleft.T)
    Rright = np.linalg.inv(Rleft)
    return Rleft, Yleft, Rright


class PrelutNodeFirst(PrelutNode):
    """
    Specialised first node for preLUT generation.

    Initialises the node with boundary conditions based on the phase angle (beta) and station spacing (ds).
    """

    def __init__(self, beta, ds):
        """Initialise the first node with the prescribed boundary conditions."""
        PrelutNode.__init__(self)
        self.set_s(sleft=0, sright=ds)
        sinbeta, cosbeta = np.sin(beta), np.cos(beta)
        self.Yleft = np.array(
            [
                [-sinbeta, 0, cosbeta, 0, 0, 0],
                [cosbeta, 0, sinbeta, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, -sinbeta, 0, cosbeta, 0, 0],
                [0, cosbeta, 0, sinbeta, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.complex128,
        ).T
        self.Rleft = np.eye(6, dtype=np.complex128)


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
            kzmaz: Maximum s-coordinate
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
            # smaxx is reduced for very stable conditions
            self.smaxx = np.log(np.min([kzmax / kz0, 1.0e8, 10 / zeta0]))
        else:
            # Neutral and unstable
            self.smaxx = np.log(np.minimum(kzmax / kz0, 1e8))
        self.acc = accgoal / self.smaxx
        self.cdivkL = cdivkL(zeta0, kz0)
        self.psi0 = psi(zeta0, kz0, self.cdivkL)

    def sm(self):
        """
        Calculate the transitional value of s/t for the current stability conditions.

        sm is the value of s (kz) at which the system transitions from the independent variable s to the
        independent variable t. This transition occurs where d/dt = d/ds (dt/ds = 1) and depends on the stability.
        """
        # Determine max s (sm) and max kz (kzm)?
        if self.zeta0 < 0:
            # Unstable
            if self.cdivkL > 1:
                x = 1
            else:  # pragma: no cover
                x = self.cdivkL**0.2

            while True:
                dx = (1.0 - x**4 - self.cdivkL * x**5) / (4.0 * x**3 + self.cdivkL * 5.0 * x**4)
                x = x + dx
                if abs(dx / x) < 1.0e-14:
                    break
            # kzm = x
            sm = np.log(x / self.kz0)
        else:
            # Stable and neutral
            if self.cdivkL < 1:
                kzm = 1 / (1 - self.cdivkL)
                sm = np.log(kzm / self.kz0)
            else:
                sm = self.smaxx
                # kzm = self.kz0 * np.exp(sm)

        sm = min(self.smaxx, sm)
        return sm

    def make_prelut(self):
        """
        Generate the preLUT dataset by integrating the state between stations.

        Returns:
            PreLUT: A structured dataset containing the preLUT and associated metadata.
        """
        self.nodes = []
        first = PrelutNodeFirst(self.beta, self.ds)
        h = np.sqrt(self.acc * 6 / 3.125)
        self.lastkz = self.kz0

        yerr = self.rk2(first, first.Yleft, 0.0, h, COORD_T)
        first.reset_dyx()
        h = get_new_h2(h, self.acc, yerr, first.Yright)
        sm = self.sm()

        # cumsum gives slightly different results than arange (more equal to fortran implementation)
        s_lst = np.sort(np.r_[0, np.cumsum(np.full(int(self.smaxx // self.ds) + 1, self.ds)), sm])

        # equal(first.Yleft, f'yleft{0:6.3f}')
        segment, h = self.integrate_between_stations(first, h, yerr, self.acc, COORD_T)
        # equal(segment.Yright, f'yright{0:6.3f}')
        for s1, s2 in tqdm(list(zip(s_lst[1:], s_lst[2:])), disable=1):  # noqa: B905
            self.nodes.append(segment)
            segment = segment.get_next(s1, s2)

            # Before sm: integrate in t, after sm: integrate in s
            coordsys = COORD_T if s1 < sm else COORD_S

            segment, h = self.integrate_between_stations(segment, h, yerr, self.acc, coordsys)
            # equal(segment.Yright, f'yright{s1:6.3f}')
            if len(self.nodes) > max_recs:  # pragma: no cover
                break
        else:  # max_recs not reached
            if sm < self.smaxx:
                self.nodes.append(segment)
                segment.get_next(s2, s2)
                segment.Yright = segment.Yleft
            # compare(segment.Yright, 'yright%6.3f' % s1)
            # if s2 < self.smaxx:
            #     self.nodes.append(segment)

            # allocate(segment%next)
            # segment%next%prev=>segment
            # call GMRES(segment)
            # segment=>segment%next

        # if s2 == self.smaxx:
        #     self.nodes.pop(-1)
        # else:
        #     last.sright = last.sleft
        #     last.yright = last.Yleft
        #     last.Rright = self.nodes[0].Rleft
        #

        # segment.dat.level = i
        # segment.dat.sleft = s1
        # segment.dat.sright = segment.dat.sleft = s1
        # segment.Yright = segment.dat.Yleft

        # segment.dat.Rright = np.zeros_like(segment.dat.Yleft)
        # self.last = segment

        # def get_res(node, k):
        #     if node.next is None:
        #         return [getattr(node.dat, k)]
        #     else:
        #         return [getattr(node.dat, k)] + get_res(node.next, k)
        # print(self.counter)
        var_names = [
            "Yleft",
            "Rleft",
            "Rright",
            "dbx_const",
            "dbx_lin",
            "dby_const",
            "dby_lin",
            "dbz_const",
            "dbz_lin",
            "sleft",
            "sright",
        ]
        var_values = [np.moveaxis([getattr(n, k) for n in self.nodes], 1, 2) for k in var_names[:3]] + [
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

    def rk2(self, node, y, x, h, j):
        """
        A wrapper for the jit-compilable rk2 function to allow it to be called from within a class.

        Args:
            ...
            j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        """
        Yright, yerr = modified_midpoint_integration_step(
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
        node.Yright = Yright
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
            sright = p.sright
            Yright, h, s2, lastkz = integrate_between_stations(
                p.Yleft,
                p.sleft,
                sright,
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
            p.Yright = Yright

            if s2 >= p.sright:
                break

            p.sright = s2
            self.nodes.append(p)
            p = p.get_next(s2, sright)

        return p, h


@jit("double(double, double, double, double, double, double)")
def get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL):
    """Get kz from t and stability parameter zeta0."""

    kz = lastkz
    if zeta0 < 0:
        # Unstable -psi_m at z=z0 plus a constant?
        b0 = psi0 + np.log(Cm1 * zeta0 / 8)
    else:
        b0 = 0
    if np.abs(zeta0) < 1e-14:
        # Neutral
        kz = kz0 * np.exp(t)
    elif zeta0 > 0:
        # Stable
        a = Cm2 * zeta0
        b = t + a + np.log(a)
        if b < 1:  # pragma: no cover
            if kz < 0:
                ax = np.exp(b)
            else:
                ax = a * lastkz / kz0
            while True:
                dax = (np.exp(b - ax) - ax) / (1 + ax)
                ax = ax + dax
                if abs(dax / ax) < 1e-14:
                    break
        else:
            if kz < 0:  # pragma: no cover
                ax = b
            else:
                ax = a * lastkz / kz0
            while True:
                dax = (b - ax - np.log(ax)) / (1 + 1 / ax)
                ax = ax + dax
                if abs(dax / ax) < 1e-14:
                    break
        kz = kz0 * ax / a
    else:
        # Unstable
        b = t + b0
        if kz < 0:  # pragma: no cover
            x = np.exp(b)
        else:
            aux = phi_inverse(lastkz, cdivkL)
            x = (cdivkL * lastkz) / ((aux**2 + 1) * (1 + aux) ** 2)
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1) ** 2
        while abs(dx / x) > 1e-14:
            # print(dx)
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1) ** 2
            x = x + dx
            if x < 0:  # pragma: no cover
                x = np.exp(b)
                dx = x
        kz = 8 * x * (1 + x**2) / (cdivkL * (1 - x) ** 4)
    return kz


@jit("double(double,double,double,double)")
def u0(kz, kz0, psi, psi0):
    """
    Compute the wind speed from Monin-Obukhov Similarity Theory (MOST) normalised by
    the friction velocity, ustar.

    Arg:
        kz: Current wavenumber
        kz0: Base wavenumber
        psi: Stability correction function at kz
        psi0: Stability correction function at kz0

    Returns:
        The normalised wind speed
    """
    return (np.log(kz / kz0) + psi - psi0) / kappa


@jit("complex128[:,:](int32,double,double, double, double,double,double,double,double)")
def getM(j, t, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta):
    """
    Compute the adjoint system matrix M = -A* with stability effects included in the eddy diffusivity.
    Refactored to use helper functions for clarity.

    Args:
        j: Coordinate system indicator (COORD_T = 1: t = u0*kappa, COORD_S = 2: s = kz).
        t: The input coordinate (t or s depending on j).
        kz0: The base s-coordinate.
        psi0: The base stability correction value at kz0.
        lastkz: The s-coordinate from the last station or integration step.
        zeta0: The stability parameter (z0/L).
        cdivkL: Eddy diffusivity scaling parameter.
        cosbeta: The cosine of the phase angle.
        sinbeta: The sine of the phase angle.

    Returns:
        A 6x6 complex matrix M.
    """
    assert j in (COORD_T, COORD_S)
    if j == COORD_T:
        kz = get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL)
        u = t / kappa
    elif j == COORD_S:
        kz = t
        u = u0(kz, kz0, psi(zeta0, kz, cdivkL), psi0)

    if zeta0 < 0:
        # Unstable
        kK = kappa * kz * phi_inverse(kz, cdivkL)
        dKdz = kK * (1.0 / kz + 0.25 / (1.0 / cdivkL + kz))
    else:
        # Stable and neutral
        aux = phi(zeta0, kz, cdivkL)
        kK = kappa * kz / aux
        dKdz = kappa / aux**2

    kKcos = kK * cosbeta
    kKsin = kK * sinbeta

    if j == COORD_T:

        return np.array(
            [
                # M(1,2)=cmplx(-kK**2,kKcos*u)/kappa2
                # M(1,5)=cmplx(0.0E0,-kKcos/kappa)
                # M(1,6)=cmplx(0.0E0,-2.0E0*dKdz*kKcos/kappa/pscale)
                [
                    0,
                    complex(-(kK**2), kKcos * u) / kappa2,
                    0,
                    0,
                    complex(0, -kKcos / kappa),
                    complex(0, -2 * dKdz * kKcos / kappa),
                ],
                # M(2,1)=cmplx(-1.0E0,0.0E0)
                # M(2,6)=cmplx(0.0E0,-kKcos/pscale)
                [-1, 0, 0, 0, 0, complex(0, -kKcos)],
                # M(3,4)=M(1,2)
                # M(3,5)=cmplx(0.0E0,-kKsin/kappa)
                # M(3,6)=cmplx(0.0E0,-2.0E0*dKdz*kKsin/kappa/pscale)
                [
                    0,
                    0,
                    0,
                    complex(-(kK**2), kKcos * u) / kappa2,
                    complex(0, -kKsin / kappa),
                    complex(0, -2 * dKdz * kKsin / kappa),
                ],
                # M(4,3)=cmplx(-1.0E0,0.0E0)
                # M(4,6)=cmplx(0.0E0,-kKsin/pscale)
                [0, 0, -1, 0, 0, complex(0, -kKsin)],
                # M(5,2)=cmplx(-1.0E0,-dKdz*kKcos)/kappa2
                # M(5,4)=cmplx(0.0E0,-dKdz*kKsin/kappa2)
                # M(5,6)=cmplx(kK**2,-kKcos*u)/(pscale*kappa)
                [
                    0,
                    complex(-1, -dKdz * kKcos) / kappa2,
                    0,
                    complex(0, -dKdz * kKsin / kappa2),
                    0,
                    complex(kK**2, -kKcos * u) / (kappa),
                ],
                # M(6,2)=cmplx(0.0E0,kKcos/kappa2*pscale)
                # M(6,4)=cmplx(0.0E0,kKsin/kappa2*pscale)
                [0, complex(0, kKcos / kappa2), 0, complex(0, kKsin / kappa2), 0, 0],
            ]
        )

    else:  # j == COORD_S:
        return np.array(
            [
                # M(1,2)=dcmplx(-1.0D0,cosbeta*u/kK)
                # M(1,5)=dcmplx(0.0D0,-cosbeta)
                # M(1,6)=dcmplx(0.0D0,-2.0D0*dKdz*cosbeta/pscale)
                [0, complex(-1, cosbeta * u / kK), 0, 0, complex(0, -cosbeta), complex(0, -2 * dKdz * cosbeta)],
                # M(2,1)=-1.0D0
                # M(2,2)=dKdz/kK
                # M(2,6)=dcmplx(0.0D0,-kK*cosbeta/pscale)
                [-1, dKdz / kK, 0, 0, 0, complex(0, -kK * cosbeta)],
                # M(3,4)=dcmplx(-1.0D0,cosbeta*u/kK)
                # M(3,5)=dcmplx(0.0D0,-sinbeta)
                # M(3,6)=dcmplx(0.0D0,-2.0D0*dKdz*sinbeta/pscale)
                [0, 0, 0, complex(-1, cosbeta * u / kK), complex(0, -sinbeta), complex(0, -2 * dKdz * sinbeta)],
                # M(4,3)=-1.0D0
                # M(4,4)=M(2,2)
                # M(4,6)=dcmplx(0.0D0,-kK*sinbeta/pscale)
                [0, 0, -1, dKdz / kK, 0, complex(0, -kK * sinbeta)],
                # M(5,2)=dcmplx(-1.0D0/kK**2,dKdz/kK*cosbeta)
                # M(5,4)=dcmplx(0.0D0,-dKdz/kK*sinbeta)
                # M(5,6)=dcmplx(kK,-cosbeta*u)
                [
                    0,
                    complex(-1 / kK**2, dKdz / kK * cosbeta),
                    0,
                    complex(0, -dKdz / kK * sinbeta),
                    0,
                    complex(kK, -cosbeta * u),
                ],
                # M(6,2)=dcmplx(0.0D0,cosbeta*pscale/kK)
                # M(6,4)=dcmplx(0.0D0,sinbeta*pscale/kK)
                [0, complex(0, cosbeta / kK), 0, complex(0, sinbeta / kK), 0, 0],
            ]
        )


@jit(
    "complex128[:,:](int32,double,double, complex128[:,:],complex128[:,:], \
        double,double,double,double,double,double,double)"
)
def rk2_integration_step(j, t, h, Ay, y1, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta):
    r"""
    Performs a single integration step of the 2nd-order Runge-Kutta (RK2) method:
    Computes a "trial" step to the midpoint of the interval. Then uses both the values of vector Y and the matrix

    M (= -A^\dagger) at that midpoint to calculate the value of Y at the end of the interval.
    Refer to "Numerical Recipes" section 17.1 Runge-Kutta Method for more details.

    Args:
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        x: Current coordinate (s or t depending on j)
        h: Integration step size.
        my: Product of the matrix M and the vector Y.
        y: Current adjoint state vector.
        kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta:
            Parameters required to compute the matrix M via getM.

    Returns:
        The updated adjoint state vector after the RK2 intermediate step.
    """

    # Compute the matrix M at the midpoint
    m = getM(j, t + h * 0.5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute the vector Y at the midpoint using Euler's method ("trial" step)
    y_midpoint = y1 + h * Ay / 2

    # Compute the final value of Y using the midpoint values
    y2 = y1 + h * np.dot(np.ascontiguousarray(m), y_midpoint)
    return y2


B1 = 4.0 / 3.0
B2 = -1.0 / 3.0
C1 = 1.0 / 6.0
C2 = -1.0 / 6.0
C3 = 4.0 / 6.0
C4 = 2.0 / 6.0


@jit(
    "Tuple((complex128[:,:],complex128[:,:]))(complex128[:,:], double,double,int64,\
    double,double,double,double,double,double,double,\
    complex128[:],complex128[:],complex128[:],complex128[:],complex128[:],complex128[:])"
)
def modified_midpoint_integration_step(
    y,
    x,
    h,
    j,
    kz0,
    psi0,
    lastkz,
    zeta0,
    cdivkL,
    cosbeta,
    sinbeta,
    dbx_const,
    dby_const,
    dbz_const,
    dbx_lin,
    dby_lin,
    dbz_lin,
):
    """
    Perform a integration step using the Modified Midpoint method, as described in Numerical Recipes, §17.3.1, with two
    substeps (n=2). The estimate is fourth-order accurate, the same as the fourth-order Runge-Kutta method, but
    requires fewer function evaluations.

    Also updates the differential forcing accumulators based on the integrated state.

    Args:
        y: Current state matrix.
        x: Current state variable.
        h: Integration step size.
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        kz0: Base s-coordinate.
        psi0: Base stability correction.
        lastkz: Previous s-coordinate.
        zeta0: Stability parameter.
        cdivkL: Eddy diffusivity coefficient.
        cosbeta: Cosine of phase angle.
        sinbeta: Sine of phase angle.
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin:
            Differential forcing accumulators.

    Returns:
        A tuple where the first element is the updated state matrix (Yright)
        and the second element is an error estimate (yerr).
    """
    # Compute matrix M and the product MY at y1, the initial point
    m = getM(j, x, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    my = np.dot(np.ascontiguousarray(m), np.ascontiguousarray(y))

    # Compute an estimate of y at x + h using RK2
    y2 = rk2_integration_step(j, x, h, my, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute an estimate of y at x + h/2 using RK2
    y3 = rk2_integration_step(j, x, h * 0.5, my, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Compute matrix M and the product MY at y3, the midpoint
    m = getM(j, x + h * 0.5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    my = np.dot(np.ascontiguousarray(m), np.ascontiguousarray(y3))

    # Compute an estimate of y at x + h using RK2 from the midpoint
    y4 = rk2_integration_step(j, x + h * 0.5, h * 0.5, my, y3, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    # Combine the two estimates to get the final output and error estimate
    yright = B1 * y4 + B2 * y2  # eq. (17.3.4) in Numerical Recipes
    yerr = yright - y4

    if j == COORD_T:
        # kz1, kzm, kz2 = self.get_kz(x + np.array([0, .5, 1]) * h)

        kz1, kzm, kz2 = [get_kz(x + s * h, zeta0, kz0, lastkz, psi0, cdivkL) for s in [0.0, 0.5, 1.0]]

        if zeta0 < 0:
            # Unstable - the constant part of the forcing matrix F_i
            a1 = kz1 * phi_inverse(kz1, cdivkL)
            am = kzm * phi_inverse(kzm, cdivkL)
            a2 = kz2 * phi_inverse(kz2, cdivkL)
        else:
            # Stable and neutral - the constant part of the forcing matrix F_i
            a1 = 1 / (1 / kz1 + cdivkL)
            am = 1 / (1 / kzm + cdivkL)
            a2 = 1 / (1 / kz2 + cdivkL)

        # dbx_const=dbx_const+h*(conjg(a1*C1*y(2,:)+am*C3*y3(2,:)+a2*(C4*y4(2,:)+C2*y2(2,:))))
        # dby_const=dby_const+h*(conjg(a1*C1*y(4,:)+am*C3*y3(4,:)+a2*(C4*y4(4,:)+C2*y2(4,:))))
        # dbz_const=dbz_const+h*(conjg(a1*C1*y(6,:)+am*C3*y3(6,:)+a2*(C4*y4(6,:)+C2*y2(6,:))))*kappa
        # dbx_lin=dbx_lin+h*(conjg(kz1*a1*C1*y(2,:)+kzm*am*C3*y3(2,:)+kz2*a2*(C4*y4(2,:)+C2*y2(2,:))))
        # dby_lin=dby_lin+h*(conjg(kz1*a1*C1*y(4,:)+kzm*am*C3*y3(4,:)+kz2*a2*(C4*y4(4,:)+C2*y2(4,:))))
        # dbz_lin=dbz_lin+h*(conjg(kz1*a1*C1*y(6,:)+kzm*am*C3*y3(6,:)+kz2*a2*(C4*y4(6,:)+C2*y2(6,:))))*kappa

        # Update the differential forcing accumulators using Simpson's rule
        dbx_const += h * (np.conj(a1 * C1 * y[1, :] + am * C3 * y3[1, :] + a2 * (C4 * y4[1, :] + C2 * y2[1, :])))
        dby_const += h * (np.conj(a1 * C1 * y[3, :] + am * C3 * y3[3, :] + a2 * (C4 * y4[3, :] + C2 * y2[3, :])))
        dbz_const += (
            h * (np.conj(a1 * C1 * y[5, :] + am * C3 * y3[5, :] + a2 * (C4 * y4[5, :] + C2 * y2[5, :]))) * kappa
        )
        dbx_lin += h * (
            np.conj(kz1 * a1 * C1 * y[1, :] + kzm * am * C3 * y3[1, :] + kz2 * a2 * (C4 * y4[1, :] + C2 * y2[1, :]))
        )
        dby_lin += h * (
            np.conj(kz1 * a1 * C1 * y[3, :] + kzm * am * C3 * y3[3, :] + kz2 * a2 * (C4 * y4[3, :] + C2 * y2[3, :]))
        )
        dbz_lin += (
            h
            * (np.conj(kz1 * a1 * C1 * y[5, :] + kzm * am * C3 * y3[5, :] + kz2 * a2 * (C4 * y4[5, :] + C2 * y2[5, :])))
            * kappa
        )
    else:  # j == COORD_S

        xm = x + 0.5 * h
        x2 = x + h
        a1 = phi(zeta0, x, cdivkL)
        am = phi(zeta0, xm, cdivkL)
        a2 = phi(zeta0, x2, cdivkL)

        # Update the differential forcing accumulators using Simpson's rule
        dbx_const += h * np.conj(
            a1 * C1 * y[1, :] / x + am * C3 * y3[1, :] / xm + a2 * (C4 * y4[1, :] + C2 * y2[1, :]) / x2
        )
        dby_const += h * np.conj(
            a1 * C1 * y[3, :] / x + am * C3 * y3[3, :] / xm + a2 * (C4 * y4[3, :] + C2 * y2[3, :]) / x2
        )
        dbz_const += kappa * h * np.conj(C1 * y[5, :] + C3 * y3[5, :] + C4 * y4[5, :] + C2 * y2[5, :]) * kappa
        dbx_lin += h * np.conj((a1 * C1 * y[1, :] + am * C3 * y3[1, :]) + a2 * (C4 * y4[1, :] + C2 * y2[1, :]))
        dby_lin += h * np.conj((a1 * C1 * y[3, :] + am * C3 * y3[3, :]) + a2 * (C4 * y4[3, :] + C2 * y2[3, :]))
        dbz_lin += (
            kappa
            * h
            * np.conj((x * C1 * y[5, :] + xm * C3 * y3[5, :]) + x2 * (C4 * y4[5, :] + a2 * C2 * y2[5, :]))
            * kappa
        )
    return yright, yerr


c2 = "complex128[:,:],"
c1 = "complex128[:],"
d = "double,"
dyx = c1 * 6


@jit(f"""Tuple(({c2}{d}{d}{d}))({c2}{d}{d}{dyx}{d}{c2}{d}int32,{d}{d}{d}{d}{d}{d}{d})""")
def integrate_between_stations(
    Yleft,
    sleft,
    sright,
    dbx_const,
    dby_const,
    dbz_const,
    dbx_lin,
    dby_lin,
    dbz_lin,
    h,
    yerr,
    acc,
    j,
    kz0,
    lastkz,
    zeta0,
    cdivkL,
    psi0,
    cosbeta,
    sinbeta,
):
    """
    Integrates the adjoint system between two stations using the Modified Midpoint Method with adaptive step size
    control.

    Measures the error for each step, (d)yerr, and adjusts the following step size (h) accordingly to meet the accuracy
    requirement (acc).

    If it is the last step to reach the next station, the step size is adjusted to exactly reach that point.

    Measures the norm of the state vector after each step; if the norm exceeds a threshold, a substation is created.

    Args:
        Yleft: State matrix at the lower bound of the segment.
        sleft: s-coordinate of the lower bound of the current segment.
        sright: s-coordinate of the upper bound of the current segment.
        dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin:
            Forcing differential accumulators.
        h: Initial integration step size.
        yerr: Current accumulated error.
        acc: Desired accuracy level.
        j: Coordinate system indicator (COORD_T for t = u0 * kappa, COORD_S for s = kz).
        kz0: Base s-coordinate.
        lastkz: Previous s-coordinate.
        zeta0: Stability parameter (z0/L).
        cdivkL: Eddy diffusivity coefficient.
        psi0: Stability correction parameter.
        cosbeta: Cosine of phase angle.
        sinbeta: Sine of phase angle.

    Returns:
        The updated state matrix after integration and the new step size.
    """
    kz1 = kz0 * np.exp(sleft)
    kz2 = kz0 * np.exp(sright)

    if j == COORD_T:
        t1 = kappa * u0(kz1, kz0, psi(zeta0, kz1, cdivkL), psi0)
        t2 = kappa * u0(kz2, kz0, psi(zeta0, kz2, cdivkL), psi0)
    else:
        t1 = kz1
        t2 = kz2

    t = t1

    Ynorm1 = np.linalg.norm(Yleft[:, 0])
    Y1 = Yleft
    norm_lst = []
    # print(f'{p.sleft:.3f}, {h:0.5f}')
    # if p.sleft >= 18.15:
    #     print()
    counter = 0
    while True:
        counter += 1
        if t + 1.1 * h > t2:
            # step, h, big enough to reach t2 -> take the final step
            h = t2 - t
            # dyerr = self.rk2(p, Y1, t, h, j)
            Yright, dyerr = modified_midpoint_integration_step(
                Y1,
                t,
                h,
                j,
                kz0,
                psi0,
                lastkz,
                zeta0,
                cdivkL,
                cosbeta,
                sinbeta,
                dbx_const,
                dby_const,
                dbz_const,
                dbx_lin,
                dby_lin,
                dbz_lin,
            )

            yerr += dyerr
            h = get_new_h2(h, acc, dyerr, Yright)

            # if p.sleft > 18:
            #     import matplotlib.pyplot as plt
            #     plt.title(p.sleft)
            #     plt.plot(norm_lst)
            #     plt.show()
            # print(kz0, sleft, counter)
            return Yright, h, sright, lastkz
        else:
            # take step, h
            # dyerr = self.rk2(p, Y1, t, h, j)  # here Yright is updated, only Yright and deltaBs
            Yright, dyerr = modified_midpoint_integration_step(
                Y1,
                t,
                h,
                j,
                kz0,
                psi0,
                lastkz,
                zeta0,
                cdivkL,
                cosbeta,
                sinbeta,
                dbx_const,
                dby_const,
                dbz_const,
                dbx_lin,
                dby_lin,
                dbz_lin,
            )

            yerr += dyerr
            Y1 = Yright.copy()
            t = t + h
            h = get_new_h2(h, acc, dyerr, Yright)
            Ynorm = np.linalg.norm(Yright[:, 0]) / Ynorm1
            norm_lst.append(Ynorm)
            # If Ynorm is too large, then we make a "sublevel" or "substation"
            # One can have multiple sublevels between two levels if necessary.
            # print(t, Ynorm)
            if Ynorm > Ythreshold:
                if j == COORD_T:  # pragma: no cover
                    kz1 = get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL)
                    lastkz = kz1
                    s1 = np.log(kz1 / kz0)
                else:
                    # This might be a mistake according to sqot, but it is used.
                    s1 = np.log(t / kz0)

                return Yright, h, s1, lastkz

                #       Y1=p%dat%Yleft
        #       Ynorm1=norm(Y1(:,1))
