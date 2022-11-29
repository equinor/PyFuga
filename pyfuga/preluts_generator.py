from tqdm import tqdm

import numpy as np
from pyfuga.constants import max_recs, kappa, Ythreshold, kappa2, Cm1, Cm2, n_eq
from pyfuga.common import cdivkL, psi, get_new_h2, dphiu, phi
from pyfuga.utils import jit
from pyfuga.preluts import PreLUT


# np.set_printoptions(precision=2, linewidth=200)
# left: (sub)station (lower height)
# right: (sub)station + 1 (higher height)
# (Y+).x = b is a set of boundary condition equations.
# R is a Gram-Smidt (or QR) transformation of Y
# dyxu0 = Delta_b for constant longitudinal forcing
# dyxu1 = Delta_b for longitudinal forcing proportional to kz
# Additional if transversal and/or vertical forcing is present:
#  dyxv0 = Delta_b for constant transversal forcing
#  dyxv1 = Delta_b for transversal forcing proportional to kz
#  dyxw0 = Delta_b for constant vertical forcing
#  dyxw1 = Delta_b for vertical forcing proportional to kz
# Delta_b = integral of (Y+).f between two  (sub)stations
# where f is the wind turbine forcing.


class PrelutNode():
    def __init__(self):
        self.reset_dyx()

    def set_s(self, sleft, sright):
        self.sleft = sleft
        self.sright = sright

    def reset_dyx(self):
        self.dyxu0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxu1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw1 = np.zeros(n_eq, dtype=np.complex128)

    def get_next(self, sleft, sright):
        next_node = self.GMRES()
        next_node.set_s(sleft, sright)
        return next_node

    def GMRES(self):
        # Modified Gram-Schmidt ortonormalization
        # Yright, Yleft, Rleft and Rright are nxn matrices
        # Columns of Yright are linearly independent vectors (the input)
        # Columns of Yleft form an orthonormal basis (Yleft is unitary)
        # Rleft and Rright are lower triangular
        # Rright is the inverse of Rleft
        # Yright=Yleft Rleft.T*  where Rleft.T* is the conjugate transpose of Rleft

        node = PrelutNode()
        Rleft, Yleft, Rright = gmres(self.Yright)
        node.Rleft = Rleft
        node.Yleft = Yleft
        self.Rright = Rright
        return node


@jit('Tuple((complex128[:,:],complex128[:,:],complex128[:,:]))(complex128[:,:])')
def gmres(Yright):
    Yleft, Rleft = np.linalg.qr(Yright)
    Rleft = np.conj(Rleft.T)
    Rright = np.linalg.inv(Rleft)
    return Rleft, Yleft, Rright


class PrelutNodeFirst(PrelutNode):
    def __init__(self, beta, ds):
        PrelutNode.__init__(self)
        self.set_s(sleft=0, sright=ds)
        sinbeta, cosbeta = np.sin(beta), np.cos(beta)
        self.Yleft = np.array([[-sinbeta, 0, cosbeta, 0, 0, 0],
                               [cosbeta, 0, sinbeta, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, -sinbeta, 0, cosbeta, 0, 0],
                               [0, cosbeta, 0, sinbeta, 0, 0],
                               [0, 0, 0, 0, 0, 1]], dtype=np.complex128).T
        self.Rleft = np.eye(6, dtype=np.complex128)


class PreLUTGenerator():
    # Internt benyttes variablene:
    # u,v,w,p og t=u/kappa for kz<kzm
    # u,v,w,p og kz        for kz>kzm
    # kzm=phi(zm/L)
    # preluts formuleres i termer af  u,v,w,p og s=log(z/z0).
    # Stationer ved s=j*ds samt ved s=Log(kzm/kz0) og evt. extra stationer.
    # Level incrementeres ved stationer med s=j*ds
    # Der orthonormaliseres ved hver station og gemmes i en hægtet liste.
    # Bruger anden ordens R-K
    def __init__(self, zeta0, kz0, beta, kzmax, ds, accgoal):

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
                if (abs(dx / x) < 1.0e-14):
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
        self.nodes = []
        first = PrelutNodeFirst(self.beta, self.ds)
        h = np.sqrt(self.acc * 6 / 3.125)
        self.lastkz = self.kz0

        yerr = self.rk2(first, first.Yleft, 0., h, j=1)
        first.reset_dyx()
        h = get_new_h2(h, self.acc, yerr, first.Yright)
        sm = self.sm()

        # cumsum gives slightly different results than arange (more equal to fortran implementation)
        s_lst = np.sort(np.r_[0, np.cumsum(np.full(int(self.smaxx // self.ds) + 1, self.ds)), sm])

        # equal(first.Yleft, f'yleft{0:6.3f}')
        segment, h = self.solve2(first, h, yerr, self.acc, 1)
        # equal(segment.Yright, f'yright{0:6.3f}')
        for (s1, s2) in tqdm(list(zip(s_lst[1:], s_lst[2:])), disable=1):
            self.nodes.append(segment)
            segment = segment.get_next(s1, s2)
            j = 1 + (s1 >= sm)

            segment, h = self.solve2(segment, h, yerr, self.acc, j)
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
        var_names = ['Yleft', 'Rleft', 'Rright',
                     'dyxu0', 'dyxu1', 'dyxv0', 'dyxv1', 'dyxw0', 'dyxw1',
                     'sleft', 'sright']
        var_values = ([np.moveaxis([getattr(n, k) for n in self.nodes], 1, 2) for k in var_names[:3]] +
                      [np.array([getattr(n, k) for n in self.nodes]) for k in var_names[3:]])
        return PreLUT({**{n: (('i', 'j', 'k')[:len(v.shape)], v)
                          for n, v in zip(var_names, var_values)},
                       'beta': self.beta, 'kz0': self.kz0,
                       **{'level': (('i',), np.round(var_values[-2] / self.ds, 3).astype(int))}},
                      attrs={'ds': self.ds, 'kzmax': self.kzmax, 'zeta0': self.zeta0, 'accgoal': self.accgoal})

    def rk2(self, node, y, x, h, j):
        Yright, yerr = rk2(
            y, x, h, j, self.kz0, self.psi0, self.lastkz, self.zeta0, self.cdivkL, self.cosbeta, self.sinbeta,
            node.dyxu0, node.dyxv0, node.dyxw0, node.dyxu1, node.dyxv1, node.dyxw1)
        node.Yright = Yright
        return yerr

    def solve2(self, p, h, yerr, acc, j):
        # return self.solve2_old(p, h, yerr, acc, j)
        while True:
            sright = p.sright
            Yright, h, s2, lastkz = solve2(p.Yleft, p.sleft, sright, p.dyxu0, p.dyxv0,
                                           p.dyxw0, p.dyxu1, p.dyxv1, p.dyxw1, h, yerr, acc, j,
                                           self.kz0, self.lastkz, self.zeta0, self.cdivkL, self.psi0,
                                           self.cosbeta, self.sinbeta)
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
            aux = dphiu(lastkz, cdivkL)
            x = (cdivkL * lastkz) / ((aux**2 + 1) * (1 + aux)**2)
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1)**2
        while abs(dx / x) > 1e-14:
            # print(dx)
            dx = -(2 * np.arctan(x) + np.log(x) - b) * x * (1 + x**2) / (x + 1)**2
            x = x + dx
            if x < 0:  # pragma: no cover
                x = np.exp(b)
                dx = x
        kz = 8 * x * (1 + x**2) / (cdivkL * (1 - x)**4)
    return kz


@jit('double(double,double,double,double)')
def u0(kz, kz0, psi, psi0):
    # Wind speed from MOST normalized by uStar
    #  use params, only: mykind,kz0,kappa,psi0
    #  implicit none
    #  real(mykind) u0,kz,psi
    return (np.log(kz / kz0) + psi - psi0) / kappa


@jit("complex128[:,:](int32,double,double, double, double,double,double,double,double)")
def getM(j, t, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta):
    # ! This version includes stability in the eddy diffusivity.
    # ! Parameter zeta0=z0/L
    # ! In reality it returns -A* , minus the adjoint of A
    # ! s=kappa*u for j=1
    # ! s=k z for j=2
    # ! variables: u,u',v,v',w and p/pscale (for both j=1 and j=2).
    # ! To change back to p=q*s uncomment lines 1261-1265 in prelut_machine)
    # ! K = k u* z/phi
    # ! Only case(1) and case(2) are called
    # ! Only simple closure
    # use params, only: mykind,n,cdivkL,cosbeta,sinbeta,kappa,kappa2,lastkz, &
    #                   pscale,zeta0,lidt
    assert j in [1, 2]
    if j == 1:
        kz = get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL)
        u = t / kappa
    elif j == 2:
        kz = t
        u = u0(kz, kz0, psi(zeta0, kz, cdivkL), psi0)

    dphiu_ = dphiu(kz, cdivkL)
    if zeta0 < 0:
        # Unstable
        kK = kappa * kz * dphiu_
        dKdz = kK * (1.0 / kz + 0.25 / (1.0 / cdivkL + kz))
    else:
        # Stable and neutral
        aux = dphiu_
        kK = kappa * kz / aux
        dKdz = kappa / aux**2

    kKcos = kK * cosbeta
    kKsin = kK * sinbeta

    if j == 1:

        return np.array([
            # M(1,2)=cmplx(-kK**2,kKcos*u)/kappa2
            # M(1,5)=cmplx(0.0E0,-kKcos/kappa)
            # M(1,6)=cmplx(0.0E0,-2.0E0*dKdz*kKcos/kappa/pscale)
            [0, complex(-kK**2, kKcos * u) / kappa2, 0, 0,
             complex(0, -kKcos / kappa),
             complex(0, -2 * dKdz * kKcos / kappa)],
            # M(2,1)=cmplx(-1.0E0,0.0E0)
            # M(2,6)=cmplx(0.0E0,-kKcos/pscale)
            [-1, 0, 0, 0, 0,
             complex(0, -kKcos)],
            # M(3,4)=M(1,2)
            # M(3,5)=cmplx(0.0E0,-kKsin/kappa)
            # M(3,6)=cmplx(0.0E0,-2.0E0*dKdz*kKsin/kappa/pscale)
            [0, 0, 0, complex(-kK**2, kKcos * u) / kappa2,
             complex(0, - kKsin / kappa),
             complex(0, - 2 * dKdz * kKsin / kappa)],
            # M(4,3)=cmplx(-1.0E0,0.0E0)
            # M(4,6)=cmplx(0.0E0,-kKsin/pscale)
            [0, 0, -1, 0, 0,
             complex(0, -kKsin)],
            # M(5,2)=cmplx(-1.0E0,-dKdz*kKcos)/kappa2
            # M(5,4)=cmplx(0.0E0,-dKdz*kKsin/kappa2)
            # M(5,6)=cmplx(kK**2,-kKcos*u)/(pscale*kappa)
            [0, complex(-1, - dKdz * kKcos) / kappa2, 0,
             complex(0, -dKdz * kKsin / kappa2), 0,
             complex(kK**2, - kKcos * u) / (kappa)],
            # M(6,2)=cmplx(0.0E0,kKcos/kappa2*pscale)
            # M(6,4)=cmplx(0.0E0,kKsin/kappa2*pscale)
            [0, complex(0, kKcos / kappa2), 0,
             complex(0, kKsin / kappa2), 0, 0]])

    else:  # j == 2:
        return np.array([
            # M(1,2)=dcmplx(-1.0D0,cosbeta*u/kK)
            # M(1,5)=dcmplx(0.0D0,-cosbeta)
            # M(1,6)=dcmplx(0.0D0,-2.0D0*dKdz*cosbeta/pscale)
            [0, complex(-1, cosbeta * u / kK), 0, 0,
             complex(0, -cosbeta),
             complex(0, -2 * dKdz * cosbeta)],

            # M(2,1)=-1.0D0
            # M(2,2)=dKdz/kK
            # M(2,6)=dcmplx(0.0D0,-kK*cosbeta/pscale)
            [-1, dKdz / kK, 0, 0, 0,
             complex(0, -kK * cosbeta)],

            # M(3,4)=dcmplx(-1.0D0,cosbeta*u/kK)
            # M(3,5)=dcmplx(0.0D0,-sinbeta)
            # M(3,6)=dcmplx(0.0D0,-2.0D0*dKdz*sinbeta/pscale)
            [0, 0, 0, complex(-1, cosbeta * u / kK),
             complex(0, - sinbeta),
             complex(0, - 2 * dKdz * sinbeta)],
            # M(4,3)=-1.0D0
            # M(4,4)=M(2,2)
            # M(4,6)=dcmplx(0.0D0,-kK*sinbeta/pscale)
            [0, 0, -1, dKdz / kK, 0,
             complex(0, -kK * sinbeta)],

            # M(5,2)=dcmplx(-1.0D0/kK**2,dKdz/kK*cosbeta)
            # M(5,4)=dcmplx(0.0D0,-dKdz/kK*sinbeta)
            # M(5,6)=dcmplx(kK,-cosbeta*u)
            [0, complex(-1 / kK**2, dKdz / kK * cosbeta), 0,
             complex(0, -dKdz / kK * sinbeta), 0,
             complex(kK, - cosbeta * u)],
            # M(6,2)=dcmplx(0.0D0,cosbeta*pscale/kK)
            # M(6,4)=dcmplx(0.0D0,sinbeta*pscale/kK)
            [0, complex(0, cosbeta / kK), 0,
             complex(0, sinbeta / kK), 0, 0]])


@jit('complex128[:,:](int32,double,double, complex128[:,:],complex128[:,:],double,double,double,double,double,double,double)')
def rk2step(j, t, h, Ay, y1, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta):
    A = getM(j, t + h * 0.5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    ym = y1 + h * Ay / 2
    y2 = y1 + h * np.dot(np.ascontiguousarray(A), ym)
    return y2


B1 = 4.0 / 3.0
B2 = -1.0 / 3.0
C1 = 1.0 / 6.0
C2 = -1.0 / 6.0
C3 = 4.0 / 6.0
C4 = 2.0 / 6.0


@jit('Tuple((complex128[:,:],complex128[:,:]))(complex128[:,:], double,double,int64,double,double,double,double,double,double,double,complex128[:],complex128[:],complex128[:],complex128[:],complex128[:],complex128[:])')
def rk2(y, x, h, j, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta,
        dyxu0, dyxv0, dyxw0, dyxu1, dyxv1, dyxw1
        ):

    A = getM(j, x, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    Ay = np.dot(np.ascontiguousarray(A), np.ascontiguousarray(y))

    y2 = rk2step(j, x, h, Ay, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    y3 = rk2step(j, x, h * .5, Ay, y, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    A = getM(j, x + h * .5, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)
    Ay = np.dot(np.ascontiguousarray(A), np.ascontiguousarray(y3))

    y4 = rk2step(j, x + h * .5, h * .5, Ay, y3, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta)

    yout = B1 * y4 + B2 * y2
    yerr = yout - y4
    Yright = yout

    if j == 1:
        # kz1, kzm, kz2 = self.get_kz(x + np.array([0, .5, 1]) * h)

        kz1, kzm, kz2 = [get_kz(x + s * h, zeta0, kz0, lastkz, psi0, cdivkL) for s in [0., .5, 1.]]

        if zeta0 < 0:
            # Unstable
            a1 = kz1 * dphiu(kz1, cdivkL)
            am = kzm * dphiu(kzm, cdivkL)
            a2 = kz2 * dphiu(kz2, cdivkL)
        else:
            # Stable and neutral
            a1 = 1 / (1 / kz1 + cdivkL)
            am = 1 / (1 / kzm + cdivkL)
            a2 = 1 / (1 / kz2 + cdivkL)
        # dyxu0=dyxu0+h*(conjg(a1*C1*y(2,:)+am*C3*y3(2,:)+a2*(C4*y4(2,:)+C2*y2(2,:))))
        # dyxv0=dyxv0+h*(conjg(a1*C1*y(4,:)+am*C3*y3(4,:)+a2*(C4*y4(4,:)+C2*y2(4,:))))
        # dyxw0=dyxw0+h*(conjg(a1*C1*y(6,:)+am*C3*y3(6,:)+a2*(C4*y4(6,:)+C2*y2(6,:))))*kappa
        # dyxu1=dyxu1+h*(conjg(kz1*a1*C1*y(2,:)+kzm*am*C3*y3(2,:)+kz2*a2*(C4*y4(2,:)+C2*y2(2,:))))
        # dyxv1=dyxv1+h*(conjg(kz1*a1*C1*y(4,:)+kzm*am*C3*y3(4,:)+kz2*a2*(C4*y4(4,:)+C2*y2(4,:))))
        # dyxw1=dyxw1+h*(conjg(kz1*a1*C1*y(6,:)+kzm*am*C3*y3(6,:)+kz2*a2*(C4*y4(6,:)+C2*y2(6,:))))*kappa
        dyxu0 += h * (np.conj(a1 * C1 * y[1, :] + am * C3 * y3[1, :] + a2 * (C4 * y4[1, :] + C2 * y2[1, :])))
        dyxv0 += h * (np.conj(a1 * C1 * y[3, :] + am * C3 * y3[3, :] + a2 * (C4 * y4[3, :] + C2 * y2[3, :])))
        dyxw0 += h * (np.conj(a1 * C1 * y[5, :] + am * C3 * y3[5, :] +
                              a2 * (C4 * y4[5, :] + C2 * y2[5, :]))) * kappa
        dyxu1 += h * (np.conj(kz1 * a1 * C1 * y[1, :] + kzm * am *
                              C3 * y3[1, :] + kz2 * a2 * (C4 * y4[1, :] + C2 * y2[1, :])))
        dyxv1 += h * (np.conj(kz1 * a1 * C1 * y[3, :] + kzm * am *
                              C3 * y3[3, :] + kz2 * a2 * (C4 * y4[3, :] + C2 * y2[3, :])))
        dyxw1 += h * (np.conj(kz1 * a1 * C1 * y[5, :] + kzm * am * C3 *
                              y3[5, :] + kz2 * a2 * (C4 * y4[5, :] + C2 * y2[5, :]))) * kappa
    else:

        xm = x + 0.5 * h
        x2 = x + h
        a1 = phi(zeta0, x, cdivkL)
        am = phi(zeta0, xm, cdivkL)
        a2 = phi(zeta0, x2, cdivkL)
        dyxu0 += h * \
            np.conj(a1 * C1 * y[1, :] / x + am * C3 * y3[1, :] / xm + a2 * (C4 * y4[1, :] + C2 * y2[1, :]) / x2)
        dyxv0 += h * \
            np.conj(a1 * C1 * y[3, :] / x + am * C3 * y3[3, :] / xm + a2 * (C4 * y4[3, :] + C2 * y2[3, :]) / x2)
        dyxw0 += kappa * h * \
            np.conj(C1 * y[5, :] + C3 * y3[5, :] + C4 * y4[5, :] + C2 * y2[5, :]) * kappa
        dyxu1 += h * \
            np.conj((a1 * C1 * y[1, :] + am * C3 * y3[1, :]) + a2 * (C4 * y4[1, :] + C2 * y2[1, :]))
        dyxv1 += h * \
            np.conj((a1 * C1 * y[3, :] + am * C3 * y3[3, :]) + a2 * (C4 * y4[3, :] + C2 * y2[3, :]))
        dyxw1 += kappa * h * \
            np.conj((x * C1 * y[5, :] + xm * C3 * y3[5, :]) + x2 * (C4 * y4[5, :] + a2 * C2 * y2[5, :])) * kappa
    return Yright, yerr


c2 = 'complex128[:,:],'
c1 = 'complex128[:],'
d = 'double,'
dyx = c1 * 6


# @jit(f'''Tuple(({c2}{d}{d}{d}))({c2}{d}{d}{dyx}{d}{c2}{d}int32,{d}{d}{d}{d}{d}{d}{d})''')
def solve2(Yleft, sleft, sright, dyxu0, dyxv0, dyxw0, dyxu1, dyxv1, dyxw1, h, yerr, acc, j,
           kz0, lastkz, zeta0, cdivkL, psi0, cosbeta, sinbeta):
    # use params, only: mykind,n,kz0,Ythreshold,kappa,reccount
    # use contr, only: Tprelutnode
    # use GMRES_interface
    # implicit none
    # type(Tprelutnode), pointer :: p
    # integer(4) j
    # real(mykind), intent(inout) :: h
    # real(mykind), intent(in) :: acc
    # complex(mykind), dimension(n,n), intent(out) :: Yerr
    # complex(mykind), dimension(n,n) :: Y1,Y2,dYerr,A
    # real(mykind) t,t1,t2,hh,err,Ynorm,Ynorm1,kz1,kz2,s1,s2
    # integer(4) i
    # real(mykind) u0,norm
    kz1 = kz0 * np.exp(sleft)
    kz2 = kz0 * np.exp(sright)

    if (j == 1):
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
        if (1.1 * h + t > t2):
            # step, h, big enough to reach t2 -> take the final step
            h = t2 - t
            # dyerr = self.rk2(p, Y1, t, h, j)
            Yright, dyerr = rk2(Y1, t, h, j, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta,
                                dyxu0, dyxv0, dyxw0, dyxu1, dyxv1, dyxw1)

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
            Yright, dyerr = rk2(Y1, t, h, j, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta,
                                dyxu0, dyxv0, dyxw0, dyxu1, dyxv1, dyxw1)

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
                if j == 1:  # pragma: no cover
                    kz1 = get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL)
                    lastkz = kz1
                    s1 = np.log(kz1 / kz0)
                else:
                    # This might be a mistake according to sqot, but it is used.
                    s1 = np.log(t / kz0)

                return Yright, h, s1, lastkz

                #       Y1=p%dat%Yleft
        #       Ynorm1=norm(Y1(:,1))
