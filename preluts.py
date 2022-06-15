import numpy as np
from pathlib import Path
from PyPreludium.utils import get_beta, psi, phi, dphiu, get_new_h2, save_complex, compare, GMRES, rel_err
from PyPreludium.constants import Cm1, n_eq, kappa, kappa2, pscale, max_recs, Ythreshold
from numpy import newaxis as na
from tqdm import tqdm
from py_wake.utils.profiling import timeit
import struct
import xarray as xr
from PyPreludium.tests.test_files import tfp
#np.set_printoptions(precision=2, linewidth=200)


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
        self.dyxu0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxu1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw1 = np.zeros(n_eq, dtype=np.complex128)


class PrelutNodeFirst(PrelutNode):
    def __init__(self, beta):
        PrelutNode.__init__(self)
        sinbeta, cosbeta = np.sin(beta), np.cos(beta)
        self.Yleft = np.array([[-sinbeta, 0, cosbeta, 0, 0, 0],
                               [cosbeta, 0, sinbeta, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, -sinbeta, 0, cosbeta, 0, 0],
                               [0, cosbeta, 0, sinbeta, 0, 0],
                               [0, 0, 0, 0, 0, 1]], dtype=np.complex128).T
        self.Rleft = np.eye(6, dtype=np.complex128)


class PrelutNodeNext(PrelutNode):
    def __init__(self, previuos_node):
        PrelutNode.__init__(self)
        self.GMRES(previuos_node)

    def GMRES(self, prev):
        # use params, only: mykind,n
        # use contr, only: Tprelutnode
        # use vector_functions, only: outer
        # implicit none

        # Modified Gram-Schmidt ortonormalization
        # Y, V, R and invR are nxn matrices
        # Columns of Y are linearly independent vectors (the input)
        # Columns of V form an orthonormal basis (V is unitary)
        # R and invR are lower triangular
        # invR is the inverse of R
        # Y=V R*  where R* is the conjugate transpose of R

        # type(Tprelutnode),pointer :: p
        # complex(mykind), dimension(n,n) :: B
        # real(mykind) aux
        # integer(4) i,j,k
        # real(mykind) norm

        # aux = np.linalg.norm(prev.Yright, axis=0)
        # next.dat.Yleft = Yleft = prev.Yright / aux
        # next.dat.Rleft = np.diag(aux)
        # for j in range(5):
        #     next.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j]) @ Yleft[:, j + 1:]
        #     Yleft[:, j + 1:] = Yleft[:, j + 1:] - (Yleft[:, j] * np.conj(Yleft[:, j])[:, na]) @ Yleft[:, j + 1:]

        Yleft = prev.Yright.copy()
        self.Rleft = np.zeros_like(Yleft)
        for j in range(5):
            aux = np.linalg.norm(Yleft[:, j])
            Yleft[:, j] = Yleft[:, j] / aux
            self.Rleft[j, j] = aux
            self.Rleft[j, j + 1:] = np.conj(Yleft[:, j])@Yleft[:, j + 1:]
            Yleft[:, j + 1:] = Yleft[:, j + 1:] - \
                np.dot((Yleft[:, j] * np.conj(Yleft[:, j])[:, na]).T, Yleft[:, j + 1:])

        aux = np.linalg.norm(Yleft[:, -1])
        Yleft[:, -1] = Yleft[:, -1] / aux
        self.Rleft[-1, -1] = aux

        # B = np.zeros_like(Yleft)
        # for j in range(1, 6):
        #     B[:j, j] = self.dat.Rleft[:j, j] / self.dat.Rleft[j, j]
        B = np.triu(self.Rleft / np.diag(self.Rleft), 1)  # upper triangle without diagonal
        # prev.dat.Rright = np.diag(1 / np.diag(self.dat.Rleft))
        # for i in range(6):
        #     for j in range(i + 1, 6):
        #         for k in range(i, j):
        #             prev.dat.Rright[i, j] = prev.dat.Rright[i, j] - prev.dat.Rright[i, k] * B[k, j]

        prev.Rright = np.diag(1 / np.diag(self.Rleft))
        for i in range(6):
            for j in range(i + 1, 6):
                prev.Rright[i, j] -= np.sum(prev.Rright[i, i:j] * B[i:j, j])

        self.Rleft = np.conj(self.Rleft.T)
        prev.Rright = np.conj(prev.Rright.T)
        self.Yleft = Yleft

        # rel_err(ref.sel(i=prev.level + 1).Yleft.T.values, self.Yleft)


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
        self.nodes = [PrelutNodeFirst(beta)]

        self.zeta0 = zeta0
        self.kz0 = kz0
        self.beta = beta
        self.ds = ds
        self.cosbeta = np.cos(beta)
        self.sinbeta = np.sin(beta)
        self.accgoal = accgoal

        self.smaxx = np.log(np.minimum(kzmax / kz0, 1e8))
        self.acc = accgoal / self.smaxx

    def make_prelut(self):
        first = self.nodes[0]
        h = np.sqrt(self.acc * 6 / 3.125)
        self.lastkz = self.kz0

        yerr = self.rk2(first, first.Yleft, 0, h, j=1)
        first.dyxu0 = 0
        first.dyxv0 = 0
        first.dyxw0 = 0
        first.dyxu1 = 0
        first.dyxv1 = 0
        first.dyxw1 = 0
        h = get_new_h2(h, self.acc, yerr, first.Yright)
        sm = np.minimum(self.smaxx, self.smaxx)
        first.sleft = 0
        first.sright = self.ds
        s1 = 0.0
        s2 = self.ds
        s_lst = np.r_[np.arange(0, sm, self.ds), sm, np.arange(sm, self.smaxx)]

        for i, (s1, s2) in enumerate(zip(s_lst, s_lst[1:][:max_recs])):
            segment = self.nodes[-1]
            segment.level = i

            segment.sleft = s1
            segment.sright = s2
            h = self.solve2(segment, h, yerr, self.acc, 1)
            if s2 < self.smaxx:

                self.nodes.append(PrelutNodeNext(segment))

            # allocate(segment%next)
            # segment%next%prev=>segment
            # call GMRES(segment)
            # segment=>segment%next

        PrelutNodeNext(self.nodes[-1])
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

        var_names = ['Yleft', 'Rleft', 'Rright',
                     'dyxu0', 'dyxu1', 'dyxv0', 'dyxv1', 'dyxw0', 'dyxw1',
                     'sleft', 'sright', 'level']
        var_values = ([np.moveaxis([getattr(n, k) for n in self.nodes], 1, 2) for k in var_names[:3]] +
                      [np.array([getattr(n, k) for n in self.nodes]) for k in var_names[3:]])
        return xr.Dataset({**{n: (('i', 'j', 'k')[:len(v.shape)], v)
                              for n, v in zip(var_names, var_values)},
                           'zeta0': self.zeta0, 'beta': self.beta, 'kz0': self.kz0},
                          attrs={'ds': 0.05, 'accgoal': self.accgoal, })

    def rk2(self, node, y, x, h, j):

        B1 = 4.0 / 3.0
        B2 = -1.0 / 3.0
        c1 = 1.0 / 6.0
        c2 = -1.0 / 6.0
        c3 = 4.0 / 6.0
        c4 = 2.0 / 6.0

        def rk2step(t1, h, Ay, y1, j):
            #compare(Ay, 'rk2step_Ay')
            #compare(y1, 'rk2step_y1')
            ym = y1 + h * Ay / 2
            #compare(ym, 'rk2step_ym')
            A = self.getM(t1 + h * 0.5, j)
            #compare(A, 'rk2step_A')
            compare(np.dot(A, ym), 'rk2step_Aym')
            y2 = y1 + h * np.dot(A, ym)
            #compare(y2, 'rk2step_y2')
            return y2

        A = self.getM(x, j)
        # compare(A, 'rk2_A')

        Ay = A @ y
        # compare(Ay, 'rk2_Ay')

        y2 = rk2step(x, h, Ay, y, j)
        # compare(y2, 'rk2_y2')

        y3 = rk2step(x, h * 0.5, Ay, y, j)
        # compare(y3, 'rk2_y3')

        A = self.getM(x + h * 0.5, j)
        # compare(A, 'rk2_A2')

        Ay = A @y3
        # compare(Ay, 'rk2_Ay2')

        y4 = rk2step(x + h * 0.5, h * 0.5, Ay, y3, j)
        compare(y4, 'rk2_y4')

        yout = B1 * y4 + B2 * y2
        # compare(yout, 'rk2_yout')
        yerr = yout - y4
        node.Yright = yout

        if j == 1:
            kz1 = self.lastkz
            kz1 = self.get_kz(x)
            t = (x + h * 0.5)
            kzm = kz1
            kzm = self.get_kz(t)
            t = (x + h)
            kz2 = kz1
            kz2 = self.get_kz(t)
            if self.zeta0 < 0:
                # Unstable
                a1 = kz1 * dphiu(kz1, self.cdivkL)
                am = kzm * dphiu(kzm, self.cdivkL)
                a2 = kz2 * dphiu(kz2, self.cdivkL)
            else:
                # Stable and neutral
                a1 = 1 / (1 / kz1 + self.cdivkL)
                am = 1 / (1 / kzm + self.cdivkL)
                a2 = 1 / (1 / kz2 + self.cdivkL)
            node.dyxu0 += h * (np.conj(a1 * c1 * y[1, :] + am * c3 * y3[1, :] + a2 * (c4 * y4[1, :] + c2 * y2[1, :])))
            node.dyxv0 += h * (np.conj(a1 * c1 * y[3, :] + am * c3 * y3[3, :] + a2 * (c4 * y4[3, :] + c2 * y2[3, :])))
            node.dyxw0 += h * (np.conj(a1 * c1 * y[5, :] + am * c3 * y3[5, :] +
                                       a2 * (c4 * y4[5, :] + c2 * y2[5, :]))) * kappa
            node.dyxu1 += h * (np.conj(kz1 * a1 * c1 * y[2, :] + kzm * am *
                                       c3 * y3[1, :] + kz2 * a2 * (c4 * y4[1, :] + c2 * y2[1, :])))
            node.dyxv1 += h * (np.conj(kz1 * a1 * c1 * y[3, :] + kzm * am *
                                       c3 * y3[3, :] + kz2 * a2 * (c4 * y4[3, :] + c2 * y2[3, :])))
            node.dyxw1 += h * (np.conj(kz1 * a1 * c1 * y[5, :] + kzm * am * c3 *
                                       y3[5, :] + kz2 * a2 * (c4 * y4[5, :] + c2 * y2[5, :]))) * kappa
        else:
            raise NotImplementedError()
        return yerr

    def solve2(self, p, h, yerr, acc, j):
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
        kz1 = self.kz0 * np.exp(p.sleft)
        kz2 = self.kz0 * np.exp(p.sright)

        if (j == 1):
            t1 = kappa * self.u0(kz1)
            t2 = kappa * self.u0(kz2)
        else:
            t1 = kz1
            t2 = kz2

        t = t1

        Ynorm1 = np.linalg.norm(p.Yleft[:, 1])
        Y1 = p.Yleft
        while True:
            if (1.1 * h + t > t2):
                h = t2 - t
                dyerr = self.rk2(p, Y1, t, h, j)
                yerr = yerr + dyerr
                #     p%Yright=Y2
                h = get_new_h2(h, self.acc, dyerr, p.Yright)
                #     t=t2
                return h
            else:
                dyerr = self.rk2(p, Y1, t, h, j)
                yerr = yerr + dyerr
                Y1 = p.Yright.copy()
                t = t + h
                h = get_new_h2(h, self.acc, dyerr, p.Yright)
                Ynorm = np.linalg.norm(p.Yright[:, 1]) / Ynorm1
            # If Ynorm is too large, then we make a "sublevel" or "substation"
            # One can have multiple sublevels between two levels if necessary.
            if Ynorm > Ythreshold:
                raise NotImplementedError()
                # p.Yright = Y2
        #       s2=p%dat%sright
        #       if (j==1) then
        #         call Getkz(t,kz1)
        #         s1=log(kz1/kz0)
        #       else
        #         ! This might be a mistake according to sqot, but it is used.
        #         s1=log(t/kz0)
        #       end if
        #       p%dat%sright=s1
        #       ! Make station
        #       reccount=reccount+1
        #       allocate(p%next)
        #       p%next%prev=>p
        #       call GMRES(p)
        #       p=>p%next
        #       p%dat%level=p%prev%dat%level
        #       p%dat%sleft=s1
        #       p%dat%sright=s2
        #
        #       p%dat%dyxu0=cmplx(0.0,0.0,mykind)
        #       p%dat%dyxv0=cmplx(0.0,0.0,mykind)
        #       p%dat%dyxw0=cmplx(0.0,0.0,mykind)
        #       p%dat%dyxu1=cmplx(0.0,0.0,mykind)
        #       p%dat%dyxv1=cmplx(0.0,0.0,mykind)
        #       p%dat%dyxw1=cmplx(0.0,0.0,mykind)
        #       ! End make station
        #       p%dat%level=p%prev%dat%level
        #       call GetM(A,t,j)
        #       Y1=p%dat%Yleft
        #       Ynorm1=norm(Y1(:,1))

    def getM(self, t, j):
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
        if j == 1:

            kz = self.get_kz(t)
            aux = self.phi(kz)
            kK = kappa * kz / aux
            dKdz = kappa / aux**2

            kKcos = kK * self.cosbeta
            kKsin = kK * self.sinbeta
            u = t / kappa

            return np.array([
                # M(1,2)=cmplx(-kK**2,kKcos*u)/kappa2
                # M(1,5)=cmplx(0.0E0,-kKcos/kappa)
                # M(1,6)=cmplx(0.0E0,-2.0E0*dKdz*kKcos/kappa/pscale)
                [0, complex(-kK**2, kKcos * u) / kappa2, 0, 0,
                 complex(0, -kKcos / kappa),
                 complex(0, -2 * dKdz * kKcos / kappa / pscale)],
                # M(2,1)=cmplx(-1.0E0,0.0E0)
                # M(2,6)=cmplx(0.0E0,-kKcos/pscale)
                [-1, 0, 0, 0, 0,
                 complex(0, -kKcos / pscale)],
                # M(3,4)=M(1,2)
                # M(3,5)=cmplx(0.0E0,-kKsin/kappa)
                # M(3,6)=cmplx(0.0E0,-2.0E0*dKdz*kKsin/kappa/pscale)
                [0, 0, 0, complex(-kK**2, kKcos * u) / kappa2,
                 complex(0, - kKsin / kappa),
                 complex(0, - 2 * dKdz * kKsin / kappa / pscale)],
                # M(4,3)=cmplx(-1.0E0,0.0E0)
                # M(4,6)=cmplx(0.0E0,-kKsin/pscale)
                [0, 0, -1, 0, 0,
                 complex(0, -kKsin / pscale)],
                # M(5,2)=cmplx(-1.0E0,-dKdz*kKcos)/kappa2
                # M(5,4)=cmplx(0.0E0,-dKdz*kKsin/kappa2)
                # M(5,6)=cmplx(kK**2,-kKcos*u)/(pscale*kappa)
                [0, complex(-1, - dKdz * kKcos) / kappa2, 0,
                 complex(0, -dKdz * kKsin / kappa2), 0,
                 complex(kK**2, - kKcos * u) / (pscale * kappa)],
                # M(6,2)=cmplx(0.0E0,kKcos/kappa2*pscale)
                # M(6,4)=cmplx(0.0E0,kKsin/kappa2*pscale)
                [0, complex(0, kKcos / kappa2 * pscale), 0,
                 complex(0, kKsin / kappa2 * pscale), 0, 0]])

            return np.array([
                # M(1,2)=cmplx(-kK**2,kKcos*u)/kappa2
                # M(1,5)=cmplx(0.0E0,-kKcos/kappa)
                # M(1,6)=cmplx(0.0E0,-2.0E0*dKdz*kKcos/kappa/pscale)
                [0, (-kK**2 + 1j * kKcos * u) / kappa2, 0, 0, -1j * \
                 kKcos / kappa, -1j * 2 * dKdz * kKcos / kappa / pscale],
                # M(2,1)=cmplx(-1.0E0,0.0E0)
                # M(2,6)=cmplx(0.0E0,-kKcos/pscale)
                [-1, 0, 0, 0, 0, -1j * kKcos / pscale],
                # M(3,4)=M(1,2)
                # M(3,5)=cmplx(0.0E0,-kKsin/kappa)
                # M(3,6)=cmplx(0.0E0,-2.0E0*dKdz*kKsin/kappa/pscale)
                [0, 0, 0, (-kK**2 + 1j * kKcos * u) / kappa2, -1j * kKsin / \
                 kappa, -1j * 2 * dKdz * kKsin / kappa / pscale],
                # M(4,3)=cmplx(-1.0E0,0.0E0)
                # M(4,6)=cmplx(0.0E0,-kKsin/pscale)
                [0, 0, -1, 0, 0, -1j * kKsin / pscale],
                # M(5,2)=cmplx(-1.0E0,-dKdz*kKcos)/kappa2
                # M(5,4)=cmplx(0.0E0,-dKdz*kKsin/kappa2)
                # M(5,6)=cmplx(kK**2,-kKcos*u)/(pscale*kappa)
                [0, (-1 - 1j * dKdz * kKcos) / kappa2, 0,
                 -1j * dKdz * kKsin / kappa2, 0, (kK**2 - 1j * kKcos * u) / (pscale * kappa)],
                # M(6,2)=cmplx(0.0E0,kKcos/kappa2*pscale)
                # M(6,4)=cmplx(0.0E0,kKsin/kappa2*pscale)
                [0, 1j * kKcos / kappa2 * pscale, 0, 1j * kKsin / kappa2 * pscale, 0, 0]])
        elif j == 2:
            raise NotImplementedError()

        def u0(self, kz):
            # Wind speed from MOST normalized by uStar
            #  use params, only: mykind,kz0,kappa,psi0
            #  implicit none
            #  real(mykind) u0,kz,psi
            return (np.log(kz / self.kz0) + self.psi(kz) - self.psi0) / kappa


class NeutralPreLUTGenerator(PreLUTGenerator):

        # type(Tprelutdata) :: dat
        # type(Tprelutnode), pointer :: first,last

    def __init__(self, zeta0, kz0, beta, kzmax, ds, accgoal):
        PreLUTGenerator.__init__(self, zeta0, kz0, beta, kzmax, ds, accgoal)
        self.cdivkL = 0.0
        self.psi0 = self.psi(kz0)

    def get_kz(self, t):
        kz = self.kz0 * np.exp(t)
        self.lastkz = kz
        return kz

    def phi(self, kz):
        return 1

    def psi(self, kz):
        return 0

    def u0(self, kz):
        # Wind speed from MOST normalized by uStar
        return (np.log(kz / self.kz0)) / kappa


class PreLUT():

    @staticmethod
    def from_pre_file(filename, zeta0, kz0, beta, kzmax, ds):
        with open(filename, 'rb') as fid:
            fid.seek(-1, 2)     # go to the file end.
            eof = fid.tell()   # get the end of file location
            fid.seek(0, 0)      # go back to file beginning

            def read_complex(shape):
                n = np.prod(shape)
                v = np.reshape(struct.unpack('d' * 2 * n, fid.read(16 * n)), shape + (2,))
                return np.sum(v * np.array([1, 1j]), -1)

            def read_level():
                r = ([read_complex((6, 6)) for _ in range(3)] +   # Yleft, Rleft, Rright
                     [read_complex((6,)) for _ in range(6)] +   # dyxu0, dyxu1, dyxv0, dyxv1, dyxw0, dyxw1
                     list(struct.unpack('ddi', fid.read(20))))  # sleft, sright, level
                struct.unpack('i', fid.read(4))
                return r
            r = []
            while fid.tell() < eof:
                if fid.tell() == 467124:
                    print(fid.tell())
                r.append(read_level())

        return xr.Dataset({**{k: (dims, np.array(v)) for (k, dims), v in zip([
            ('Yleft', ['i', 'j', 'k']),
            ('Rleft', ['i', 'j', 'k']),
            ('Rright', ['i', 'j', 'k']),
            ('dyxu0', ['i', 'j']),
            ('dyxu1', ['i', 'j']),
            ('dyxv0', ['i', 'j']),
            ('dyxv1', ['i', 'j']),
            ('dyxw0', ['i', 'j']),
            ('dyxw1', ['i', 'j']),
            ('sleft', ['i']),
            ('sright', ['i']),
            ('level', ['i'])],
            zip(*r))}, 'zeta0': zeta0, 'beta': beta, 'kz0': kz0}, attrs={'ds': ds, 'kzmax': kzmax})

    @staticmethod
    def make_prelut(zeta0, kz0, beta, kzmax, ds, accgoal):
        if zeta0 == 0:
            return NeutralPreLUTGenerator(zeta0, kz0, beta, kzmax, ds, accgoal).make_prelut()
        else:
            raise NotImplementedError()


ref = PreLUT.from_pre_file(tfp + '0.0000-09.0000.pre', zeta0=0, beta=0, kz0=0, kzmax=0, ds=0.05)


def prelut(lut_path, prelutname, zeta0, nkz0, kz0min, kz0max, nbeta, mbeta, ds, kzmax, accgoal):
    prelutpath = Path(lut_path) / prelutname
    prelutpath.mkdir(exist_ok=True)

    jmin = np.floor(0.5 + np.log10(kz0min) * nkz0)
    jmax = np.floor(0.5 + np.log10(kz0max) * nkz0)

    xx = np.linspace(0, np.pi / 2, nbeta + mbeta + 1)

    betatab = get_beta(xx)
    # npreluts = (jmax - jmin + 1) * (mbeta + nbeta + 1)
    j_lst = np.arange(jmin, jmax + 1)
    kz0_lst = 10**(j_lst / nkz0)

    # b0 = 0
    # if zeta0 < 0:  # Unstable -psi_m at z=z0 plus a constant?
    #     raise NotImplementedError("Not tested")
    #     b0 = psi0 + np.log(Cm1 * zeta0 / 8)
    import xarray as xr
    import pandas as pd

    for i, kz0 in enumerate(tqdm(kz0_lst)):

        res = xr.concat([NeutralPreLUT(zeta0, kz0, beta, kzmax, ds).make_prelut(accgoal) for beta in betatab],
                        pd.Index(betatab, name='beta'))
        res['kz0'] = kz0
        save_complex(res, Path(lut_path) / prelutname / f'ikz0={i:03.0f}.nc')


if __name__ == '__main__':
    prelut(r'C:\mmpe\programming\python\Topfarm\CuttingEdge\Fuga\Easylut/lut',
           'preLUTs_Zeta0=0.00E+00_1_2',
           zeta0=0,
           nkz0=1,
           kz0min=1e-09,
           kz0max=0.1,
           nbeta=2,
           mbeta=0,
           ds=0.05,
           kzmax=300,
           accgoal=0.0001  # ! Level of accuracy
           )
