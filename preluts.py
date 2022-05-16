import numpy as np
from pathlib import Path
from PyPreludium.utils import get_beta, psi, phi, dphiu, get_new_h2, save_complex, compare
from PyPreludium.constants import Cm1, n_eq, kappa, kappa2, pscale, max_recs, Ythreshold
from numpy import newaxis as na
from tqdm import tqdm
from py_wake.utils.profiling import timeit

#np.set_printoptions(precision=2, linewidth=200)


# class Prelut():
#     def __init__(self, ):

class NodeData():
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
    def __init__(self, beta):
        self.dyxu0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw0 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxu1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxv1 = np.zeros(n_eq, dtype=np.complex128)
        self.dyxw1 = np.zeros(n_eq, dtype=np.complex128)
        level = 0
        # complex(mykind), dimension(n,n) :: Yleft,Rleft,Rright
        # complex(mykind), dimension(n) :: dyxu0,dyxu1,dyxv0,dyxv1,dyxw0, &
        #                                  dyxw1
        # real(mykind) sleft,sright
        # integer(4) level

        sinbeta, cosbeta = np.sin(beta), np.cos(beta)
        self.Yleft = np.array([[-sinbeta, 0, cosbeta, 0, 0, 0],
                               [cosbeta, 0, sinbeta, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, -sinbeta, 0, cosbeta, 0, 0],
                               [0, cosbeta, 0, sinbeta, 0, 0],
                               [0, 0, 0, 0, 0, 1]], dtype=np.complex128).T
        self.Rleft = np.zeros_like(self.Yleft)


class PrelutNode():
    _next = None

    def __init__(self, beta):
        self.dat = NodeData(beta)

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, node):
        self._next = node
        self.GMRES()

    def GMRES(self):
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

        next = self.next
        # aux = np.linalg.norm(self.Yright, axis=0)
        # next.dat.Yleft = Yleft = self.Yright / aux
        # next.dat.Rleft = np.diag(aux)
        # for j in range(5):
        #     next.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j]) @ Yleft[:, j + 1:]
        #     Yleft[:, j + 1:] = Yleft[:, j + 1:] - (Yleft[:, j] * np.conj(Yleft[:, j])[:, na]) @ Yleft[:, j + 1:]

        next.dat.Yleft = Yleft = self.Yright
        next.dat.Rleft = np.zeros_like(Yleft)
        for j in range(5):
            aux = np.linalg.norm(Yleft[:, j])
            Yleft[:, j] = Yleft[:, j] / aux
            next.dat.Rleft[j, j] = aux
            next.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j])@Yleft[:, j + 1:]
            Yleft[:, j + 1:] = Yleft[:, j + 1:] - \
                np.dot((Yleft[:, j] * np.conj(Yleft[:, j])[:, na]).T, Yleft[:, j + 1:])

        aux = np.linalg.norm(Yleft[:, -1])
        Yleft[:, -1] = Yleft[:, -1] / aux
        next.dat.Rleft[-1, -1] = aux

        # B = np.zeros_like(Yleft)
        # for j in range(1, 6):
        #     B[:j, j] = next.dat.Rleft[:j, j] / next.dat.Rleft[j, j]
        B = np.triu(next.dat.Rleft / np.diag(next.dat.Rleft), 1)  # upper triangle without diagonal
        # self.dat.Rright = np.diag(1 / np.diag(next.dat.Rleft))
        # for i in range(6):
        #     for j in range(i + 1, 6):
        #         for k in range(i, j):
        #             self.dat.Rright[i, j] = self.dat.Rright[i, j] - self.dat.Rright[i, k] * B[k, j]


# 0 Rright 3.3887095708951127e-13 2.9769347466917524 (4, 5)
# (2.889161255556684e-23-1.5021795710389924e-10j)
# (1.1415144561211173e-23-1.4987908614680973e-10j)
        self.dat.Rright = np.diag(1 / np.diag(next.dat.Rleft))
        for i in range(6):
            for j in range(i + 1, 6):
                self.dat.Rright[i, j] -= np.sum(self.dat.Rright[i, i:j] * B[i:j, j])

        next.dat.Rleft = np.conj(next.dat.Rleft.T)
        self.dat.Rright = np.conj(self.dat.Rright.T)


class PrelutData():
    def __init__(self, kz0, kzmax, accgoal):
        self.smaxx = np.log(np.minimum(kzmax / kz0, 1e8))

        self.accgoal = accgoal

    #
    # character(len=18)::filename
    #   real(mykind) ds,smaxx,kz0,beta,kzmax,accgoal

      # complex(mykind), dimension(n,n) :: Yright
      # complex(mykind), dimension(n) :: x,Yx
      # type(Tprelutnode), pointer :: prev,next


class PreLUT():
    # Internt benyttes variablene:
    # u,v,w,p og t=u/kappa for kz<kzm
    # u,v,w,p og kz        for kz>kzm
    # kzm=phi(zm/L)
    # preluts formuleres i termer af  u,v,w,p og s=log(z/z0).
    # Stationer ved s=j*ds samt ved s=Log(kzm/kz0) og evt. extra stationer.
    # Level incrementeres ved stationer med s=j*ds
    # Der orthonormaliseres ved hver station og gemmes i en hægtet liste.
    # Bruger anden ordens R-K
    def __init__(self, zeta0, kz0, beta, kzmax, accgoal, ds):
        self.dat = PrelutData(kz0, kzmax, accgoal)
        self.first = PrelutNode(beta)
        self.zeta0 = zeta0
        self.kz0 = kz0
        self.beta = beta
        self.ds = ds
        self.cosbeta = np.cos(beta)
        self.sinbeta = np.sin(beta)
        self.acc = self.dat.accgoal / self.dat.smaxx

    def make_prelut(self):

        h = np.sqrt(self.acc * 6 / 3.125)
        self.lastkz = self.kz0

        yerr = self.rk2(self.first, 0, h, j=1)
        h = get_new_h2(h, self.acc, yerr, self.first.Yright)
        self.first.dat.Rleft = np.eye(6, dtype=np.complex128)
        sm = np.minimum(self.dat.smaxx, self.sm())
        self.first.dat.sleft = 0
        self.first.dat.sright = self.ds
        segment = self.first
        s1 = 0.0
        s2 = self.ds
        s_lst = np.r_[np.arange(0, sm, self.ds), sm, np.arange(sm, self.dat.smaxx)]

        for i, (s1, s2) in enumerate(zip(s_lst, s_lst[1:][:max_recs])):
            segment.dat.level = i
            # segment%dat%dyxu0=cmplx(0.0,0.0,mykind)
            # segment%dat%dyxv0=cmplx(0.0,0.0,mykind)
            # segment%dat%dyxw0=cmplx(0.0,0.0,mykind)
            # segment%dat%dyxu1=cmplx(0.0,0.0,mykind)
            # segment%dat%dyxv1=cmplx(0.0,0.0,mykind)
            # segment%dat%dyxw1=cmplx(0.0,0.0,mykind)
            segment.dat.sleft = s1
            segment.dat.sright = s2
            h = self.solve2(segment, h, yerr, self.acc, 1)
            segment.next = PrelutNode(self.beta)
            segment.next.prev = segment
            segment = segment.next

            # allocate(segment%next)
            # segment%next%prev=>segment
            # call GMRES(segment)
            # segment=>segment%next

        segment.dat.level = i
        segment.dat.sleft = s1
        segment.dat.sright = segment.dat.sleft = s1
        segment.Yright = segment.dat.Yleft
        # segment%dat%dyxu0=cmplx(0.0,0.0,mykind)
        # segment%dat%dyxv0=cmplx(0.0,0.0,mykind)
        # segment%dat%dyxw0=cmplx(0.0,0.0,mykind)
        # segment%dat%dyxu1=cmplx(0.0,0.0,mykind)
        # segment%dat%dyxv1=cmplx(0.0,0.0,mykind)
        # segment%dat%dyxw1=cmplx(0.0,0.0,mykind)
        segment.dat.Rright = np.zeros_like(segment.dat.Yleft)
        self.last = segment

        def get_res(node, k):
            if node.next is None:
                return [getattr(node.dat, k)]
            else:
                return [getattr(node.dat, k)] + get_res(node.next, k)

        import xarray as xr
        level = get_res(self.first, 'level')
        var_names = ['Yleft', 'Rleft', 'Rright',
                     'dyxu0', 'dyxu1', 'dyxv0', 'dyxv1', 'dyxw0', 'dyxw1',
                     'sleft', 'sright']
        var_values = [np.array(get_res(self.first, k)) for k in var_names]
        dims = ('level', 'm', 'n')
        return xr.Dataset({n: (dims[:len(v.shape)], v) for n, v in zip(var_names, var_values)},
                          coords={'level': level}).transpose('level', 'n', 'm', missing_dims='ignore')

    def rk2(self, node, x, h, j):
        dat = node.dat
        y = dat.Yleft
        B1 = 4.0 / 3.0
        B2 = -1.0 / 3.0
        c1 = 1.0 / 6.0
        c2 = -1.0 / 6.0
        c3 = 4.0 / 6.0
        c4 = 2.0 / 6.0

        def rk2step(t1, h, Ay, y1, j):
            ym = y1 + h * 0.5 * Ay
            A = self.getM(t1 + h * 0.5, j)
            return y1 + h * np.dot(A, ym)

        A = self.getM(x, j)
        # compare(A)

        Ay = A @ y
        y2 = rk2step(x, h, Ay, y, j)
        y3 = rk2step(x, h * 0.5, Ay, y, j)
        A = self.getM(x + h * 0.5, j)
        Ay = A @y3
        y4 = rk2step(x + h * 0.5, h * 0.5, Ay, y3, j)

        yout = B1 * y4 + B2 * y2
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
            dat.dyxu0 = h * (np.conj(a1 * c1 * y[1, :] + am * c3 * y3[1, :] + a2 * (c4 * y4[1, :] + c2 * y2[1, :])))
            dat.dyxv0 = h * (np.conj(a1 * c1 * y[3, :] + am * c3 * y3[3, :] + a2 * (c4 * y4[3, :] + c2 * y2[3, :])))
            dat.dyxw0 = h * (np.conj(a1 * c1 * y[5, :] + am * c3 * y3[5, :] +
                                     a2 * (c4 * y4[5, :] + c2 * y2[5, :]))) * kappa
            dat.dyxu1 = h * (np.conj(kz1 * a1 * c1 * y[2, :] + kzm * am *
                                     c3 * y3[1, :] + kz2 * a2 * (c4 * y4[1, :] + c2 * y2[1, :])))
            dat.dyxv1 = h * (np.conj(kz1 * a1 * c1 * y[3, :] + kzm * am *
                                     c3 * y3[3, :] + kz2 * a2 * (c4 * y4[3, :] + c2 * y2[3, :])))
            dat.dyxw1 = h * (np.conj(kz1 * a1 * c1 * y[5, :] + kzm * am * c3 *
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
        kz1 = self.kz0 * np.exp(p.dat.sleft)
        kz2 = self.kz0 * np.exp(p.dat.sright)

        if (j == 1):
            t1 = kappa * self.u0(kz1)
            t2 = kappa * self.u0(kz2)
        else:
            t1 = kz1
            t2 = kz2

        t = t1
        # Yerr=cmplx(0.0,0.0,mykind)
        # p%dat%dyxu0=cmplx(0.0,0.0,mykind)
        # p%dat%dyxv0=cmplx(0.0,0.0,mykind)
        # p%dat%dyxw0=cmplx(0.0,0.0,mykind)
        # p%dat%dyxu1=cmplx(0.0,0.0,mykind)
        # p%dat%dyxv1=cmplx(0.0,0.0,mykind)
        # p%dat%dyxw1=cmplx(0.0,0.0,mykind)
        # Y1=p%dat%Yleft
        Ynorm1 = np.linalg.norm(p.dat.Yleft[:, 1])
        while True:
            if (1.1 * h + t > t2):
                h = t2 - t
                dyerr = self.rk2(p, t, h, j)
                yerr = yerr + dyerr
                #     p%Yright=Y2
                h = get_new_h2(h, self.acc, dyerr, p.Yright)
                #     t=t2
                return h
            else:
                dyerr = self.rk2(p, t, h, j)
                yerr = yerr + dyerr
                p.dat.Yleft = p.Yright
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


class NeutralPreLUT(PreLUT):

        # type(Tprelutdata) :: dat
        # type(Tprelutnode), pointer :: first,last

    def __init__(self, zeta0, kz0, beta, kzmax, accgoal, ds):
        PreLUT.__init__(self, zeta0, kz0, beta, kzmax, accgoal, ds)
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

    def sm(self):
        return self.dat.smaxx


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

        res = xr.concat([NeutralPreLUT(zeta0, kz0, beta, kzmax, accgoal, ds).make_prelut() for beta in betatab],
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
