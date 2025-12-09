import numpy as np

from pyfuga.common import complex_norm
from pyfuga.preluts_generator import PrelutNode
from pyfuga.utils import jit


@jit('double(double,double,complex128[:,:],complex128[:,:])')
def get_new_h2_Preludium(h, acc, Yerr, Y):
    err = np.max(complex_norm(Yerr, axis=0) / complex_norm(Y, axis=0))
    return np.maximum(np.minimum(0.9 * h * (acc * h / err)**(1 / 3), 4.0E-1), 1.0E-4)

    # return np.clip(0.9 * h * (acc / err)**(1 / 3), 1.0E-4, 4.0E-1)


class PrelutNodePreludium(PrelutNode):
    def GMRES(self):
        from numpy import newaxis as na

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
        # aux = np.linalg.norm(self.Yright, axis=0)
        # node.dat.Yleft = Yleft = self.Yright / aux
        # node.dat.Rleft = np.diag(aux)
        # for j in range(5):
        #     node.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j]) @ Yleft[:, j + 1:]
        #     Yleft[:, j + 1:] = Yleft[:, j + 1:] - (Yleft[:, j] * np.conj(Yleft[:, j])[:, na]) @ Yleft[:, j + 1:]

        Yleft = self.Yright.copy()
        node = PrelutNodePreludium()
        node.Rleft = np.zeros_like(Yleft)
        for j in range(5):
            aux = np.linalg.norm(Yleft[:, j])
            Yleft[:, j] = Yleft[:, j] / aux
            node.Rleft[j, j] = aux
            node.Rleft[j, j + 1:] = np.dot(np.conj(Yleft[:, j]), Yleft[:, j + 1:])
            Yleft[:, j + 1:] = Yleft[:, j + 1:] - \
                np.dot((Yleft[:, j] * np.conj(Yleft[:, j])[:, na]).T, Yleft[:, j + 1:])

        aux = np.linalg.norm(Yleft[:, -1])
        Yleft[:, -1] = Yleft[:, -1] / aux
        node.Rleft[-1, -1] = aux

        # B = np.zeros_like(Yleft)
        # for j in range(1, 6):
        #     B[:j, j] = node.dat.Rleft[:j, j] / node.dat.Rleft[j, j]
        B = np.triu(node.Rleft / np.diag(node.Rleft), 1)  # upper triangle without diagonal
        # self.dat.Rright = np.diag(1 / np.diag(node.dat.Rleft))
        # for i in range(6):
        #     for j in range(i + 1, 6):
        #         for k in range(i, j):
        #             self.dat.Rright[i, j] = self.dat.Rright[i, j] - self.dat.Rright[i, k] * B[k, j]

        self.Rright = np.diag(1 / np.diag(node.Rleft))
        for i in range(6):
            for j in range(i + 1, 6):
                self.Rright[i, j] -= np.sum(self.Rright[i, i:j] * B[i:j, j])

        node.Rleft = np.conj(node.Rleft.T)
        self.Rright = np.conj(self.Rright.T)
        node.Yleft = Yleft
        return node

        # rel_err(ref.sel(i=self.level + 1).Yleft.T.values, node.Yleft)
