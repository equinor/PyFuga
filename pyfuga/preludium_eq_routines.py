"""preludium_eq_routines.py
=========================
Automaton-equivalent routines for Preludium code model compatibility.

This module provides alternative implementations of node advancement and
error control functions that match the Preludium Fortran code more closely.
Used for testing and validating against the original Preludium atmospheric
model. Controlled via the :data:`pyfuga.utils.preludium_equivalent` flag.
"""

import numpy as np

from pyfuga.common import complex_norm
from pyfuga.preluts_generator import PreLUTNode
from pyfuga.utils import jit


@jit("double(double,double,complex128[:,:],complex128[:,:])")
def get_new_h2_Preludium(h, acc, Yerr, Y):
    err = np.max(complex_norm(Yerr, axis=0) / complex_norm(Y, axis=0))
    return np.maximum(np.minimum(0.9 * h * (acc * h / err) ** (1 / 3), 4.0e-1), 1.0e-4)

    # return np.clip(0.9 * h * (acc / err)**(1 / 3), 1.0E-4, 4.0E-1)


class PrelutNodePreludium(PreLUTNode):
    def advance_node_by_qr_decomposition(self):
        from numpy import newaxis as na

        # use params, only: mykind,n
        # use contr, only: Tprelutnode
        # use vector_functions, only: outer
        # implicit none
        # Modified Gram-Schmidt orthonormalisation
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
        # node.dat.Y_lower = Y_lower = self.Yright / aux
        # node.dat.Rleft = np.diag(aux)
        # for j in range(5):
        #     node.dat.Rleft[j, j + 1:] = np.conj(Y_lower[:, j]) @ Y_lower[:, j + 1:]
        #     Y_lower[:, j + 1:] =
        #  Y_lower[:, j + 1:] - (Y_lower[:, j] * np.conj(Y_lower[:, j])[:, na]) @ Y_lower[:, j + 1:]

        Y_lower = np.array(self.Y_upper, copy=True)
        node = PrelutNodePreludium()
        node.R_lower = np.zeros_like(Y_lower)
        for j in range(5):
            aux = np.linalg.norm(Y_lower[:, j])
            Y_lower[:, j] = Y_lower[:, j] / aux
            node.R_lower[j, j] = aux
            node.R_lower[j, j + 1 :] = np.dot(np.conj(Y_lower[:, j]), Y_lower[:, j + 1 :])
            Y_lower[:, j + 1 :] = Y_lower[:, j + 1 :] - np.dot(
                (Y_lower[:, j] * np.conj(Y_lower[:, j])[:, na]).T, Y_lower[:, j + 1 :]
            )

        aux = np.linalg.norm(Y_lower[:, -1])
        Y_lower[:, -1] = Y_lower[:, -1] / aux
        node.R_lower[-1, -1] = aux

        # B = np.zeros_like(Y_lower)
        # for j in range(1, 6):
        #     B[:j, j] = node.dat.Rleft[:j, j] / node.dat.Rleft[j, j]
        B = np.triu(node.R_lower / np.diag(node.R_lower), 1)  # upper triangle without diagonal
        # self.dat.Rright = np.diag(1 / np.diag(node.dat.Rleft))
        # for i in range(6):
        #     for j in range(i + 1, 6):
        #         for k in range(i, j):
        #             self.dat.Rright[i, j] = self.dat.Rright[i, j] - self.dat.Rright[i, k] * B[k, j]

        self.R_upper = np.diag(1 / np.diag(node.R_lower))
        for i in range(6):
            for j in range(i + 1, 6):
                self.R_upper[i, j] -= np.sum(self.R_upper[i, i:j] * B[i:j, j])

        node.R_lower = np.conj(node.R_lower.T)
        self.R_upper = np.conj(self.R_upper.T)
        node.Y_lower = Y_lower
        return node

        # rel_err(ref.sel(i=self.level + 1).Y_lower.T.values, node.Y_lower)
