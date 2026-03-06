"""
Defines the PreLUTNode and PreLUTNodeFirst classes for preLUT generation. Key features
include:

- The PrelutNode class, which represents the solution state at a station. It provides methods
  for resetting error accumulators and progressing to the next state through a Modified
  Gram-Schmidt orthonormalisation (GMRES) that produces a QR decomposition of the state matrix.

- The PrelutNodeFirst class, a specialised subclass of PrelutNode that initialises the first
  state with prescribed boundary conditions based on the input wind direction (beta) and
  station spacing (ds).
"""

import numpy as np

from pyfuga import utils
from pyfuga.constants import N_EQ
from pyfuga.typing import ComplexArray
from pyfuga.utils import jit


@jit("Tuple((complex128[:,:],complex128[:,:],complex128[:,:]))(complex128[:,:])")
def qr_decomposition(input_matrix: ComplexArray) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """
    Perform a QR decomposition of an input matrix with columns representing linearly independent vectors.

    Returns a tuple containing:
        1) the unitary, orthonormal basis matrix, Q,
        2) the Hermitian adjoint (conjugate transpose) of the lower triangular matrix, R, and
        3) R_adj_inv, the inverse of the Hermitian adjoint of R.
    """
    Q, R = np.linalg.qr(input_matrix)
    R_adj = np.conj(R.T)
    R_adj_inv = np.linalg.inv(R_adj).astype(np.complex128)
    return Q, R_adj, R_adj_inv


class PreLUTNode:
    """
    Represents a node in the preLUT generation process.

    Encapsulates state variables and error accumulators at a station and provides
    methods to reset errors and advance to the next node using a QR decomposition.
    """

    Y_upper: ComplexArray | None
    R_lower: ComplexArray | None
    Y_lower: ComplexArray | None
    R_upper: ComplexArray | None

    def __init__(self):
        self.reset_forcing_accumulators()
        self.Y_upper = None
        self.R_lower = None
        self.Y_lower = None
        self.R_upper = None

    def set_segment_bounds(self, log_s_lower, log_s_upper):
        """Set the lower and upper bounds for the segment in log(s)."""
        self.log_s_lower = log_s_lower
        self.log_s_upper = log_s_upper

    def reset_forcing_accumulators(self):
        """Reset the differential error accumulators for forcing ($\\Delta b$)."""
        self.dbx_const = np.zeros(N_EQ, dtype=np.complex128)
        self.dby_const = np.zeros(N_EQ, dtype=np.complex128)
        self.dbz_const = np.zeros(N_EQ, dtype=np.complex128)
        self.dbx_lin = np.zeros(N_EQ, dtype=np.complex128)
        self.dby_lin = np.zeros(N_EQ, dtype=np.complex128)
        self.dbz_lin = np.zeros(N_EQ, dtype=np.complex128)

    def generate_next_node(self, sleft, sright) -> "PreLUTNode":
        """Generate and return the next node via QR decomposition."""
        next_node = self.advance_node_by_qr_decomposition()
        next_node.set_segment_bounds(sleft, sright)
        return next_node

    def advance_node_by_qr_decomposition(self) -> "PreLUTNode":
        """
        Perform QR decomposition of the final state matrix of a segment, Y_upper, to
        get the first statement matrix of the next segment, Y_lower. Store the
        decomposition results (R_lower, Y_lower, R_upper) in the new node.
        """
        node = PreLUTNode()
        if self.Y_upper is None:
            raise ValueError("Y_upper must be initialised before QR decomposition.")
        Y_lower, R_lower, R_upper = qr_decomposition(self.Y_upper)
        node.R_lower = R_lower
        node.Y_lower = Y_lower
        self.R_upper = R_upper
        return node


if utils.preludium_equivalent:
    from pyfuga.preludium_eq_routines import PrelutNodePreludium as PreLUTNode


class PreLUTNodeFirst(PreLUTNode):
    """
    Specialised first node for preLUT generation.

    Initialises the node with boundary conditions based on the phase angle (beta) and station spacing (ds).
    """

    def __init__(self, beta, ds):
        """Initialise the first node with the prescribed boundary conditions."""
        PreLUTNode.__init__(self)
        self.set_segment_bounds(log_s_lower=0, log_s_upper=ds)
        sinbeta, cosbeta = np.sin(beta), np.cos(beta)
        self.Y_lower = np.array(
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
        self.R_lower = np.eye(6, dtype=np.complex128)
