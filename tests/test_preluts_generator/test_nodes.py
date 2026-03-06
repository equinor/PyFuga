import numpy as np
import pytest

from pyfuga.preluts_generator.nodes import PreLUTNode


def test_advance_node_by_qr_decomposition_with_none_y_upper():
    """Test that ValueError is raised when Y_upper is None."""
    node = PreLUTNode()
    node.Y_upper = None

    with pytest.raises(ValueError, match="Y_upper must be initialised before QR decomposition."):
        node.advance_node_by_qr_decomposition()


def test_advance_node_by_qr_decomposition_valid_matrix():
    """Test QR decomposition with a valid complex matrix."""
    node = PreLUTNode()
    node.Y_upper = np.array(
        [[1 + 1j, 2 + 0j], [0 + 0j, 1 + 1j], [1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j], [1 + 1j, 1 + 1j], [0 + 0j, 0 + 1j]],
        dtype=np.complex128,
    )

    next_node = node.advance_node_by_qr_decomposition()

    assert next_node.Y_lower is not None
    assert next_node.Y_lower.shape == (6, 2)
    assert next_node.R_lower is not None
    assert next_node.R_lower.shape == (2, 2)
    assert node.R_upper is not None
    assert np.allclose(next_node.Y_lower @ np.conj(next_node.R_lower.T), node.Y_upper)
