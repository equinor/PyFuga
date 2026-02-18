"""Tests for the typing module."""

from typing import get_args, get_origin

import numpy as np

from pyfuga.typing import ComplexArray


def test_complex_array_type_alias_is_defined():
    """Verify ComplexArray type alias is properly defined."""
    assert ComplexArray is not None


def test_complex_array_is_ndarray_of_complex128():
    """Verify ComplexArray resolves to NDArray[complex128]."""
    # Get the origin and args of the type alias
    origin = get_origin(ComplexArray)
    args = get_args(ComplexArray)

    # ComplexArray should resolve to np.ndarray
    assert origin is np.ndarray
    # NDArray[dtype] has 2 args: (shape, dtype)
    assert len(args) == 2
    # The second arg should be the complex128 dtype
    # It's wrapped in numpy.dtype, so check the type name is in the string
    assert "complex128" in str(args[1])
