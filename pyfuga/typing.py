"""This module defines common type aliases used throughout the PyFuga library."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Use TypeAlias instead of `type` keyword for Python 3.10/3.11 compatibility
ComplexArray: TypeAlias = NDArray[np.complex128]  # noqa: UP040
