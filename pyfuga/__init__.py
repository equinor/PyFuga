import functools
import inspect
from importlib.metadata import PackageNotFoundError, version

from . import _fuga

try:
    __version__ = version("pyfuga")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["get_luts", "__version__"]


@functools.wraps(_fuga.get_luts)
def get_luts(*args, **kwargs):
    return _fuga.get_luts(*args, **kwargs)


get_luts.__signature__ = inspect.signature(_fuga.get_luts)
