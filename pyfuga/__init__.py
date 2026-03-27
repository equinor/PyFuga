import functools
import inspect
from importlib.metadata import PackageNotFoundError, version

from . import _fuga

try:
    __version__ = version("pyfuga")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["get_luts", "get_fluts", "get_preluts", "__version__"]


@functools.wraps(_fuga.get_luts)
def get_luts(*args, **kwargs):
    return _fuga.get_luts(*args, **kwargs)


@functools.wraps(_fuga.get_fluts)
def get_fluts(*args, **kwargs):
    return _fuga.get_fluts(*args, **kwargs)


@functools.wraps(_fuga.get_preluts)
def get_preluts(*args, **kwargs):
    return _fuga.get_preluts(*args, **kwargs)


get_luts.__signature__ = inspect.signature(_fuga.get_luts)
get_fluts.__signature__ = inspect.signature(_fuga.get_fluts)
get_preluts.__signature__ = inspect.signature(_fuga.get_preluts)
