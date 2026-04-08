import functools
import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import _fuga

try:
    __version__ = version("pyfuga")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["get_luts", "get_fluts", "get_preluts", "__version__"]


@functools.wraps(_fuga.get_luts)
def get_luts(*args: Any, **kwargs: Any) -> Any:
    return _fuga.get_luts(*args, **kwargs)


@functools.wraps(_fuga.get_fluts)
def get_fluts(*args: Any, **kwargs: Any) -> Any:
    return _fuga.get_fluts(*args, **kwargs)


@functools.wraps(_fuga.get_preluts)
def get_preluts(*args: Any, **kwargs: Any) -> Any:
    return _fuga.get_preluts(*args, **kwargs)


get_luts.__signature__ = inspect.signature(_fuga.get_luts)  # type: ignore[attr-defined]
get_fluts.__signature__ = inspect.signature(_fuga.get_fluts)  # type: ignore[attr-defined]
get_preluts.__signature__ = inspect.signature(_fuga.get_preluts)  # type: ignore[attr-defined]
