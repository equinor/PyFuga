from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyfuga")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["get_luts", "__version__"]


def get_luts(*args, **kwargs):
    from ._fuga import get_luts as _get_luts

    return _get_luts(*args, **kwargs)
