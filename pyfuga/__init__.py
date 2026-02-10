__all__ = ["get_luts"]


def get_luts(*args, **kwargs):
    from ._fuga import get_luts as _get_luts

    return _get_luts(*args, **kwargs)
