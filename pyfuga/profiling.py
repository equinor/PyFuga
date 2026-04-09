"""profiling.py
==============
Performance profiling utilities for PyFuga functions.

This module provides decorators for timing function calls and profiling
line-by-line execution time. Used to identify performance bottlenecks
in the preLUT generation and Fourier LUT stages.
"""

from __future__ import annotations

import functools
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


def timeit(
    func: Callable[P, R],
    min_time: float = 0,
    min_runs: int = 1,
    verbose: bool = False,
    line_profile: bool = False,
    profile_funcs: Sequence[Callable[..., object]] | None = None,
) -> Callable[P, tuple[R | None, list[float]]]:
    if profile_funcs is None:
        profile_funcs_list: list[Callable[..., object]] = []
    else:
        profile_funcs_list = list(profile_funcs)

    @functools.wraps(func)
    def newfunc(*args: P.args, **kwargs: P.kwargs) -> tuple[R | None, list[float]]:
        if line_profile and sys.gettrace() is None:  # pragma: no cover
            lp_wrapper = line_timeit(func, profile_funcs_list)
            t = time.time()
            profiled_res, lp = lp_wrapper(*args, **kwargs)
            t = time.time() - t
            if verbose:
                lp.print_stats()
            return profiled_res, [t]
        t_lst: list[float] = []
        res: R | None = None
        time_start = time.time()
        for _i in range(100000):
            t0 = time.time_ns()
            res = func(*args, **kwargs)
            t_lst.append((time.time_ns() - t0) * 1e-9)
            if (time.time() - time_start) > min_time and len(t_lst) >= min_runs:
                break

        if verbose:  # pragma: no cover
            fn = func.__name__ if hasattr(func, "__name__") else "Function"
            print(f"{fn}: {np.mean(t_lst):f} +/-{np.std(t_lst):f} ({len(t_lst):d} runs)")
        return res, t_lst

    return newfunc


def line_timeit(  # pragma: no cover
    func: Callable[P, R],
    profile_funcs: Sequence[Callable[..., object]] | None = None,
) -> Callable[P, tuple[R, Any]]:
    if profile_funcs is None:
        profile_funcs_list: list[Callable[..., object]] = []
    else:
        profile_funcs_list = list(profile_funcs)
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp_any = cast(Any, lp)
    lp_any.timer_unit = 1e-6

    for f in profile_funcs_list:
        lp_any.add_function(f)
    lp_wrapper = lp_any(func) if sys.gettrace() is None else func

    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[R, Any]:
        return lp_wrapper(*args, **kwargs), lp_any

    return _wrapped


def print_time(f: Callable[P, R]) -> Callable[P, R]:  # pragma: no cover
    """Print time decorator
    prints name of method and time of execution

    >>> @print_time
    >>> def test():
    >>>    time.sleep(1)
    >>>
    >>> test()
    test            1.000s
    """

    def wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        t = time.time()
        res = f(*args, **kwargs)
        print(f"{f.__name__:<12}\t{time.time() - t:.3f}s")
        return res

    w: Callable[P, R] = wrap
    w.__name__ = f.__name__
    return w
