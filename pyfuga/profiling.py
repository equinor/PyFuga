import functools
import sys
import time

import numpy as np


def timeit(func, min_time=0, min_runs=1, verbose=False, line_profile=False, profile_funcs=None):
    if profile_funcs is None:
        profile_funcs = []

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if line_profile and sys.gettrace() is None:  # pragma: no cover
            lp_wrapper = line_timeit(func, profile_funcs)
            t = time.time()
            res, lp = lp_wrapper(*args, **kwargs)
            t = time.time() - t
            if verbose:
                lp.print_stats()
            return res, [t]
        else:
            t_lst = []
            res = None
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


def line_timeit(func, profile_funcs=None):  # pragma: no cover
    if profile_funcs is None:
        profile_funcs = []
    from line_profiler import LineProfiler

    lp = LineProfiler()

    lp = LineProfiler()
    lp.timer_unit = 1e-6

    for f in profile_funcs:
        lp.add_function(f)
    lp_wrapper = lp(func) if sys.gettrace() is None else func  # in debug mode if sys.gettrace() is not None
    return lambda *args, lp=lp, **kwargs: (lp_wrapper(*args, **kwargs), lp)


def print_time(f):  # pragma: no cover
    """Print time decorator
    prints name of method and time of execution

    >>> @print_time
    >>> def test():
    >>>    time.sleep(1)
    >>>
    >>> test()
    test            1.000s
    """

    def wrap(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        print(f"{f.__name__:<12}\t{time.time() - t:.3f}s")
        return res

    w = wrap
    w.__name__ = f.__name__
    return w
