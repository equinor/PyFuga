import functools
import sys
import time
import numpy as np


def timeit(func, min_time=0, min_runs=1, verbose=False, line_profile=False, profile_funcs=[]):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        if line_profile and getattr(sys, 'gettrace')() is None:  # pragma: no cover
            lp_wrapper = line_timeit(func, profile_funcs)
            t = time.time()
            res, lp = lp_wrapper(*args, **kwargs)
            t = time.time() - t
            if verbose:
                lp.print_stats()
            return res, [t]
        else:
            t_lst = []
            time_start = time.time()
            for i in range(100000):
                t0 = time.time_ns()
                res = func(*args, **kwargs)
                t_lst.append((time.time_ns() - t0) * 1e-9)
                if (time.time() - time_start) > min_time and len(t_lst) >= min_runs:
                    break

            if verbose:  # pragma: no cover
                if hasattr(func, '__name__'):
                    fn = func.__name__
                else:
                    fn = "Function"
                print('%s: %f +/-%f (%d runs)' % (fn, np.mean(t_lst), np.std(t_lst), i + 1))
            return res, t_lst
    return newfunc


def line_timeit(func, profile_funcs=[]):  # pragma: no cover
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.timer_unit = 1e-6

    for f in profile_funcs:
        lp.add_function(f)
    if getattr(sys, 'gettrace')() is None:
        lp_wrapper = lp(func)
    else:
        # in debug mode
        lp_wrapper = func
    return lambda *args, lp=lp, **kwargs: (lp_wrapper(*args, **kwargs), lp)


def print_time(f):
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
        print("%-12s\t%.3fs" % (f.__name__, time.time() - t))
        return res
    w = wrap
    w.__name__ = f.__name__
    return w
