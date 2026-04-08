from pyfuga.profiling import timeit


def test_timeit_accepts_explicit_profile_funcs():
    def add(a, b):
        return a + b

    wrapped = timeit(add, min_runs=1, profile_funcs=(add,))
    result, timings = wrapped(2, 3)

    assert result == 5
    assert len(timings) >= 1
