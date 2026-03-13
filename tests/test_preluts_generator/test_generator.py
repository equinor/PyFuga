import math

import numpy as np
import pytest

from pyfuga.preluts_generator.generator import PreLUTGenerator


def test_make_prelut_raises_on_log_s_len_mismatch(monkeypatch):
    """Inject a mismatched log_s_lst slice to cover the defensive check on lines 171-172."""
    import pyfuga.preluts_generator.generator as gen_module

    gen = PreLUTGenerator(zeta0=0, kz0=0.1, beta=0.0, kzmax=10.0, ds=0.5, accgoal=1e-2)

    real_sort = gen_module.np.sort

    # np.sort is called exactly once inside make_prelut (for log_s_lst). Wrap only that
    # first call to return an object whose [1:-1] slice has one extra element vs [2:],
    # triggering the length check.
    class _LenMismatch:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            v = self._data[key]
            if key == slice(1, -1, None):
                return np.r_[v, 0.0]  # one extra element → len(log_s_from) > len(log_s_to)
            return v

    fired = [False]

    def patched_sort(a, *args, **kwargs):
        result = real_sort(a, *args, **kwargs)
        if not fired[0]:
            fired[0] = True
            return _LenMismatch(result)
        return result

    monkeypatch.setattr(gen_module.np, "sort", patched_sort)

    with pytest.raises(ValueError, match="Length mismatch between log_s slices"):
        gen.make_prelut()


def test_calculate_s_transition_max_iterations_runtime_error(monkeypatch):
    """Force non-convergence and assert RuntimeError is raised."""
    # Unstable regime to enter the Newton-Raphson branch
    gen = PreLUTGenerator(
        zeta0=-0.1,  # unstable
        kz0=0.1,
        beta=0.0,
        kzmax=1000.0,
        ds=0.1,
        accgoal=1e-2,
    )

    # Prevent convergence: always report not close
    monkeypatch.setattr(math, "isclose", lambda a, b, rel_tol=..., abs_tol=...: False)

    with pytest.raises(RuntimeError, match="Maximum iterations reached without convergence."):
        gen.calculate_s_transition()
