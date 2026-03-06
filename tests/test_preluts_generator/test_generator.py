import math

import pytest

from pyfuga.preluts_generator.generator import PreLUTGenerator


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
