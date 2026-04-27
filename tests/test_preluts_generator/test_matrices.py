import numpy as np

from pyfuga.preluts_generator.matrices import compute_kstar_params


def test_compute_kstar_params_unstable_branch() -> None:
    k_star, d_kstar_dz, k_star_cos, k_star_sin = compute_kstar_params(-0.1, 1.5, 0.2, 0.8, 0.6)
    assert np.isfinite(k_star)
    assert np.isfinite(d_kstar_dz)
    assert np.isclose(k_star_cos, k_star * 0.8)
    assert np.isclose(k_star_sin, k_star * 0.6)
