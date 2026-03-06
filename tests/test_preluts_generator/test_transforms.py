import numpy as np

import pyfuga.preluts_generator.transforms as transforms


def test_t_to_s_unstable_lastkz_negative_uses_exp_branch():
    """
    Exercise the 'lastkz < 0' branch in t_to_s_unstable, where x is set via
    x = exp(b). We don't care about the physical meaning here, only that the
    code path runs and returns a finite value.
    """
    # Unstable: zeta0 < 0
    zeta0 = -0.1
    lastkz = -1.0  # <- triggers the lastkz < 0 branch
    psi0 = 0.0
    cdivkL = 0.1
    t = 0.5

    result = transforms.t_to_s_unstable(t, zeta0, lastkz, psi0, cdivkL)
    assert np.isfinite(result)


def test_t_to_s_unstable_negative_x_fallback_branch():
    """
    Exercise the 'if x < 0: x = exp(b)' fallback inside the Newton loop in
    t_to_s_unstable. We force an initial negative x by patching
    transforms.get_phi_inverse, then call the Python version (.py_func)
    so coverage sees the branch.
    """

    # Save original to restore afterwards
    original_get_phi_inverse = transforms.get_phi_inverse

    try:
        # Choose a fake phi_inverse that makes x negative:
        # x = (cdivkL * lastkz) / ((aux**2 + 1) * (1 + aux)**2)
        # With aux = 0 and cdivkL < 0, lastkz > 0 -> x < 0.
        def fake_phi_inverse(kz, cdivkL):
            return 0.0

        transforms.get_phi_inverse = fake_phi_inverse

        zeta0 = -0.1  # unstable, CM_UNSTABLE * zeta0 / 8 > 0 so log(...) is defined
        lastkz = 1.0  # > 0 -> we go into the Newton branch (not the lastkz < 0 branch)
        psi0 = 0.0
        cdivkL = -1.0  # with aux = 0, this makes x negative on first evaluation
        t = 0.5

        result = transforms.t_to_s_unstable(t, zeta0, lastkz, psi0, cdivkL)
        assert np.isfinite(result)

    finally:
        # Restore original function to avoid leaking the patch into other tests
        transforms.get_phi_inverse = original_get_phi_inverse
