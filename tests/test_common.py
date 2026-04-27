from pyfuga.common import get_phi


def test_get_phi_unstable_branch_returns_positive_value() -> None:
    phi = get_phi(-0.1, 1.0, 0.2)
    assert phi > 0
