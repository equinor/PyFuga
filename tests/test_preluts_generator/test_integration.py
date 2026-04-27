import numpy as np

from pyfuga.preluts_generator.integration import COORD_T, modified_midpoint_integration_step


def test_modified_midpoint_integration_step_unstable_coord_t_updates_const_terms() -> None:
    y = np.eye(6, dtype=np.complex128)
    zeros = np.zeros(6, dtype=np.complex128)
    y_upper, y_err = modified_midpoint_integration_step(
        y=y,
        x=0.01,
        h=0.001,
        j=COORD_T,
        kz0=0.1,
        psi0=0.0,
        lastkz=0.2,
        zeta0=-0.1,
        cdivkL=0.2,
        cosbeta=1.0,
        sinbeta=0.0,
        dbx_const=zeros.copy(),
        dby_const=zeros.copy(),
        dbz_const=zeros.copy(),
        dbx_lin=zeros.copy(),
        dby_lin=zeros.copy(),
        dbz_lin=zeros.copy(),
    )
    assert y_upper.shape == (6, 6)
    assert y_err.shape == (6, 6)
