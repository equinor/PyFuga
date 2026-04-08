"""constants.py
==============
Global constants for atmospheric stability, numerical algorithms, and data structure sizes.

Key constants:
    - **Stability (MO)**: CM_STABLE, CM_UNSTABLE for Monin-Obukhov relations
    - **Numerics**: KAPPA (von Kármán), MAX_NODES (max preLUT nodes),
      Y_NORM_THRESHOLD (threshold to trigger QR decomposition and substations
      between nodes)
    - **Data**: N_EQ (number of differential equations), UVW_LT (variable names)
    - **Coordinate systems**: COORD_T (log profile), COORD_S (kz)
"""

# PreLUT constants
CM_STABLE: float = 5.0  # MO Stability constant in phi_m (stable)
CM_UNSTABLE: float = -19.0  # MO Stability constant in phi_m (unstable) Hogstrom (1988)
N_EQ: int = 6  # ! Number of equations
KAPPA: float = 0.4  # Von Karman constant
KAPPA_SQUARED: float = 0.16  # Square of kappa
P_SCALE: int = 1  # Pressure normalisation variable
MAX_NODES: int = 400
Y_NORM_THRESHOLD: int = 100  # Max value of Ynorm before a sublevel is made

# LUT constant
FIRST_Z_OUTPUT: float = 1.0  # First height level in output
DS: float = 0.05  # Vertical grid spacing in log-height coordinate s = ln(z/z0)

# Other constants
UVW_LT: list[str] = ["UL", "UT", "VL", "VT", "WL", "WT", "PL", "PT"]

# Coordinate system indicators
COORD_T: int = 1  # t = u0 * KAPPA
COORD_S: int = 2  # s = kz
