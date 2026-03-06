"""
Constants used in PyFuga.
"""

# PreLUT constants
CM_STABLE = 5.0  # MO Stability constant in phi_m (stable)
CM_UNSTABLE = -19.0  # MO Stability constant in phi_m (unstable) Hogstrom (1988)
N_EQ = 6  # ! Number of equations
KAPPA = 0.4  # Von Karman constant
KAPPA_SQUARED = 0.16  # Square of kappa
P_SCALE = 1  # Pressure normalization variable
MAX_NODES = 400
Y_NORM_THRESHOLD = 100  # Max value of Ynorm before a sublevel is made

# LUT constant
FIRST_Z_OUTPUT = 1.0  # First height level in output

# Other constants
UVW_LT = ["UL", "UT", "VL", "VT", "WL", "WT", "PL", "PT"]

# Coordinate system indicators
COORD_T = 1  # t = u0 * KAPPA
COORD_S = 2  # s = kz
