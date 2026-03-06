import pyfuga.utils as utils
from pyfuga.preluts_generator.generator import PreLUTGenerator
from pyfuga.preluts_generator.nodes import PreLUTNode, PreLUTNodeFirst

if utils.preludium_equivalent:
    from pyfuga.preludium_eq_routines import PrelutNodePreludium as PreLUTNode
else:
    from .nodes import PreLUTNode

__all__ = ["PreLUTGenerator", "PreLUTNode", "PreLUTNodeFirst"]

# np.set_printoptions(precision=2, linewidth=200)
# left: (sub)station (lower height)
# right: (sub)station + 1 (higher height)
# (Y+).x = b is a set of boundary condition equations.
# R is a Gram-Smidt (or QR) transformation of Y
# dbx_const = Delta_b for constant longitudinal forcing
# dbx_lin = Delta_b for longitudinal forcing proportional to kz
# Additional if transversal and/or vertical forcing is present:
#  dby_const = Delta_b for constant transversal forcing
#  dby_lin = Delta_b for transversal forcing proportional to kz
#  dbz_const = Delta_b for constant vertical forcing
#  dbz_lin = Delta_b for vertical forcing proportional to kz
# Delta_b = integral of (Y+).f between two  (sub)stations
# where f is the wind turbine forcing.
