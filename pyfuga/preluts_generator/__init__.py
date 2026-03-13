import pyfuga.utils as utils
from pyfuga.preluts_generator.generator import PreLUTGenerator
from pyfuga.preluts_generator.nodes import PreLUTNode, PreLUTNodeFirst

if utils.preludium_equivalent:
    from pyfuga.preludium_eq_routines import PrelutNodePreludium as PreLUTNode
else:
    from .nodes import PreLUTNode

__all__ = ["PreLUTGenerator", "PreLUTNode", "PreLUTNodeFirst"]
