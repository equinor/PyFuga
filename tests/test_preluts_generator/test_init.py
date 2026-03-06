# tests/test_preluts_generator_init.py

import importlib

import pyfuga.preluts_generator as preluts_gen
import pyfuga.utils as utils


def test_preluts_generator_init_respects_preludium_flag():
    """
    Ensure that pyfuga.preluts_generator.__init__ honours utils.preludium_equivalent:

    - when False, PrelutNode is the standard implementation from .nodes
    - when True, PrelutNode is imported from the Preludium routines

    This also lightly checks that PreLUTGenerator and PrelutNodeFirst remain importable.
    """

    original_flag = utils.preludium_equivalent
    try:
        # ------------------------------------------------------------------
        # 1) Normal mode: preludium_equivalent = False
        # ------------------------------------------------------------------
        utils.preludium_equivalent = False
        importlib.reload(preluts_gen)

        from pyfuga.preluts_generator import PreLUTGenerator, PreLUTNode, PreLUTNodeFirst

        # Sanity: the public API is there
        assert PreLUTGenerator is not None
        assert PreLUTNodeFirst is not None

        # In normal mode, PrelutNode should come from the .nodes module
        assert PreLUTNode.__module__.endswith(".preluts_generator.nodes")

        # ------------------------------------------------------------------
        # 2) Preludium mode: preludium_equivalent = True
        # ------------------------------------------------------------------
        utils.preludium_equivalent = True
        importlib.reload(preluts_gen)

        from pyfuga.preluts_generator import PreLUTNode as PreLUTNode

        # In Preludium mode, PrelutNode should now be provided by the
        # Preludium routines module
        assert "preludium_eq_routines" in PreLUTNode.__module__

    finally:
        # Restore original flag and module state so other tests are unaffected
        utils.preludium_equivalent = original_flag
        importlib.reload(preluts_gen)
