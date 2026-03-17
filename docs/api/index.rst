API Reference
=============

This section documents the public API of PyFuga.

PyFuga provides one high-level entry point for full LUT generation and a
set of lower-level pipeline components for advanced workflows.

High-level API
--------------

.. autosummary::
   :toctree: generated

   pyfuga.get_luts

Pipeline API
------------

.. autosummary::
   :toctree: generated

   pyfuga.preluts_generator.generator.PreLUTGenerator.make_prelut
   pyfuga.flut.FourierLUTGenerator.make_lut
   pyfuga.trafalgar.Trafalgar.make_luts

Core classes
------------

.. autosummary::
   :toctree: generated

   pyfuga.preluts_generator.generator.PreLUTGenerator
   pyfuga.flut.FourierLUTGenerator
   pyfuga.trafalgar.Trafalgar
   pyfuga.preluts.PreLUT
   pyfuga.preluts.PreLUTs

Module reference
----------------

.. autosummary::
   :toctree: generated
   :recursive:

   pyfuga
