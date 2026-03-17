.. image:: _static/logo.svg
   :alt: PyFuga logo
   :align: center

|

Welcome to PyFuga's documentation!
==================================

PyFuga is a Python implementation of the Fuga wake model, a linearised RANS / CFD model for wind
farm flows. These pages describe the theoretical formulation, user workflows, and Python API of
PyFuga.

.. note::
    
   This documentation is a work in progress. The theory pages are
   being developed alongside the Fuga report, and the API pages will
   be expanded as the code matures. All sections marked with an asterisk (*) are
   incomplete.

Contents
--------

.. toctree::
    :caption: Theory
    :maxdepth: 1

    theory/introduction
    theory/governing_equations
    theory/linearisation
    theory/mixed_spectral
    theory/numerics
    theory/ffit
    theory/yawed_extension
    theory/notation
    theory/references
    theory/appendices

.. toctree::
    :caption: Examples
    :maxdepth: 1

    examples/01_QuickStart

.. toctree::
    :caption: User Guide
    :maxdepth: 1

    user_guide/installation
    user_guide/usage
    user_guide/configuration
    user_guide/integration
    user_guide/faq

.. toctree::
    :caption: API Reference
    :maxdepth: 1

    api/index

Fuga variants
-------------

Historically, there have been several variants of the Fuga model. PyFuga contains some differences
from the original Fortran implementation:

PreLUT generation
^^^^^^^^^^^^^^^^^

- Uses `numpy.linalg.qr` for QR decomposition.

  This mirrors the logic of the original code but not its exact floating-point behaviour, meaning
  PreLUTs are **not numerically identical** to legacy outputs.

- Removes an additional `h` term in `get_new_h2` present in the Fortran code, which caused
  unnecessarily small integration steps.

Trafalgar
^^^^^^^^^

- Interpolation changed from linear to cubic.
- An improved wavenumber sampling logic.

History, contributors and funding
---------------------------------

Fuga was originally developed at DTU by Søren Ott and colleagues as a Windows application. It has 
since evolved into a Python implementation (PyFuga), and is being prepared for open-source release 
within the PyWake ecosystem.

Contributors include (in alphabetical order):

- Elvira Caroline Jarmbæk Jacobsen
- Gunner Christian Larsen
- Leonardo Alcayaga
- Mads Mølgaard Pedersen
- Nils Joseph Gaukroger
- Søren Ott

The development of Fuga and PyFuga has benefitted from support by DTU, industrial partners such as 
Equinor, and funding bodies such as The Carbon Trust.

Citing PyFuga
-------------

If you use PyFuga in scientific publications, presentations, or academic work,
please cite both the *Fuga model* and the *PyFuga implementation*.

The references below provide the appropriate theoretical background for the
linearised RANS formulation, the mixed–spectral method, and the original
Fuga development at DTU.

Please also cite the PyFuga software directly (citation entry provided below).

Why cite?
^^^^^^^^^

Citations help the maintainers of Fuga and PyFuga demonstrate scientific impact,
support future funding, and ensure correct attribution to the researchers who
developed the model.

Recommended citations
^^^^^^^^^^^^^^^^^^^^^

**1. The underlying Fuga theory**

The original Fuga model and its mixed-spectral formulation were developed at
DTU Wind Energy. The most relevant references are:

.. code-block:: bibtex

    @TechReport{Ott2011,
        author      = {S{\o}ren Ott and Jacob Berg and Morten Nielsen},
        title       = {Linearised CFD Models for Wakes},
        institution = {Ris{\o} National Laboratory for Sustainable Energy},
        year        = 2011,
        number      = {Ris{\o}--R--1772(EN)},
    }

    @TechReport{Ott2014,
        author      = {S{\o}ren Ott and Morten Nielsen},
        title       = {Developments of the offshore wind turbine wake model Fuga},
        institution = {DTU Wind Energy},
        year        = 2014,
        number      = {E--0048},
    }

(See ``theory/references.rst`` for the full list of foundational papers.)

**2. The PyFuga implementation**

If you use the Python implementation directly (for simulations, optimisation,
or integration into workflows such as PyWake), please cite the software:

.. code-block:: bibtex

    @misc{PyFuga2025,
        title        = {PyFuga: Python implementation of the {FUGA} linearised wake model},
        author       = {Alcayaga Rom{\'a}n, Leonardo and
                        Gaukroger, Nils Joseph and
                        Jacobsen, Elvira Caroline Jarmb{\ae}k and
                        Larsen, Gunner Christian and
                        Pedersen, Mads M{\o}lgaard and
                        Ott, S{\o}ren},
        year         = {2025},
        institution  = {DTU Wind \& Energy Systems},
        howpublished = {\url{https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga}},
        note         = {Version X.Y, access date: \today}
    }


Acknowledgements
^^^^^^^^^^^^^^^^

Development of PyFuga has been supported by:

- **DTU Wind and Energy Systems**
- **The Carbon Trust**
- **Equinor**, for long-term support of the Fuga model family

If your use of PyFuga relates to a collaborative project with any of these
organisations, please include a suitable acknowledgement.

Version information
^^^^^^^^^^^^^^^^^^^

You may also wish to record the version of PyFuga you used.  
To obtain it programmatically:

.. code-block:: python

   import pyfuga
   print(pyfuga.__version__)


Look-up tables (LUTs)
^^^^^^^^^^^^^^^^^^^^^

PyFuga produces several intermediate look-up tables, referred to as **PreLUTs** and **fLUTs**, 
before generating the final look-up tables (LUTs) used by PyWake. The final LUTs contain 
turbine-induced perturbation fields evaluated during wind-farm simulations.

The intermediate stages are internal and only need to be set up when generating LUTs for a new 
turbine or atmospheric configuration.
