Installation
============

PyFuga can be installed in several ways depending on your workflow.

For most users, installation from PyPI is recommended. Alternative installation
methods are also available for Conda, Pixi, GitLab, and local editable source
checkouts.

Requirements
------------

PyFuga currently supports:

- Python ``>=3.10, <3.15``
- Linux
- Windows

The standard LUT-generation workflow does not require external non-Python
tools or legacy binary files.

Install from PyPI
-----------------

Install the latest released version from PyPI:

.. code-block:: bash

   pip install pyfuga

Install from conda-forge
------------------------

If you use Conda, install PyFuga from conda-forge:

.. code-block:: bash

   conda install conda-forge:pyfuga

Install with Pixi
-----------------

If you use Pixi, add PyFuga to your environment with:

.. code-block:: bash

   pixi add pyfuga

Install from the GitLab repository
----------------------------------

To install the latest version directly from the GitLab repository without
cloning it locally:

.. code-block:: bash

   pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga.git

This is useful if you want the newest development version but do not plan to
edit the source code locally.

Install from a local source checkout
------------------------------------

To clone the repository and install PyFuga from a local checkout:

.. code-block:: bash

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga.git
   cd PyFuga
   pip install -e .

An editable install is recommended if you want to inspect the source, test
changes locally, or contribute to development.

Verify the installation
-----------------------

After installation, verify that PyFuga is available and that the main public
entry point can be imported:

.. code-block:: bash

   python -c "import pyfuga; print(pyfuga.__version__)"
   python -c "from pyfuga import get_luts; print(get_luts)"

If both commands run successfully, PyFuga is installed correctly.

Choosing an installation method
-------------------------------

Use:

- **PyPI** for standard released installations
- **conda-forge** if you manage Python packages with Conda
- **Pixi** if you use Pixi-managed environments
- **GitLab** if you want the latest repository version without a local clone
- **local source checkout** if you want an editable install or plan to contribute

Notes
-----

PyFuga also contains compatibility code for some historical file-based and
binary-input workflows from earlier Fuga implementations. These are not
required for the standard Python workflow and are therefore not part of the
basic installation path.