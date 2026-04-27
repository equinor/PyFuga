Configuration
=============

PyFuga is configured primarily through the arguments passed to
:func:`pyfuga.get_luts`.

This page groups the main parameters by purpose and highlights the most
important tradeoffs.

Example configuration
---------------------

.. code-block:: python

   luts = get_luts(
       folder="luts",
       zeta0=0,
       nkz0=8,
       nbeta=32,
       diameter=80,
       zhub=70,
       z0=0.00001,
       zi=400,
       zlow=70,
       zhigh=70,
       lut_vars=["UL"],
       nx=2048,
       ny=512,
       dx=None,
       dy=None,
       jit=True,
   )

Physical and atmospheric parameters
-----------------------------------

``zeta0``
^^^^^^^^^

Atmospheric stability parameter.

This controls the atmospheric regime used when generating LUTs and is one of
the key parameters defining the configuration.

``diameter``
^^^^^^^^^^^^

Wind turbine rotor diameter.

This determines the turbine size associated with the generated LUTs.

``zhub``
^^^^^^^^

Wind turbine hub height.

This is the reference hub height for the generated LUT data.

``z0``
^^^^^^

Surface roughness length.

This controls the roughness used in the atmospheric configuration.

``zi``
^^^^^^

Inversion height.

This sets the atmospheric inversion height used in the model.

Output domain
-------------

``zlow`` and ``zhigh``
^^^^^^^^^^^^^^^^^^^^^^

Lower and upper bounds of the vertical output domain.

If ``zlow == zhigh == zhub``, the output contains only a single vertical
layer at hub height.

This is often sufficient for workflows focused only on hub-height quantities.

Spectral and angular resolution
-------------------------------

``nkz0``
^^^^^^^^

Wave-number resolution parameter.

This is one of the most important configuration choices.

Lower values of ``nkz0``:

- reduce runtime
- reduce PreLUT file size
- may introduce larger numerical wriggles

From the current Quick Start example:

- ``nkz0=8`` appears sufficient for many wake-deficit-oriented studies
- ``nkz0=16`` may be preferable when accurate speedup effects are important

A note from current exploratory testing is that ``nkz0=32`` may also show
wriggles, possibly due to overfitting. This observation should be treated as
practical guidance rather than a strict rule.

``nbeta``
^^^^^^^^^

Number of beta angles.

This controls the angular resolution of the generated LUTs.

Output variables
----------------

``lut_vars``
^^^^^^^^^^^^

List of output variables to generate.

Possible values include combinations of:

- ``"UL"``
- ``"UT"``
- ``"VL"``
- ``"VT"``
- ``"WL"``
- ``"WT"``
- ``"PL"``
- ``"PT"``

For example:

.. code-block:: python

   lut_vars=["UL"]

or

.. code-block:: python

   lut_vars=["UL", "VL", "WL"]

Generate only the variables you need. This can reduce runtime and output size.

Grid definition
---------------

``nx``
^^^^^^

Number of points in the LUT along the ``x`` direction.

Higher values provide finer resolution but increase computational cost and
file size.

``ny``
^^^^^^

Number of points in the LUT along the ``y`` direction.

Only one half of the domain is stored.

``dx`` and ``dy``
^^^^^^^^^^^^^^^^^

Spacing of points along the ``x`` and ``y`` directions.

If set to ``None``, PyFuga determines suitable values internally.

Performance settings
--------------------

``jit``
^^^^^^^

If ``True``, selected slow functions are just-in-time compiled.

This is enabled by default in the Quick Start example and is generally
recommended for production runs.

``n_cpu``
^^^^^^^^^

Number of parallel worker processes used in each LUT pipeline stage (PreLUT
generation, Fourier LUTs, and Trafalgar), when you call
:func:`pyfuga.get_luts`.

For :func:`pyfuga.get_luts`, leaving ``n_cpu=None`` (default) enables stage-aware
defaults:

- ``preluts``: all CPUs
- ``fluts``: all CPUs
- ``trafalgar``: all CPUs

Provide an explicit integer to use a single global worker count across stages.
Large ``nx``/``ny`` and several workers can require substantial RAM.

``n_cpu_stage``
^^^^^^^^^^^^^^^

Optional per-stage override dictionary for advanced tuning. Supported keys are
``"preluts"``, ``"fluts"``, and ``"trafalgar"``.

Each provided key overrides the resolved stage setting for that stage only.
When ``n_cpu`` is omitted/``None``, omitted keys keep the stage-aware defaults.
Precedence is: stage defaults -> global ``n_cpu`` (if provided) ->
``n_cpu_stage`` keys.
A stage may be set to ``None`` to use all CPUs for that stage only.

Example:

.. code-block:: python

   get_luts(
       ...,
       n_cpu=4,
       n_cpu_stage={"preluts": 2, "fluts": 4, "trafalgar": 1},
   )

This runs stage 3 serially to reduce memory pressure while keeping earlier
stages parallel.

Practical parameter tradeoffs
-----------------------------

When adjusting parameters, the main practical tradeoffs are:

Resolution vs runtime
^^^^^^^^^^^^^^^^^^^^^

Higher resolution usually improves fidelity but increases runtime and storage
requirements.

Storage vs flexibility
^^^^^^^^^^^^^^^^^^^^^^

Generating many variables or high-resolution grids increases the size of
intermediate and final NetCDF files.

Accuracy vs numerical smoothness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters such as ``nkz0`` can affect both accuracy and visible numerical
wriggles.

Recommended starting point
--------------------------

For a first run, use values close to the Quick Start example and adjust only
one or two parameters at a time.

A good workflow is:

1. start with a known working configuration
2. generate LUTs
3. inspect the output
4. increase resolution only where needed