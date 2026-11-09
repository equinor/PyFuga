Usage
=====

For most users, the main entry point to PyFuga is :func:`pyfuga.get_luts`.

This function manages the full LUT-generation workflow:

- reading the user-defined configuration
- generating or loading intermediate PreLUTs
- generating or loading Fourier LUTs (fLUTs)
- assembling the final LUTs used by PyWake

The workflow is split into stages deliberately. PyFuga separates the generation
of PreLUTs, Fourier LUTs, and final LUTs so that intermediate results can be
reused across related configurations.

In particular, once a PreLUT has been created, it can be reused for a broader
range of turbine geometries and boundary-layer heights without recomputing the
earliest stage of the method. This makes the workflow more general and reduces
unnecessary recomputation.

The final LUTs are also designed for reuse. The wake deficits stored in the LUT
files are scaled perturbation fields, which means that the same generated LUTs
can be applied with different power curves, thrust-coefficient curves, and wind
speeds in downstream PyWake simulations.

This staged and scaled design improves both generality and efficiency. If
intermediate files are already available, PyFuga reuses them. Otherwise, it
keeps intermediates in memory and saves only the final NetCDF LUT. Pass
``cache_intermediate=True`` to also store intermediates for later configurations.

Quick start
-----------

A minimal workflow looks like this:

.. code-block:: python

   from pyfuga import get_luts

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

Advanced per-stage parallel tuning
----------------------------------

By default, :func:`pyfuga.get_luts` uses stage-aware workers:

- ``preluts``: all CPUs
- ``fluts``: all CPUs
- ``trafalgar``: all CPUs

You can provide ``n_cpu`` as a global override for all stages, and use
``n_cpu_stage`` to override selected stages:

.. code-block:: python

   luts = get_luts(
       ...,
       n_cpu=4,
       n_cpu_stage={"preluts": 2, "fluts": 4, "trafalgar": 1},
   )

In this example, stage 3 (Trafalgar) runs serially to reduce memory pressure,
while stage 1-2 use 4 workers each from the global ``n_cpu`` value.

The output folder
-----------------

The ``folder`` argument specifies where PyFuga stores all generated data,
including the final file and, with ``cache_intermediate=True``, intermediate files.

For the example above, the folder may contain files such as:

.. code-block:: text

   luts/
   ├── preLUTs_Zeta0=0.00_8_32.nc
   ├── fLUTs_Zeta0=0.00_8_32_D80_zhub70_zi400_z0=0.00001000_z70.0_UL.nc
   └── LUTs_Zeta0=0.00_8_32_D80_zhub70_zi400_z0=0.00001000_z70.0_UL_nx2048_ny512_dx20.0_dy5.0.nc

These correspond to:

- **PreLUTs**: intermediate precomputed quantities
- **fLUTs**: Fourier-space intermediate LUT data
- **LUTs**: the final spatial-domain LUTs used by PyWake

The final LUTs do not correspond to a single fixed operating point. They store
scaled wake perturbation fields that can later be combined with turbine-specific
operating data such as power curves and thrust-coefficient curves.

Returned object
---------------

``get_luts()`` returns an :class:`xarray.Dataset`.

This dataset contains:

- coordinates such as ``z``, ``x``, and ``y``
- perturbation variables such as ``UL``
- scalar quantities such as ``diameter``, ``hubheight``, and ``z0``
- metadata stored as dataset attributes

You can inspect the returned dataset directly:

.. code-block:: python

   luts

Typical output is an ``xarray.Dataset`` with dimensions such as:

- ``z``
- ``x``
- ``y``

and variables such as:

- ``UL``

Inspecting and plotting results
-------------------------------

Because the result is an ``xarray.Dataset``, you can access variables directly
and use xarray plotting methods.

For example, to plot a wake slice:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(16, 4))
   luts.UL.squeeze()[500:600, :100].plot(x="x")
   plt.axis("equal")

To plot a cross-wind profile 5D downstream:

.. code-block:: python

   axes = plt.subplots(1, 2, figsize=(16, 6))[1]
   for ax in axes:
       luts.UL.interp(x=80 * 5).plot(ax=ax)
       ax.grid()

   axes[1].set_ylim([-0.0025, 0.0025])
   axes[0].set_title("Normalised deficit profile 5D downstream")
   axes[1].set_title("Zoom of speedup region")

Reusing cached files
--------------------

PyFuga is designed to reuse previously generated files when possible.

If matching PreLUTs or Fourier LUTs already exist in the output folder, they
are loaded instead of recomputed. This typically makes repeated runs much
faster.

This reuse is not only a cache convenience: it reflects the staged structure of
the method, where earlier intermediate products such as PreLUTs are designed to
support multiple downstream LUT configurations.

As a result:

- the first run for a new configuration is usually slower
- later runs with the same configuration are often faster
- the output folder caches final LUTs and, with ``cache_intermediate=True``,
  reusable intermediate results

Single-layer output
-------------------

If:

.. code-block:: python

   zlow == zhigh == zhub

then the output contains only one vertical layer, located at hub height.

This is a useful configuration for hub-height-only workflows.

Typical workflow in practice
----------------------------

A common PyFuga workflow is:

1. define the turbine and atmospheric configuration
2. generate LUTs with :func:`pyfuga.get_luts`
3. inspect or validate the returned dataset
4. store and reuse the generated files
5. use the final LUTs in downstream PyWake simulations