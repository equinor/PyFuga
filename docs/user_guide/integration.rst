Integration
===========

PyFuga is part of the **PyWake** ecosystem and provides the look-up tables
required by PyWake's Fuga wake deficit model.

This page describes how PyFuga fits into larger workflows.

PyFuga within the PyWake ecosystem
----------------------------------

PyFuga is not primarily a wind-farm simulation framework by itself. Its main
role is to generate the LUTs required by PyWake's Fuga-based wake modelling
workflow.

In practice, PyFuga is used to prepare turbine- and atmosphere-specific LUTs,
which are then used downstream in PyWake simulations.

Generating LUTs for PyWake
--------------------------

The standard integration path is:

1. define a turbine and atmospheric configuration in PyFuga
2. generate LUTs with :func:`pyfuga.get_luts`
3. store the resulting files
4. use the final LUTs in PyWake's Fuga wake deficit model

PyFuga therefore acts as the LUT-generation layer for PyWake's Fuga workflow.

Working with Python analysis workflows
--------------------------------------

PyFuga returns its final result as an :class:`xarray.Dataset`.

This makes it easy to integrate with Python-based scientific workflows using:

- xarray
- NumPy
- Matplotlib
- NetCDF-based tools

Because the returned object is an ``xarray.Dataset``, users can:

- inspect dimensions and variables
- slice or interpolate data
- plot data directly
- save or reload results using xarray-compatible workflows

NetCDF-based storage
--------------------

PyFuga stores both intermediate and final data as NetCDF files.

This gives two practical benefits:

- generated data can be reused in later runs
- files can be inspected independently of the Python session that created them

The output folder may contain:

- PreLUT files
- Fourier LUT files
- final LUT files

These files provide a persistent record of the generated configuration and
allow efficient caching of expensive intermediate steps.

Legacy workflows
----------------

PyFuga also contains support for some historical workflows inherited from
earlier Fuga implementations, including older file-based and binary-input
formats.

These are not required for the standard PyFuga-to-PyWake workflow and are
therefore not part of the main user path. They are mainly relevant for:

- compatibility work
- validation against historical outputs
- maintenance of legacy datasets

At this stage, such details are better treated as advanced topics or API-level
material rather than part of the main user workflow.

Where lower-level interfaces belong
-----------------------------------

PyFuga includes lower-level components involved in intermediate generation
steps such as PreLUTs, Fourier LUT generation, and Trafalgar processing.

For now, these are documented primarily in the **API Reference**, while the
main user-facing workflow remains centered on :func:`pyfuga.get_luts`.