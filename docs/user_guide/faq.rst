FAQ
===

What does ``get_luts()`` do?
----------------------------

:func:`pyfuga.get_luts` is the main user-facing entry point in PyFuga.

It manages the full workflow from user input parameters to:

- PreLUT generation or loading
- Fourier LUT generation or loading
- final LUT assembly

It also stores generated files and returns the final result as an
:class:`xarray.Dataset`.

What is the difference between PreLUTs, fLUTs, and LUTs?
--------------------------------------------------------

PyFuga uses several stages:

- **PreLUTs** are intermediate precomputed quantities
- **fLUTs** are Fourier-space intermediate LUTs
- **LUTs** are the final look-up tables used by PyWake

For most users, these internal stages do not need to be managed manually,
because :func:`pyfuga.get_luts` handles them automatically.

Why does PyFuga write files to disk if it already returns a dataset?
--------------------------------------------------------------------

The files serve two important purposes:

- they provide a persistent record of generated data
- they allow PyFuga to reuse expensive intermediate results in later runs

This makes repeated runs with the same configuration more efficient.

Why is the first run slower than later runs?
--------------------------------------------

The first run for a new configuration may need to generate missing PreLUTs and
Fourier LUTs.

Later runs can often reuse existing files, which makes them much faster.

What does ``jit=True`` do?
--------------------------

It enables just-in-time compilation for selected slow functions.

This can improve performance for LUT generation and is generally recommended
for standard runs.

How should I choose ``nkz0``?
-----------------------------

``nkz0`` controls the wave-number resolution.

Lower values:

- reduce runtime
- reduce file size
- may increase wriggles in the result

Current practical guidance from the Quick Start notebook is:

- ``nkz0=8`` is often a reasonable starting point
- ``nkz0=16`` may be better when accurate speedup effects matter

Why do I see wriggles in the output?
------------------------------------

Wriggles are typically related to resolution choices, especially ``nkz0``.

If wriggles are too large for your use case, try increasing resolution and
comparing results. However, note that very high values do not automatically
guarantee smoother behaviour.

When should I use ``nkz0=8`` and when should I use ``nkz0=16``?
---------------------------------------------------------------

Use ``nkz0=8`` as a practical default for many initial studies.

Consider ``nkz0=16`` when:

- you need better accuracy in speedup-sensitive analyses
- you want a more conservative resolution setting
- you are comparing subtle differences between configurations

What happens when ``zlow=zhigh=zhub``?
--------------------------------------

In that case, PyFuga generates only one horizontal output layer at hub height.

This is useful when only hub-height results are needed.

What does ``lut_vars`` control?
-------------------------------

``lut_vars`` selects which perturbation variables are generated.

Use only the variables you need. This can reduce runtime and output size.

Do I need legacy binary files?
------------------------------

No, not for the standard PyFuga workflow.

Legacy binary or historical file-based inputs are only relevant for older
compatibility workflows and are not required for standard LUT generation.

Which installation method should I choose?
------------------------------------------

Use:

- **PyPI** for standard released installations
- **conda-forge** if you use Conda-based environments
- **Pixi** if you manage environments with Pixi
- **GitLab** if you want the latest repository version directly
- **local source checkout** if you need an editable install or want to contribute

Where are the lower-level classes documented?
---------------------------------------------

Lower-level interfaces such as PreLUT-related and Trafalgar-related classes are
documented in the **API Reference**.

For most users, the main supported workflow is through
:func:`pyfuga.get_luts`.