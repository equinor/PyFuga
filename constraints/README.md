# CI Constraints

This folder contains pip constraints files used **only in CI**.

They exist to ensure that, for each Python minor version tested in CI,
we install package versions that:

- have pre-built wheels available (no source builds),
- are compatible with that Python version,
- avoid unexpected resolver behaviour when new upstream releases appear.

## Why this is needed

CI uses `python:X.Y-slim` Docker images and installs dependencies via:

    uv pip install --system ".[tests]"

Slim images do **not** include a compiler toolchain. If pip resolves to a
package version without wheels for the current Python version (e.g. NumPy
on a newly released Python), pip attempts to build from source and fails.

To prevent this, we pin selected packages (currently NumPy) per Python
minor version.

Example failure without constraints:

- Python 3.13
- Resolver selects NumPy 1.26.4
- No wheel exists for cp313
- pip attempts Meson build
- Build fails (no compiler in slim image)

The constraints files prevent pip from selecting incompatible versions.

## Structure

Each file corresponds to one Python minor version tested in CI:

    constraints/
        py310.txt
        py311.txt
        py312.txt
        py313.txt
        py314.txt

The CI job automatically selects the correct file based on the
`PYTHON_VERSION` matrix entry.

## What should go in these files?

Only **minimal pins required for CI stability**.

Currently we constrain:

- `numpy` — because wheel availability differs across Python minors.

Avoid duplicating full dependency lists here. The authoritative dependency
definitions live in `pyproject.toml`.

## When to update these files

Update constraints when:

- Adding support for a new Python minor version
- Dropping support for an old Python minor version
- A dependency breaks CI due to wheel availability
- A new NumPy major/minor becomes stable across all supported Python versions

If a Python version gains stable wheels for newer NumPy versions,
the corresponding constraint can be relaxed.

## Important

These constraints are a **CI stabilisation mechanism**, not a substitute
for proper dependency specification in `pyproject.toml`.

Local development via Pixi does **not** use these constraints.
