# Contributing to PyFuga

Thank you for contributing to PyFuga.

This document describes the recommended workflow for contributing code,
documentation and improvements.

---

## 1. Development setup (Pixi recommended)

PyFuga uses **Pixi** for reproducible environments across platforms.

Install Pixi:
https://pixi.sh

Then run:

```bash
pixi install
pixi run test
```

Useful tasks:

```bash
pixi run fmt
pixi run check-fmt
pixi run test
pixi run build-docs
pixi run ci
```

## 2. Alternative setup (without Pixi)

### Conda (recommended on Windows)

```powershell
conda env create -f environment.yml
conda activate pyfuga
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

### Virtual environment (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

## 3. Workflow

### 3.1 Create or link an issue

For non-trivial changes:
1. Create a GitLab issue [here](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga/-/issues) describing the problem or feature.
2. Discuss approach if needed.
3. Create a branch linked to the issue.

### 3.2 Branch naming

Use descriptive branch names:

```
feature/<short-description>
fix/<short-description>
refactor/<short-description>
docs/<short-description>
chore/<short-description>
```

If linked to an issue, optionally include the issue number:
```
feature/123-yaw-lut-interpolation
fix/87-prelut-indexing
```

#### Documentation branches

Branches with `docs/` automatically trigger a documentation preview build in CI.

Use this prefix for documentation-only changes:
```
docs/update-quickstart
docs/improve-developer-guide
```

## 4. Before opening a Merge Request

Ensure formatting and tests pass locally.

### With Pixi

```bash
pixi run ci
```

### Without Pixi

```bash
python scripts/dev.py ci
```

If formatting is needed:

```bash
python scripts/dev.py fmt
```

## 5. Writing tests

- All new functionality should include appropriate tests.
- Tests should be placed in the `tests/` directory.
- Keep tests deterministic and platform-independent where possible.
- Avoid large temporary files unless necessary.

## 6. Updating documentation

If your change affects:
- public API
- user-facing behaviour
- numerical behaviour
- workflows

Update documentation accordingly.

Build docs locally:

```bash
pixi run build-docs
```

or 

```bash
python scripts/dev.py build-docs
```

If you want live-preview, use:

```bash
pixi run autobuild-docs
```

or

```bash
python scripts/dev.py autobuild-docs
```


Documentation is located in the [`docs/`](docs/) directory.

## 7. Code style

PyFuga uses:
- **Black** (line length 120)
- **Ruff** (linting + import sorting)
- Type hints encouraged

Formatting is enforced in CI.

## 8. Commit messages

Use short, descriptive commit messages:
```
feature: add yaw LUT interpolating support
fix: correct PreLUT indexing for negative kz
refactor: simplify ODE integration interface
docs: update QuickStart example
```

Reference issues when relevant:
```
fix: correct PreLUT indexing

Closes #87.
```

## 9. CI behaviour

CI runs:
- formatting checks
- full test suite
- documentation build (when relevant)

If something passes locally but fails in CI:

```bash
python scripts/dev.py doctor
```

Check:
- active Python version
- environment consistency
- pre-commit hooks installed
