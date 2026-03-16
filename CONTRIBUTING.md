# Contributing to PyFuga

Thank you for contributing to PyFuga.

This document describes the recommended workflow for contributing code,
documentation and improvements.

---

## 1. Development setup

First clone the repo.

```bash
git clone git@gitlab.windenergy.dtu.dk:TOPFARM/cuttingedge/pywake/fuga/pyfuga.git
```

PyFuga uses **Pixi** for reproducible environments across platforms.

Install Pixi:
[https://pixi.sh](https://pixi.sh)

Then run:

```bash
cd pyfuga
pixi install
pixi run pre-commit install
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

## 2. Alternative setup

### Conda

```powershell
conda env create -f environment.yml
conda activate pyfuga
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

### Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

### Useful tasks

We have created a `scripts/dev.py` file which one can run to get commands similar to those available with pixi. Run `python scripts/dev.py --help` for further details.

## 3. Workflow

### 3.1 Create or link an issue

For non-trivial changes:

1. [Create a GitLab issue](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga/-/issues) describing the problem or feature.
2. Discuss approach if needed.
3. Create a branch linked to the issue.

### 3.2 Branch naming

Use descriptive branch names:

```text
feature/<short-description>
fix/<short-description>
refactor/<short-description>
docs/<short-description>
chore/<short-description>
```

If linked to an issue, optionally include the issue number:

```text
feature/123-yaw-lut-interpolation
fix/87-prelut-indexing
```

#### Documentation branches

Branches with `docs/` automatically trigger a documentation preview build in CI.

Use this prefix for documentation-only changes:

```text
docs/update-quickstart
docs/improve-developer-guide
```

## 3.3 Daily workflow

For a detailed Git workflow guide covering sync, conflicts, and troubleshooting, see [Git Workflow Guide](docs/git_workflow.md).

**Quick summary:**

1. Start your session: `git fetch` to check for remote updates
2. If behind the remote: `git rebase @{u}` to sync your branch
3. Make your changes and test locally
4. Before pushing: Ensure formatting and tests pass (see section 4 below)
5. Push: `git push`

For common scenarios (branch behind, conflicts, undoing changes, comparing versions) and how to debug CI failures, see the [Git Workflow Guide](docs/git_workflow.md).

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

```text
feature: add yaw LUT interpolating support
fix: correct PreLUT indexing for negative kz
refactor: simplify ODE integration interface
docs: update QuickStart example
```

Reference issues when relevant:

```text
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

## 10. Releasing to TestPyPI / PyPI (maintainers)

Package publication is automated in CI and runs only on **protected tags**.

### 10.1 One-time GitLab setup

In **Settings → CI/CD → Variables**, create:

- `TWINE_USERNAME` = `__token__`
- `TWINE_PASSWORD` = PyPI API token for `pyfuga`
- `TEST_PYPI_TOKEN` = TestPyPI API token for `pyfuga` (optional, when available)
- Mark it as **Masked** and **Protected**

In **Settings → Repository → Protected tags**, protect your release tag pattern (for example `v*`).

### 10.2 Release flow

The release tag must match `project.version` in `pyproject.toml` (for example tag `v0.1.0` for version `0.1.0`). CI will fail early if they differ.
The conda recipe must also be synced before release: run `pixi run sync-conda-recipe`.
CI enforces this with `python scripts/sync_conda_recipe.py --check` in `build_release_artifacts`.

1. Ensure `build_release_artifacts` passes
2. Create a release tag (for example `v0.1.0`)
3. Push the tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This triggers:

- artifact build + validation (`python -m build`, `twine check`)
- TestPyPI upload job (`publish_testpypi`) on protected tags when `TEST_PYPI_TOKEN` is configured
- PyPI upload job (`publish_pypi`) on protected tags when `TWINE_USERNAME` and `TWINE_PASSWORD` are configured

### 10.3 TestPyPI install check

After a TestPyPI upload, verify install in a clean environment:

```bash
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyfuga==<version>
```

### 10.4 Failure handling

- If upload fails before any file is published, fix the issue and re-run the pipeline
- If upload partially succeeds, do not overwrite files; bump version and publish a new tag
