Code style
==========

- We use **Black** for formatting.
    - Default line length: 88 characters.
    - Use `# fmt: off` / `# fmt: on` sparingly for genuinely hard-to-read edge cases.

- We use **Ruff** for linting and import sorting.
    - Configuration is in `pyproject.toml`.
    - Import order follows Ruff's isort rules with `pyfuga` as first-party code.

- Recommended workflow:
    - Install pre-commit: `pip install pre-commit`.
    - Run `pre-commit install` once in your clone.
    - Optionally run `pre-commit run --all-files` before large refactors or before MR2/3.

- Type checking and editor diagnostics still use **Pylance** in VS Code.
