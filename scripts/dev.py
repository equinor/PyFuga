"""
Developer helper CLI for PyFuga.

This script provides a lightweight, cross-platform interface for common
development tasks without requiring Make, Bash, or platform-specific tooling.
It is intended to work inside any active Python environment (Pixi, conda,
venv, etc.) and mirrors a subset of the Pixi tasks defined in pyproject.toml.

The goals are:

- Provide simple commands for formatting, testing, and documentation.
- Help diagnose common setup issues (missing hooks, wrong environment, etc.).
- Reduce friction for contributors, especially on Windows.
- Avoid duplicating complex build logic — this is a thin convenience layer.

Available commands:

    doctor            Show environment and git hook diagnostics.
    fmt               Run pre-commit on all files.
    test [args]       Run pytest, forwarding any additional arguments.
    docs              Build HTML documentation (requires .[docs]).
    docs-autobuild    Run live-reloading documentation preview.

This script does not manage environments. Contributors are expected to:

    - Install dependencies via Pixi (recommended), or
    - Install development dependencies with:
          python -m pip install -e ".[dev]"

Formatting and linting are enforced by pre-commit and CI.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> int:
    """Run a command from the repo root."""
    print(f"+ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def _capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True).strip()


def _which(exe: str) -> str:
    return shutil.which(exe) or "<not found>"


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def _git_config(key: str) -> str:
    try:
        return _capture(["git", "config", "--get", key])
    except Exception:
        return ""


def _precommit_hook_path() -> Path:
    hooks_path = _git_config("core.hooksPath")
    if hooks_path:
        return (REPO_ROOT / hooks_path / "pre-commit").resolve()
    return (REPO_ROOT / ".git" / "hooks" / "pre-commit").resolve()


def doctor() -> None:
    print("=== PyFuga dev doctor ===")
    print(f"Platform:      {platform.platform()}")
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Executable:    {sys.executable}")
    print(f"Repo root:     {REPO_ROOT}")
    print()

    print("Tools:")
    print(f"  git:         {_which('git')}")
    print(f"  pre-commit:  {_which('pre-commit')}")
    print()

    print("Git:")
    print(f"  In git repo: {_is_git_repo()}")
    if _is_git_repo() and _which("git") != "<not found>":
        print(f"  core.hooksPath: {_git_config('core.hooksPath') or '<default>'}")
        hook = _precommit_hook_path()
        print(f"  pre-commit hook present: {hook.exists()} ({hook})")
    print()

    print("Environment hints:")
    print(f"  VIRTUAL_ENV:  {os.environ.get('VIRTUAL_ENV', '')}")
    print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")
    print()

    print("Common fixes:")
    print("  - Install hooks: pre-commit install")
    print('  - Install dev deps: python -m pip install -e ".[dev]"')
    print("  - Run formatting: pre-commit run --all-files")


def ensure_tool(name: str, install_hint: str) -> None:
    if _which(name) == "<not found>":
        raise SystemExit(f"Missing tool: {name}\n\nFix:\n  {install_hint}")


def cmd_fmt() -> None:
    ensure_tool("pre-commit", 'python -m pip install -e ".[dev]"')
    if not _is_git_repo():
        raise SystemExit("Not a git repository (missing .git). pre-commit needs git metadata.")
    _run(["pre-commit", "run", "--all-files"])


def cmd_test(args: list[str]) -> None:
    # Use python -m pytest so it runs in the active env (conda/venv)
    _run([sys.executable, "-m", "pytest", *args])


def cmd_docs() -> None:
    # Keep it simple: assumes docs deps are installed (.[docs])
    ensure_tool("sphinx-build", 'python -m pip install -e ".[docs]"')

    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"  # Disable Numba JIT for docs build (faster + avoids issues)

    docs_dir = REPO_ROOT / "docs"
    if not docs_dir.exists():
        raise SystemExit("docs/ directory not found.")
    _run(["sphinx-build", "-b", "html", "docs", "docs/_build"], env=env)


def cmd_docs_autobuild() -> None:
    """
    Run sphinx-autobuild for live documentation preview.
    Equivalent to: pixi run docs-autobuild
    """
    ensure_tool(
        "sphinx-autobuild",
        'python -m pip install -e ".[docs]"',
    )

    docs_dir = REPO_ROOT / "docs"
    if not docs_dir.exists():
        raise SystemExit("docs/ directory not found.")

    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"

    _run(
        [
            "sphinx-autobuild",
            "docs",
            "docs/_build",
            "--port",
            "8000",
            "--watch",
            "pyfuga",
            "--open-browser",
        ],
        env=env,
    )


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(
            "Usage: python scripts/dev.py <command> [args]\n\n"
            "Commands:\n"
            "  doctor            Show environment + git hook diagnostics\n"
            "  fmt               Run pre-commit on all files\n"
            "  test [pytest args] Run pytest (passes args through)\n"
            "  docs              Build HTML docs (requires .[docs])\n\n"
            "  docs-autobuild    Run live-reloading documentation preview (requires .[docs])\n"
            "Examples:\n"
            "  python scripts/dev.py doctor\n"
            "  python scripts/dev.py fmt\n"
            "  python scripts/dev.py test -q\n"
            '  python scripts/dev.py test -m "not local"\n'
            "  python scripts/dev.py docs\n"
            "  python scripts/dev.py docs-autobuild\n"
        )
        raise SystemExit(0)

    cmd, *rest = argv[1:]

    if cmd == "doctor":
        doctor()
    elif cmd == "fmt":
        cmd_fmt()
    elif cmd == "test":
        cmd_test(rest)
    elif cmd == "docs":
        cmd_docs()
    elif cmd == "docs-autobuild":
        cmd_docs_autobuild()
    else:
        raise SystemExit(f"Unknown command: {cmd} (try --help)")


if __name__ == "__main__":
    main(sys.argv)
