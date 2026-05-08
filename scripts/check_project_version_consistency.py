#!/usr/bin/env python3
"""Fail if tracked release metadata is inconsistent across release files.

Checked files (extend this script when adding new pinned locations):
  - CHANGELOG.md (first ## [x] heading, PEP 440 vs pyproject)
  - conda-recipe/recipe.yaml (tracks latest published PyPI package version + sdist url pattern)
  - docs/conf.py (must load version from pyproject.toml, no hard-coded semver)
  - pixi.lock (path dependency pyfuga canonical version)

Conda requirements/deps vs pyproject are validated separately by scripts/sync_conda_recipe.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import tomllib
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _project_name() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["name"])


def _canonical_version(v: str) -> str:
    return str(Version(v))


def _latest_pypi_version(package_name: str, timeout: float = 20.0) -> str:
    api_url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urlopen(api_url, timeout=timeout) as response:
            payload = json.load(response)
    except URLError as error:
        raise RuntimeError(f"failed to fetch {api_url}: {error}") from error
    latest = str(payload["info"]["version"]).strip()
    if not latest:
        raise RuntimeError(f"PyPI metadata for {package_name!r} had empty info.version")
    return latest


def _recipe_package_version(text: str) -> str:
    context_match = re.search(
        r"^context:\n(?:^  .*\n)*?^  version:\s*(?P<value>.+)$",
        text,
        flags=re.MULTILINE,
    )
    if context_match:
        return context_match.group("value").strip()

    m = re.search(
        r"^package:\n(?:^  .*\n)*?^  version:\s*(?P<value>.+)$",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        raise ValueError("could not parse package.version from conda-recipe/recipe.yaml")
    return m.group("value").strip()


def check_conda_recipe(package_name: str) -> list[str]:
    path = ROOT / "conda-recipe" / "recipe.yaml"
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)
    try:
        expected = _latest_pypi_version(package_name)
    except RuntimeError as exc:
        return [f"{rel}: {exc}"]
    try:
        found = _recipe_package_version(text)
    except ValueError as exc:
        return [f"{rel}: {exc}"]
    if found != expected:
        return [f"{rel}: package version is {found!r}; latest PyPI version is {expected!r}"]

    canon = _canonical_version(expected)
    m = re.search(r"pyfuga-([^/]+)\.tar\.gz", text)
    if not m:
        return [f"{rel}: source url must contain pyfuga-<version>.tar.gz"]
    url_ver = m.group(1)
    if url_ver == "${{ version }}":
        return []
    if url_ver != canon:
        return [
            f"{rel}: sdist name uses {url_ver!r} in url; expected canonical {canon!r} "
            f"(run: pixi run sync-conda-recipe)"
        ]
    return []


def check_changelog(expected: str) -> list[str]:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^## \[([^\]]+)\]", text, flags=re.MULTILINE)
    if not m:
        return [f"{path.relative_to(ROOT)}: no ## [version] section heading found"]
    first = m.group(1).strip()
    if Version(first) != Version(expected):
        return [
            f"{path.relative_to(ROOT)}: first section is [{first}]; "
            f"pyproject.toml version is {expected!r} (PEP 440 unequal)"
        ]
    return []


def check_docs_conf() -> list[str]:
    path = ROOT / "docs" / "conf.py"
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)
    errors: list[str] = []
    if "pyproject.toml" not in text:
        errors.append(f"{rel}: must reference pyproject.toml to set the docs version")
    if '["project"]["version"]' not in text and "['project']['version']" not in text:
        errors.append(f"{rel}: must read project version from pyproject.toml (['project']['version'])")
    if re.search(r"(?m)^version\s*=\s*[\"'][0-9]", text):
        errors.append(f"{rel}: must not hard-code a numeric version string; use pyproject.toml")
    return errors


def check_pixi_lock(expected: str) -> list[str]:
    path = ROOT / "pixi.lock"
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)
    m = re.search(r"- pypi: \./\n  name: pyfuga\n  version: (\S+)", text)
    if not m:
        return [f"{rel}: could not find local path pyfuga package entry"]
    lock_v = m.group(1)
    canon = _canonical_version(expected)
    if lock_v != canon:
        return [
            f"{rel}: locked path pyfuga version is {lock_v!r}; "
            f"expected canonical {canon!r} for pyproject {expected!r} (run: pixi lock)"
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    expected = _pyproject_version()
    package_name = _project_name()
    errors: list[str] = []
    for check in (
        check_conda_recipe(package_name),
        check_changelog(expected),
        check_docs_conf(),
        check_pixi_lock(expected),
    ):
        errors.extend(check)

    if errors:
        print(
            "Project version consistency check failed (pyproject is source of truth, recipe tracks latest PyPI).",
            file=sys.stderr,
        )
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"OK: metadata checks passed (pyproject version {expected!r}; recipe tracks latest PyPI).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
