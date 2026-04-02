import os
import re
import sys
from pathlib import Path

import tomllib

# -- Path setup --------------------------------------------------------------

# Add the project root so autodoc can import pyfuga later
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# -- Project information -----------------------------------------------------

project = "PyFuga"
author = "DTU Wind Energy"
copyright = "2025, DTU Wind Energy"

with (_ROOT / "pyproject.toml").open("rb") as _pyproject_f:
    version = tomllib.load(_pyproject_f)["project"]["version"]
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_dark_mode",
    "sphinx_multiversion",
]

autosummary_generate = True
autodoc_typehints = "description"

bibtex_bibfiles = ["theory/references.bib"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build*",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

source_suffix = ".rst"
master_doc = "index"
language = "en"

# -- sphinx-multiversion configuration ---------------------------------------

# Only build docs for main and tags that start with "v"
smv_branch_whitelist = r"^(main)$"
smv_tag_whitelist = r"^v\d+\.\d+(\.\d+)?$"

# Use remote branches from origin, not only local ones
smv_remote_whitelist = r"^origin$"

# Local `main` and `origin/main` both use ref.name "main", so they would share
# output dir `main` and one would be skipped (warning). Prefer the remote ref
# so the built "main" matches origin after `git fetch` (CI and typical clones).
smv_prefer_remote_refs = True

# Optional, but nice: directory name for each version under _build
# (default is something similar)
smv_outputdir_format = "{ref.name}"

# Git refnames look like refs/tags/v1.0.0 (library default ^tags/.*$ never matches).
smv_released_pattern = r"^refs/tags/"

# Default only for non-multiversion builds; multiversion runs override via _pick_smv_latest_version().
smv_latest_version = "main"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# sphinx_rtd_theme does not render html_sidebars; the version switcher lives in _templates/versions.html
# (included at the end of layout.html — classic bottom-left RTD flyout).

html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": False,
    "sticky_navigation": True,  # Keep the navigation visible when scrolling
}

# Fix equation numbering - eq. numbered through custom.css
# All equations with a label are numbered
html_css_files = ["custom.css"]
mathjax3_config = {
    "tex": {
        "tags": "none",
        "useLabelIds": False,
    }
}

# -- CI build configuration ---------------------------------------------------

# Don't execute notebooks in CI; they're tested separately via pytest+nbconvert
if os.environ.get("CI", "").lower() in {"1", "true"}:
    nbsphinx_execute = "never"


_TAG_NAME = re.compile(r"^v\d+\.\d+(?:\.\d+)?$")


def _smv_tag_sort_key(name: str) -> tuple[int, ...]:
    m = re.match(r"^v(\d+)\.(\d+)(?:\.(\d+))?$", name)
    if not m:
        return (-1, 0, 0, 0, 0)
    t = tuple(int(g or 0) for g in m.groups())
    # Tie-break equal semver tuples (e.g. v0.1 vs v0.1.0) toward the longer canonical tag.
    return (*t, len(name))


def _pick_smv_latest_version(app, config):
    """Set smv_latest_version to the newest semver tag present in the multiversion metadata (for banners / links)."""
    meta = getattr(config, "smv_metadata", None) or {}
    if not meta:
        return
    tag_names = [k for k in meta if _TAG_NAME.match(k)]
    if tag_names:
        config.smv_latest_version = max(tag_names, key=_smv_tag_sort_key)
    elif "main" in meta:
        config.smv_latest_version = "main"


def setup(app):
    # After sphinx_multiversion (priority 500): Sphinx runs config-inited in ascending priority order.
    app.connect("config-inited", _pick_smv_latest_version, priority=501)
