import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root so autodoc can import pyfuga later
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyFuga"
author = "DTU Wind Energy"
copyright = "2025, DTU Wind Energy"

version = "0.1"
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
    "_build",
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

# Optional, but nice: directory name for each version under _build
# (default is something similar)
smv_outputdir_format = "{ref.name}"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": False,
    "sticky_navigation": True,  # Keep the navigation visible when scrolling
}

# Keep MathJax simple
mathjax3_config = {
    "tex": {"tags": "all", "useLabelIds": True},
}

# -- CI build configuration ---------------------------------------------------

# Don't execute notebooks in CI; they're tested separately via pytest+nbconvert
if os.environ.get("CI", "").lower() in {"1", "true"}:
    nbsphinx_execute = "never"
