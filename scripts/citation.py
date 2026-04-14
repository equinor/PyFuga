from __future__ import annotations

import argparse
import importlib.metadata
import sys
import urllib.parse
import webbrowser


def _pyfuga_version() -> str:
    return importlib.metadata.version("pyfuga")


def _zenodo_search_url(query: str) -> str:
    return "https://zenodo.org/search?" + urllib.parse.urlencode({"q": query})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print citation pointers for PyFuga.\n\n"
            "Policy: cite both the underlying Fuga theory and the PyFuga software.\n"
            "For PyFuga itself, use the version-specific Zenodo DOI for the exact release you used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the Zenodo search page in your browser.",
    )
    args = parser.parse_args(argv)

    try:
        version = _pyfuga_version()
    except Exception as e:  # pragma: no cover
        print(f"Could not determine installed PyFuga version: {e}", file=sys.stderr)
        print("If you installed from source, use `git describe --tags --always` instead.", file=sys.stderr)
        return 2

    query = f'PyFuga "{version}"'
    url = _zenodo_search_url(query)

    print("Citing PyFuga")
    print("------------")
    print(f"Installed version: {version}")
    print()
    print("For the PyFuga software citation, use the Zenodo record for this exact version and copy")
    print("the citation text/BibTeX from Zenodo (single source of truth):")
    print(f"  {url}")
    print()
    print("In addition, please cite the underlying Fuga theory (see the documentation 'Citing PyFuga').")

    if args.open:
        webbrowser.open(url)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
