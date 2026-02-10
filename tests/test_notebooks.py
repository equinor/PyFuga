from pathlib import Path

import pytest

OUTPUT_DIR = Path("tests/notebook_outputs")


def run_notebook(notebook_path: Path, allow_errors: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = [
        "--execute",
        "--to",
        "html",
        "--ExecutePreprocessor.timeout=600",
        "--output-dir",
        str(OUTPUT_DIR),
        str(notebook_path),
    ]

    if allow_errors:
        args.insert(0, "--allow-errors")

    from nbconvert.nbconvertapp import NbConvertApp

    app = NbConvertApp()
    app.initialize(args)
    app.start()


@pytest.mark.parametrize("notebook_path", sorted(Path("docs/examples").glob("*.ipynb")))
def test_quickstart_notebook_exec(notebook_path: Path):
    try:
        run_notebook(notebook_path, allow_errors=False)
    except Exception as e:
        # Re-run allowing errors so we still get an .html artefact for debugging
        run_notebook(notebook_path, allow_errors=True)
        raise e
