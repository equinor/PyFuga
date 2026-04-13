import os
import sys
import tempfile
import types
from pathlib import Path

# Force "tests" to refer to *this* directory, not any site-packages package.
pkg = types.ModuleType("tests")
pkg.__path__ = [str(Path(__file__).parent)]
sys.modules["tests"] = pkg


def pytest_configure(config: object) -> None:
    """Give each pytest-xdist worker its own Numba cache dir (reduces flaky cache lock races on Windows CI)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    cache_dir = Path(tempfile.gettempdir()) / "pyfuga-numba-cache" / worker
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
