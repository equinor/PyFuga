import sys
import types
from pathlib import Path

# Force "tests" to refer to *this* directory, not any site-packages package.
pkg = types.ModuleType("tests")
pkg.__path__ = [str(Path(__file__).parent)]
sys.modules["tests"] = pkg
