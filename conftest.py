import sys
from pathlib import Path

# Ensure the `src` package is importable when running pytest from the repo root.
# The tests import from `src.*`, but there is no installed package or pytest
# pythonpath configuration, so we add `src/` to sys.path here.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
