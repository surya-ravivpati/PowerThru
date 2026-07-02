"""Make the `fatigue` package importable when running scripts directly."""
import sys
from pathlib import Path

# Add the repo root (the directory containing the `fatigue` package) to path.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
