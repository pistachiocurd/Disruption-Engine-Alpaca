"""pytest configuration. Adds project root + subdirs to sys.path so tests can import top-level modules."""
import sys
from pathlib import Path

# tests/ -> project root is one level up
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Post-2026-05-11 reorg: training/, harvesters/, research/path_g/ are subdirs.
# Tests still import the bare module names (train_tcn, fetch_history_*, etc.)
# so put those dirs on sys.path too.
for _sub in ("training", "harvesters", "research/path_g", "research/path_h_l3"):
    p = ROOT / _sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
