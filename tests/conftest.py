"""pytest configuration. Adds project root to sys.path so tests can import top-level modules."""
import sys
from pathlib import Path

# tests/ -> project root is one level up
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
