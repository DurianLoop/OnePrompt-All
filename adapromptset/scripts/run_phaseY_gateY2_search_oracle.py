"""Wrapper for Phase Y Gate Y2 Search Oracle Stability script."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "Matcher"))

from adapromptset.scripts.run_phaseY_gateY2_search_oracle import main

if __name__ == "__main__":
    main()
