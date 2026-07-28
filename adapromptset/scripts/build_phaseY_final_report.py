"""Wrapper for Phase Y Final Report builder."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "Matcher"))

from adapromptset.scripts.build_phaseY_final_report import main

if __name__ == "__main__":
    main()
