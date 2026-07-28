"""Wrapper for Phase Y Gate Y3 Bank Policy Validation script."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "Matcher"))

from adapromptset.scripts.run_phaseY_gateY3_bank_policy import main

if __name__ == "__main__":
    main()
