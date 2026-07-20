from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_candidate_scaling import main


if __name__ == "__main__":
    main()
