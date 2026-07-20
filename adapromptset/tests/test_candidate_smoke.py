from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_candidate_smoke import run_smoke


class CandidateSmokeTests(unittest.TestCase):
    def test_smoke_marks_formal_gate_pending(self) -> None:
        out_path = PROJECT_ROOT / ".test_tmp" / "candidate_smoke.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = run_smoke(out_path)

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["synthetic_smoke"], "PASS")
        self.assertEqual(payload["formal_gate_c"], "PENDING_REAL_MATCHER")
        self.assertGreaterEqual(payload["metrics"]["point_precision"], 0.5)
        self.assertGreaterEqual(payload["metrics"]["component_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
