from __future__ import annotations

# ruff: noqa: E402
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_candidate_pilot import random_positive_candidates, run_pilot


class RunCandidatePilotTests(unittest.TestCase):
    def test_random_baseline_uses_feature_grid_centers(self) -> None:
        candidates = random_positive_candidates(
            seed_text="episode:5",
            count=5,
            image_size=(518, 518),
            grid_size=37,
        )

        self.assertEqual(len(candidates), 5)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.coordinate[0], 0)
            self.assertLess(candidate.coordinate[0], 518)
            self.assertEqual(candidate.grid_index[0] * 14 + 7, candidate.coordinate[1])
            self.assertEqual(candidate.grid_index[1] * 14 + 7, candidate.coordinate[0])

    def test_dry_run_writes_predeclared_gate_rules(self) -> None:
        out_dir = PROJECT_ROOT / ".test_tmp" / "candidate_pilot_dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_pilot(
            project_root=PROJECT_ROOT,
            out_dir=out_dir,
            dry_run=True,
            max_episodes=3,
            dataset_name="GlaS",
        )

        payload = json.loads((out_dir / "candidate_pilot_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["status"], "DRY_RUN")
        self.assertEqual(payload["formal_gate_c"], "PENDING_EXECUTION")
        self.assertEqual(payload["predeclared_rules"]["min_episodes"], 3)
        self.assertEqual(payload["predeclared_rules"]["dataset_role"], "GlaS validation extension")
        self.assertIn("precision_margin", payload["predeclared_rules"])

    def test_dry_run_records_redesign_candidate_parameters(self) -> None:
        out_dir = PROJECT_ROOT / ".test_tmp" / "candidate_pilot_redesign_dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_pilot(
            project_root=PROJECT_ROOT,
            out_dir=out_dir,
            dry_run=True,
            max_episodes=90,
            dataset_name="GlaS",
            positive_limits=(5, 10, 20),
            positive_layout="local_peaks",
            min_positive_distance=2.0,
        )

        payload = json.loads((out_dir / "candidate_pilot_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["predeclared_rules"]["positive_limits"], [5, 10, 20])
        self.assertEqual(payload["predeclared_rules"]["positive_layout"], "local_peaks")
        self.assertEqual(payload["predeclared_rules"]["min_positive_distance"], 2.0)


if __name__ == "__main__":
    unittest.main()
