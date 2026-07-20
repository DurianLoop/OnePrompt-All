from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.analyze_fixed_oracle_pivot import (
    action_summary_rows,
    analyze_fixed_oracle,
    budget_pareto_rows,
    episode_regret_rows,
)


class FixedOracleAnalysisTests(unittest.TestCase):
    def test_analysis_reports_best_fixed_regret_and_no_deterministic_promotion(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.50, "iou": 0.30, "aji": 0.40, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.51, "iou": 0.31, "aji": 0.45, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.60, "iou": 0.40, "aji": 0.50, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.58, "iou": 0.38, "aji": 0.44, "sam_calls": 2},
        ]

        summary = analyze_fixed_oracle(records, deterministic_tolerance=0.005)

        self.assertEqual(summary["best_fixed_action_id"], "fixed")
        self.assertAlmostEqual(summary["mean_best_fixed_dice"], 0.55)
        self.assertAlmostEqual(summary["mean_oracle_dice"], 0.555)
        self.assertAlmostEqual(summary["mean_regret_to_oracle"], 0.005)
        self.assertTrue(summary["aji_available"])
        self.assertAlmostEqual(summary["mean_best_fixed_aji"], 0.45)
        self.assertAlmostEqual(summary["mean_oracle_aji"], 0.475)
        self.assertAlmostEqual(summary["mean_aji_regret_to_oracle"], 0.025)
        self.assertEqual(summary["deterministic_promotion_status"], "NOT_PROMOTED")
        self.assertEqual(summary["recommended_claim"], "oracle_analysis_only")

    def test_episode_regret_rows_mark_ties_with_tolerance(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.500, "iou": 0.30, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.504, "iou": 0.31, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.600, "iou": 0.40, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.590, "iou": 0.38, "sam_calls": 2},
        ]

        rows = episode_regret_rows(records, best_fixed_action_id="fixed", tolerance=0.005)

        self.assertEqual(rows[0]["episode_id"], "e0")
        self.assertEqual(rows[0]["oracle_action_id"], "alt")
        self.assertAlmostEqual(rows[0]["best_fixed_regret"], 0.004)
        self.assertEqual(rows[0]["within_tolerance"], 1)
        self.assertEqual(rows[1]["oracle_action_id"], "fixed")
        self.assertEqual(rows[1]["within_tolerance"], 1)

    def test_budget_pareto_rows_compare_best_fixed_to_oracle_per_budget(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "one", "valid": True, "dice": 0.4, "iou": 0.2, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "b", "action_family": "two", "valid": True, "dice": 0.6, "iou": 0.3, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "a", "action_family": "one", "valid": True, "dice": 0.5, "iou": 0.2, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "b", "action_family": "two", "valid": True, "dice": 0.3, "iou": 0.1, "sam_calls": 2},
        ]

        rows = budget_pareto_rows(records)

        self.assertEqual([row["budget"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["best_fixed_action_id"], "a")
        self.assertAlmostEqual(rows[0]["dice_oracle_gap"], 0.0)
        self.assertEqual(rows[1]["best_fixed_action_id"], "a")
        self.assertAlmostEqual(rows[1]["dice_oracle_gap"], 0.1)

    def test_action_summary_counts_oracle_wins(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "one", "valid": True, "dice": 0.4, "iou": 0.2, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "b", "action_family": "two", "valid": True, "dice": 0.6, "iou": 0.3, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "a", "action_family": "one", "valid": True, "dice": 0.5, "iou": 0.2, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "b", "action_family": "two", "valid": True, "dice": 0.3, "iou": 0.1, "sam_calls": 2},
        ]

        rows = action_summary_rows(records)

        self.assertEqual(rows[0]["action_id"], "a")
        self.assertEqual(rows[0]["oracle_count"], 1)
        self.assertAlmostEqual(rows[0]["mean_dice"], 0.45)
        self.assertEqual(rows[1]["action_id"], "b")
        self.assertEqual(rows[1]["oracle_count"], 1)


if __name__ == "__main__":
    unittest.main()
