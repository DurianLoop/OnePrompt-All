from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.analyze_representation_pivot import (
    episode_representation_rows,
    summarize_representation_pivot,
    write_representation_pivot_analysis,
)


class RepresentationPivotAnalysisTests(unittest.TestCase):
    def test_summary_quantifies_near_ties_and_candidate_alignment(self) -> None:
        utility_records = [
            {"episode_id": "e0", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.50, "iou": 0.30, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.505, "iou": 0.31, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "fixed", "action_family": "baseline", "valid": True, "dice": 0.60, "iou": 0.40, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "alt", "action_family": "rescue", "valid": True, "dice": 0.591, "iou": 0.38, "sam_calls": 2},
        ]
        candidate_rows = [
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.20, "component_coverage": 0.10, "random_precision": 0.05, "negative_blank_fraction": 0.0},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.60, "component_coverage": 0.20, "random_precision": 0.10, "negative_blank_fraction": 0.1},
        ]

        rows = episode_representation_rows(utility_records, candidate_rows, top_margin_tolerance=0.01)
        summary = summarize_representation_pivot(utility_records, candidate_rows, top_margin_tolerance=0.01)

        self.assertEqual([row["episode_id"] for row in rows], ["e0", "e1"])
        self.assertEqual(rows[0]["oracle_action_id"], "alt")
        self.assertEqual(rows[0]["oracle_differs_from_best_fixed"], 1)
        self.assertAlmostEqual(rows[0]["top2_dice_margin"], 0.005)
        self.assertEqual(rows[0]["near_tie_top2"], 1)
        self.assertAlmostEqual(rows[0]["candidate_point_precision_at_5"], 0.20)
        self.assertEqual(summary["best_fixed_action_id"], "fixed")
        self.assertAlmostEqual(summary["mean_regret_to_oracle"], 0.0025)
        self.assertAlmostEqual(summary["mean_top2_dice_margin"], 0.007)
        self.assertAlmostEqual(summary["near_tie_episode_fraction"], 1.0)
        self.assertEqual(summary["representation_claim"], "oracle_analysis_only_insufficient_separation")
        self.assertIn("monusac_policy_training", summary["prohibited_next_actions"])

    def test_writer_emits_json_markdown_and_episode_csv(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "representation_pivot_fixture"
        root.mkdir(parents=True, exist_ok=True)
        for existing in root.rglob("*"):
            if existing.is_file():
                existing.unlink()
        (root / "reports" / "oracle_gap_pilot_v2").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "candidate_pilot").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv").write_text(
            "\n".join(
                [
                    "episode_id,action_id,action_family,valid,dice,iou,sam_calls",
                    "e0,fixed,baseline,True,0.50,0.30,1",
                    "e0,alt,rescue,True,0.505,0.31,2",
                    "e1,fixed,baseline,True,0.60,0.40,1",
                    "e1,alt,rescue,True,0.591,0.38,2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "reports" / "candidate_pilot" / "candidate_pilot_metrics.csv").write_text(
            "\n".join(
                [
                    "episode_id,positive_limit,point_precision,component_coverage,random_precision,negative_blank_fraction",
                    "e0,5,0.20,0.10,0.05,0.0",
                    "e1,5,0.60,0.20,0.10,0.1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = write_representation_pivot_analysis(root)

        self.assertEqual(summary["status"], "PIVOT_REPRESENTATION_ANALYSIS")
        self.assertTrue((root / "reports" / "representation_analysis" / "representation_summary.json").exists())
        self.assertTrue((root / "reports" / "representation_analysis" / "episode_representation.csv").exists())
        report = (root / "reports" / "representation_analysis" / "representation_report.md").read_text(encoding="utf-8")
        self.assertIn("PIVOT_REPRESENTATION_ANALYSIS", report)
        self.assertIn("Prohibited Next Actions", report)


if __name__ == "__main__":
    unittest.main()
