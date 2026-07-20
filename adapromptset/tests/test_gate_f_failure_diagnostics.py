from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_gate_f_failure_diagnostics import (
    analyze_gate_f_failure,
    write_gate_f_failure_diagnostics,
)


class GateFFailureDiagnosticsTests(unittest.TestCase):
    def test_analysis_reports_candidate_action_and_sam_sensitivity_diagnostics(self) -> None:
        candidate_rows = [
            {"episode_id": "e0", "positive_limit": 1, "point_precision": 0.20, "component_coverage": 0.02, "random_precision": 0.10},
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.55, "component_coverage": 0.08, "random_precision": 0.10},
            {"episode_id": "e1", "positive_limit": 1, "point_precision": 0.25, "component_coverage": 0.03, "random_precision": 0.12},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.49, "component_coverage": 0.12, "random_precision": 0.12},
        ]
        utility_rows = [
            {"episode_id": "e0", "action_id": "fixed", "action_family": "single", "valid": True, "dice": 0.50, "iou": 0.30, "sam_calls": 1},
            {"episode_id": "e0", "action_id": "split", "action_family": "grouped", "valid": True, "dice": 0.503, "iou": 0.31, "sam_calls": 2},
            {"episode_id": "e0", "action_id": "many", "action_family": "grouped", "valid": True, "dice": 0.490, "iou": 0.29, "sam_calls": 5},
            {"episode_id": "e1", "action_id": "fixed", "action_family": "single", "valid": True, "dice": 0.60, "iou": 0.40, "sam_calls": 1},
            {"episode_id": "e1", "action_id": "split", "action_family": "grouped", "valid": True, "dice": 0.601, "iou": 0.41, "sam_calls": 2},
            {"episode_id": "e1", "action_id": "many", "action_family": "grouped", "valid": True, "dice": 0.580, "iou": 0.39, "sam_calls": 5},
        ]

        summary = analyze_gate_f_failure(candidate_rows, utility_rows)

        self.assertEqual(summary["status"], "PASS_DIAGNOSTIC_ONLY")
        self.assertEqual(summary["policy_training_allowed"], False)
        self.assertEqual(summary["coverage_diagnostic"]["status"], "LOW_COMPONENT_COVERAGE")
        self.assertAlmostEqual(summary["coverage_diagnostic"]["mean_component_coverage_at_max_pool"], 0.10)
        self.assertEqual(summary["proposal_diagnostic"]["near_tie_episode_fraction"], 1.0)
        self.assertLess(summary["proposal_diagnostic"]["mean_top2_dice_margin"], 0.005)
        self.assertEqual(summary["sam_sensitivity_diagnostic"]["status"], "NO_HIGHER_CALL_GAIN")
        self.assertLessEqual(summary["sam_sensitivity_diagnostic"]["best_high_call_minus_best_one_call"], 0.003)
        self.assertIn("candidate_pool_limited_coverage", summary["interpretation_flags"])
        self.assertIn("actions_are_near_tied", summary["interpretation_flags"])

    def test_writer_emits_json_and_markdown_report(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "gate_f_failure_diagnostics_writer"
        if root.exists():
            shutil.rmtree(root)
        try:
            (root / "reports" / "candidate_pilot").mkdir(parents=True)
            (root / "reports" / "oracle_gap_pilot_v2").mkdir(parents=True)
            (root / "reports" / "candidate_pilot" / "candidate_pilot_metrics.csv").write_text(
                "\n".join(
                    [
                        "episode_id,positive_limit,point_precision,component_coverage,random_precision",
                        "e0,5,0.52,0.05,0.20",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv").write_text(
                "\n".join(
                    [
                        "episode_id,action_id,action_family,valid,dice,iou,sam_calls",
                        "e0,fixed,single,True,0.50,0.30,1",
                        "e0,split,grouped,True,0.501,0.31,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_gate_f_failure_diagnostics(root)

            self.assertEqual(summary["status"], "PASS_DIAGNOSTIC_ONLY")
            payload = json.loads((root / "reports" / "gate_f_failure_diagnostics" / "gate_f_failure_diagnostics.json").read_text(encoding="utf-8"))
            text = (root / "reports" / "gate_f_failure_diagnostics" / "gate_f_failure_diagnostics.md").read_text(encoding="utf-8")
            self.assertEqual(payload["status"], "PASS_DIAGNOSTIC_ONLY")
            self.assertIn("Policy training remains prohibited", text)
            self.assertIn("LOW_COMPONENT_COVERAGE", text)
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
