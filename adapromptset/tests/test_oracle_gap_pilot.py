from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_oracle_gap_pilot import (
    _gate_f_policy_decision,
    default_pilot_paths,
    run_oracle_gap_pilot,
    summarize_oracle_gap,
)


class OracleGapPilotTests(unittest.TestCase):
    def test_summary_compares_per_episode_oracle_to_global_best_fixed(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.20, "iou": 0.1, "aji": 0.10, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e0", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.50, "iou": 0.3, "aji": 0.40, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.45, "iou": 0.2, "aji": 0.30, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.30, "iou": 0.1, "aji": 0.20, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
        ]

        summary = summarize_oracle_gap(records, min_episodes=2)

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["best_fixed_action_id"], "b")
        self.assertAlmostEqual(summary["mean_oracle_dice"], 0.475)
        self.assertAlmostEqual(summary["mean_best_fixed_dice"], 0.4)
        self.assertAlmostEqual(summary["dice_oracle_gap"], 0.075)
        self.assertTrue(summary["aji_available"])
        self.assertAlmostEqual(summary["mean_oracle_aji"], 0.35)
        self.assertAlmostEqual(summary["mean_best_fixed_aji"], 0.30)
        self.assertAlmostEqual(summary["aji_oracle_gap"], 0.05)
        self.assertEqual(summary["oracle_action_counts"], {"b": 1, "a": 1})
        self.assertEqual(summary["best_fixed_sam_calls"], 1)
        self.assertAlmostEqual(summary["mean_oracle_sam_calls"], 1.0)
        self.assertAlmostEqual(summary["budget_gaps"]["1"]["dice_oracle_gap"], 0.075)
        self.assertAlmostEqual(summary["budget_gaps"]["1"]["aji_oracle_gap"], 0.05)

    def test_summary_marks_aji_unavailable_when_utility_records_do_not_have_aji(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.20, "iou": 0.1, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e0", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.50, "iou": 0.3, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.45, "iou": 0.2, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.30, "iou": 0.1, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
        ]

        summary = summarize_oracle_gap(records, min_episodes=2)

        self.assertFalse(summary["aji_available"])
        self.assertIsNone(summary["mean_oracle_aji"])
        self.assertIsNone(summary["mean_best_fixed_aji"])
        self.assertIsNone(summary["aji_oracle_gap"])

    def test_summary_flags_dominant_single_oracle_action(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.5, "iou": 0.3, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e0", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.1, "iou": 0.1, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.6, "iou": 0.4, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.2, "iou": 0.1, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
        ]

        summary = summarize_oracle_gap(records, min_episodes=2, max_top_oracle_fraction=0.7)

        self.assertEqual(summary["gate_f_status"], "FAIL_SINGLE_ACTION_DOMINATES")
        self.assertEqual(summary["top_oracle_action_fraction"], 1.0)

    def test_default_gate_f_threshold_is_one_dice_point(self) -> None:
        records = [
            {"episode_id": "e0", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.40, "iou": 0.2, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e0", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.43, "iou": 0.2, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "a", "action_family": "A", "valid": True, "dice": 0.43, "iou": 0.2, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"episode_id": "e1", "action_id": "b", "action_family": "B", "valid": True, "dice": 0.40, "iou": 0.2, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
        ]

        summary = summarize_oracle_gap(records, min_episodes=2)

        self.assertAlmostEqual(summary["dice_oracle_gap"], 0.015)
        self.assertEqual(summary["gate_f_status"], "PASS_SUBSET_ORACLE_GAP")

    def test_policy_decision_pivots_after_rescue_bank_failure(self) -> None:
        summary = {
            "gate_f_status": "FAIL_ORACLE_GAP_SMALL",
            "action_bank": r"D:\SAM\adapromptset\configs\actions\prompt_set_bank_v2.json",
        }
        decision, note = _gate_f_policy_decision(summary)
        self.assertEqual(decision, "PIVOT_F_REQUIRED")
        self.assertIn("rescue action bank already failed", note)

    def test_default_pilot_paths_route_glass_to_separate_reports(self) -> None:
        paths = default_pilot_paths(PROJECT_ROOT, dataset_slug="glas")

        self.assertEqual(paths["episodes_csv"], PROJECT_ROOT / "manifests" / "episodes" / "glas" / "episodes_val.csv")
        self.assertEqual(paths["candidates_csv"], PROJECT_ROOT / "reports" / "candidate_pilot_glas" / "candidate_pilot_candidates.csv")
        self.assertEqual(paths["out_dir"], PROJECT_ROOT / "reports" / "oracle_gap_pilot_glas")
        self.assertEqual(paths["gate_status"], PROJECT_ROOT / "reports" / "gate_f_status_glas.md")

    def test_run_oracle_gap_pilot_writes_dataset_specific_gate_status(self) -> None:
        out_dir = PROJECT_ROOT / ".test_tmp" / "oracle_gap_glas_mock"
        gate_status = PROJECT_ROOT / "reports" / "gate_f_status_glas.md"
        if gate_status.exists():
            gate_status.unlink()
        fake_predictor = Mock()
        fake_predictor.set_image.return_value = None
        record = {
            "episode_id": "e0",
            "action_id": "a",
            "action_family": "A",
            "valid": True,
            "dice": 0.5,
            "iou": 0.3,
            "sam_calls": 1,
            "decoder_calls": 1,
            "cache_hits": 0,
        }

        with patch("adapromptset.scripts.run_oracle_gap_pilot._read_episode_ids", return_value=["e0"]), \
            patch("adapromptset.scripts.run_oracle_gap_pilot._read_episode", return_value={"episode_id": "e0"}), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.resolve_episode_paths", return_value={"query_image": Path("q.png"), "query_mask": Path("m.png")}), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.load_rgb", return_value=[[[0, 0, 0]]]), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.load_mask", return_value=[[True]]), \
            patch("adapromptset.scripts.run_oracle_gap_pilot._candidate_pools_from_csv", return_value=Mock()), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.candidate_coordinates", return_value=[]), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.load_actions_from_manifest", return_value=("a",)), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.select_valid_actions", return_value=("a",)), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.load_fusion_config", return_value=Mock()), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.load_sam_predictor", return_value=fake_predictor), \
            patch("adapromptset.scripts.run_oracle_gap_pilot.execute_action", return_value=record):
            result = run_oracle_gap_pilot(
                project_root=PROJECT_ROOT,
                dataset_slug="glas",
                max_episodes=1,
                out_dir=out_dir,
                device="cpu",
            )

        self.assertEqual(result, 2)
        self.assertTrue(gate_status.exists())
        summary = json.loads((out_dir / "oracle_gap_pilot_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["gate_f_status"], "PENDING_INSUFFICIENT_EPISODES")


if __name__ == "__main__":
    unittest.main()
