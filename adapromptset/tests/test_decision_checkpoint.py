from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.write_decision_checkpoint import (
    build_decision_checkpoint,
    oracle_action_heatmap_rows,
)


class DecisionCheckpointTests(unittest.TestCase):
    def test_rescue_failure_emits_oracle_analysis_pivot(self) -> None:
        v1 = {
            "gate_f_status": "FAIL_ORACLE_GAP_SMALL",
            "dice_oracle_gap": 0.0047,
            "best_fixed_action_id": "B_top5_pos3neg",
            "oracle_action_counts": {"B_top5_pos3neg": 5, "B_top5_pos1neg": 3},
            "episode_count": 10,
        }
        v2 = {
            "gate_f_status": "FAIL_ORACLE_GAP_SMALL",
            "dice_oracle_gap": 0.0050,
            "best_fixed_action_id": "B_top5_pos3neg",
            "oracle_action_counts": {"B_top5_pos3neg": 5, "B_top5_pos1neg": 4, "R_split5_2groups_localneg": 1},
            "episode_count": 10,
        }

        checkpoint = build_decision_checkpoint(v1, v2)

        self.assertEqual(checkpoint["phase5_status"], "PIVOT_ORACLE_ANALYSIS")
        self.assertEqual(checkpoint["prohibited_next_actions"], ["policy_training_on_monusac", "posthoc_monusac_action_redesign"])
        self.assertIn("no meaningful oracle gap after the one allowed redesign", checkpoint["rationale"])

    def test_oracle_action_heatmap_rows_are_sorted_by_episode_and_action(self) -> None:
        records = [
            {"episode_id": "e1", "action_id": "b", "valid": True, "dice": 0.1},
            {"episode_id": "e0", "action_id": "b", "valid": True, "dice": 0.4},
            {"episode_id": "e0", "action_id": "a", "valid": True, "dice": 0.5},
            {"episode_id": "e1", "action_id": "a", "valid": True, "dice": 0.2},
        ]

        rows = oracle_action_heatmap_rows(records)

        self.assertEqual(
            rows,
            [
                {"episode_id": "e0", "action_id": "a", "dice": 0.5, "is_oracle": 1},
                {"episode_id": "e0", "action_id": "b", "dice": 0.4, "is_oracle": 0},
                {"episode_id": "e1", "action_id": "a", "dice": 0.2, "is_oracle": 1},
                {"episode_id": "e1", "action_id": "b", "dice": 0.1, "is_oracle": 0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
