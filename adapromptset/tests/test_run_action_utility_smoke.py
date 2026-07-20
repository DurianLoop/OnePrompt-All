from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.actions.bank import PromptAction, PromptGroup, action_bank_manifest
from adapromptset.scripts.run_action_utility_smoke import (
    load_actions_from_manifest,
    select_valid_actions,
    summarize_utility_records,
)


class RunActionUtilitySmokeTests(unittest.TestCase):
    def test_load_actions_from_manifest_round_trips_prompt_groups(self) -> None:
        work_dir = PROJECT_ROOT / ".test_tmp" / "action_utility_smoke"
        work_dir.mkdir(parents=True, exist_ok=True)
        action = PromptAction(
            action_id="B_top3_pos1neg",
            family="single_group_positive_negative",
            groups=(PromptGroup((0, 1, 2), (5,)),),
            config={"positive_limit": 3, "negative_limit": 1},
        )
        path = work_dir / "bank.json"
        path.write_text(json.dumps(action_bank_manifest("unit", (action,)), indent=2), encoding="utf-8")

        loaded = load_actions_from_manifest(path)

        self.assertEqual(loaded, (action,))

    def test_select_valid_actions_keeps_manifest_order_and_limit(self) -> None:
        actions = (
            PromptAction("a", "family", (PromptGroup((0,)),), {}),
            PromptAction("bad", "family", (), {}, valid=False, invalid_reason="missing"),
            PromptAction("b", "family", (PromptGroup((1,)),), {}),
        )

        selected = select_valid_actions(actions, max_actions=1)

        self.assertEqual([action.action_id for action in selected], ["a"])

    def test_summarize_utility_records_reports_oracle_and_best_fixed(self) -> None:
        records = [
            {"action_id": "a", "valid": True, "dice": 0.2, "iou": 0.1, "sam_calls": 1, "decoder_calls": 1, "cache_hits": 0},
            {"action_id": "b", "valid": True, "dice": 0.5, "iou": 0.3, "sam_calls": 3, "decoder_calls": 3, "cache_hits": 2},
            {"action_id": "bad", "valid": False, "dice": None, "iou": None, "sam_calls": 0, "decoder_calls": 0, "cache_hits": 0},
        ]

        summary = summarize_utility_records(records)

        self.assertEqual(summary["valid_action_count"], 2)
        self.assertEqual(summary["oracle_action_id"], "b")
        self.assertEqual(summary["best_fixed_action_id"], "b")
        self.assertEqual(summary["oracle_dice"], 0.5)
        self.assertEqual(summary["max_sam_calls"], 3)
        self.assertEqual(summary["cache_hits"], 2)


if __name__ == "__main__":
    unittest.main()
