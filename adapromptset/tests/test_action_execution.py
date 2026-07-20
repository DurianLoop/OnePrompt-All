from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.actions.bank import PromptAction, PromptGroup
from adapromptset.fusion.masks import FusionConfig
from adapromptset.utility.action_execution import execute_action


class ActionExecutionTests(unittest.TestCase):
    def test_execute_valid_action_records_metrics_and_budget_calls(self) -> None:
        action = PromptAction(
            action_id="A_top1_one_group",
            family="single_group_positive",
            groups=(PromptGroup((0,)),),
            config={"positive_limit": 1},
        )
        coordinates = [(0, 0)]
        target = [[True, False], [False, False]]

        def segmenter(points: list[tuple[int, int]], labels: list[int]):
            self.assertEqual(points, [(0, 0)])
            self.assertEqual(labels, [1])
            return [[True, False], [False, False]], 0.9

        record = execute_action(
            episode_id="e0",
            action=action,
            coordinates=coordinates,
            target_mask=target,
            segmenter=segmenter,
            fusion_config=FusionConfig(min_area=1, iou_nms_threshold=0.95),
        )

        self.assertTrue(record["valid"])
        self.assertEqual(record["sam_calls"], 1)
        self.assertEqual(record["decoder_calls"], 1)
        self.assertEqual(record["cache_hits"], 0)
        self.assertEqual(record["dice"], 1.0)
        self.assertEqual(record["iou"], 1.0)
        self.assertEqual(record["aji"], 1.0)

    def test_execute_action_reuses_duplicate_group_cache_without_changing_budget_calls(self) -> None:
        action = PromptAction(
            action_id="duplicate_groups",
            family="unit",
            groups=(PromptGroup((0,)), PromptGroup((0,))),
            config={},
        )
        calls = 0

        def segmenter(points: list[tuple[int, int]], labels: list[int]):
            nonlocal calls
            calls += 1
            return [[True]], 0.5

        record = execute_action(
            episode_id="e0",
            action=action,
            coordinates=[(2, 2)],
            target_mask=[[True]],
            segmenter=segmenter,
            fusion_config=FusionConfig(),
        )

        self.assertEqual(record["sam_calls"], 2)
        self.assertEqual(record["decoder_calls"], 1)
        self.assertEqual(record["cache_hits"], 1)
        self.assertEqual(calls, 1)

    def test_execute_invalid_action_returns_invalid_record_without_segmenter_call(self) -> None:
        action = PromptAction(
            action_id="A_top10_one_group",
            family="single_group_positive",
            groups=(),
            config={"positive_limit": 10},
            valid=False,
            invalid_reason="requires_10_positive_candidates",
        )

        def segmenter(points: list[tuple[int, int]], labels: list[int]):
            raise AssertionError("invalid actions must not call the segmenter")

        record = execute_action(
            episode_id="e0",
            action=action,
            coordinates=[],
            target_mask=[[True]],
            segmenter=segmenter,
            fusion_config=FusionConfig(),
        )

        self.assertFalse(record["valid"])
        self.assertEqual(record["invalid_reason"], "requires_10_positive_candidates")
        self.assertEqual(record["sam_calls"], 0)
        self.assertIsNone(record["dice"])
        self.assertIsNone(record["aji"])


if __name__ == "__main__":
    unittest.main()
