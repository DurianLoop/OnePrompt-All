from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.actions.bank import (
    PromptAction,
    PromptGroup,
    action_cache_keys,
    action_manifest_record,
    action_bank_manifest,
    action_point_count,
    build_mvp_action_bank,
    build_rescue_action_bank,
    build_prompt_arrays,
    candidate_coordinates,
    sam_call_count,
)
from adapromptset.candidates.points import Candidate, CandidatePools


class ActionBankTests(unittest.TestCase):
    def test_action_hash_is_stable_and_content_based(self) -> None:
        action = PromptAction(
            action_id="top5_pos_neg",
            family="fixed",
            groups=(PromptGroup(positive_indices=(0, 1), negative_indices=(2,)),),
            config={"positive_limit": 2, "negative_limit": 1},
        )

        self.assertEqual(action.content_hash(), action.content_hash())
        self.assertEqual(len(action.content_hash()), 64)

    def test_prompt_arrays_encode_positive_then_negative_labels(self) -> None:
        group = PromptGroup(positive_indices=(1, 0), negative_indices=(2,))
        coordinates = [(10, 10), (20, 20), (30, 30)]

        points, labels = build_prompt_arrays(group, coordinates)

        self.assertEqual(points, [(20, 20), (10, 10), (30, 30)])
        self.assertEqual(labels, [1, 1, 0])

    def test_prompt_group_requires_at_least_one_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            PromptGroup(positive_indices=(), negative_indices=(0,))

    def test_mvp_action_bank_builds_fixed_actions_from_unified_candidate_pool(self) -> None:
        pools = CandidatePools(
            positives=tuple(
                Candidate(kind="positive", grid_index=(0, idx), coordinate=(idx * 10, 5), score=10 - idx)
                for idx in range(5)
            ),
            negatives=tuple(
                Candidate(kind="negative", grid_index=(1, idx), coordinate=(idx * 10, 50), score=idx)
                for idx in range(3)
            ),
        )

        actions = build_mvp_action_bank(pools, topk_limits=(1, 3, 5, 10))
        by_id = {action.action_id: action for action in actions}

        self.assertEqual(
            [action.action_id for action in actions[:4]],
            ["A_top1_one_group", "A_top3_one_group", "A_top5_one_group", "A_top10_one_group"],
        )
        self.assertEqual(by_id["A_top3_one_group"].groups, (PromptGroup(positive_indices=(0, 1, 2)),))
        self.assertEqual(by_id["B_top5_pos3neg"].groups, (PromptGroup(positive_indices=(0, 1, 2, 3, 4), negative_indices=(5, 6, 7)),))
        self.assertEqual(by_id["C_top3_independent"].groups, (PromptGroup((0,)), PromptGroup((1,)), PromptGroup((2,))))
        self.assertFalse(by_id["A_top10_one_group"].valid)
        self.assertEqual(by_id["A_top10_one_group"].invalid_reason, "requires_10_positive_candidates")

    def test_rescue_action_bank_adds_split_groups_and_group_specific_negatives(self) -> None:
        pools = CandidatePools(
            positives=(
                Candidate("positive", (0, 0), (0, 0), 5.0),
                Candidate("positive", (0, 1), (10, 0), 4.0),
                Candidate("positive", (0, 2), (20, 0), 3.0),
                Candidate("positive", (0, 3), (100, 0), 2.0),
                Candidate("positive", (0, 4), (110, 0), 1.0),
            ),
            negatives=(
                Candidate("negative", (1, 0), (1, 1), 0.9),
                Candidate("negative", (1, 1), (19, 1), 0.8),
                Candidate("negative", (1, 2), (109, 1), 0.7),
            ),
        )

        actions = build_rescue_action_bank(pools)
        by_id = {action.action_id: action for action in actions}

        self.assertEqual(
            by_id["R_split5_2groups"].groups,
            (PromptGroup((0, 1, 2)), PromptGroup((3, 4))),
        )
        self.assertEqual(
            by_id["R_split5_2groups_localneg"].groups,
            (
                PromptGroup((0, 1, 2), (5,)),
                PromptGroup((3, 4), (7,)),
            ),
        )
        self.assertEqual(
            by_id["R_top5_independent_localneg"].groups,
            (
                PromptGroup((0,), (5,)),
                PromptGroup((1,), (5,)),
                PromptGroup((2,), (6,)),
                PromptGroup((3,), (7,)),
                PromptGroup((4,), (7,)),
            ),
        )
        self.assertTrue(all(action.valid for action in actions))

    def test_action_cost_points_and_cache_keys_are_content_based(self) -> None:
        action = PromptAction(
            action_id="demo",
            family="unit",
            groups=(
                PromptGroup(positive_indices=(0, 1), negative_indices=(3,)),
                PromptGroup(positive_indices=(2,), negative_indices=(3,)),
            ),
            config={},
        )
        coordinates = [(10, 10), (20, 20), (30, 30), (5, 5)]

        self.assertEqual(sam_call_count(action), 2)
        self.assertEqual(action_point_count(action), 5)
        self.assertEqual(len(set(action_cache_keys(action, coordinates))), 2)

    def test_candidate_coordinates_use_positive_then_negative_order(self) -> None:
        pools = CandidatePools(
            positives=(Candidate("positive", (0, 0), (1, 2), 1.0),),
            negatives=(Candidate("negative", (1, 0), (3, 4), 0.5),),
        )

        self.assertEqual(candidate_coordinates(pools), [(1, 2), (3, 4)])

    def test_action_manifest_records_cost_validity_and_hash(self) -> None:
        action = PromptAction(
            action_id="A_top1_one_group",
            family="single_group_positive",
            groups=(PromptGroup((0,)),),
            config={"positive_limit": 1},
        )

        record = action_manifest_record(action)

        self.assertEqual(record["action_id"], "A_top1_one_group")
        self.assertEqual(record["sam_calls"], 1)
        self.assertEqual(record["num_positive"], 1)
        self.assertEqual(record["num_negative"], 0)
        self.assertTrue(record["valid"])
        self.assertEqual(len(record["content_hash"]), 64)

    def test_action_bank_manifest_has_stable_bank_hash(self) -> None:
        actions = (
            PromptAction("a", "family", (PromptGroup((0,)),), {"x": 1}),
            PromptAction("b", "family", (), {"x": 2}, valid=False, invalid_reason="nope"),
        )

        manifest = action_bank_manifest("unit_bank", actions)

        self.assertEqual(manifest["name"], "unit_bank")
        self.assertEqual(manifest["action_count"], 2)
        self.assertEqual(manifest["valid_action_count"], 1)
        self.assertEqual(len(manifest["bank_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
