"""Tests for Phase X: nested action bank saturation."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_nested_bank_saturation import (
    A_LARGE,
    A_MEDIUM,
    A_SMALL,
    ActionSpec,
    BankSpec,
    NESTED_BANKS,
    _action_set_for_bank,
    _bank_k_negative_grid,
    _bank_k_positive_grid,
    _bank_saturation,
    _gate_x4_status,
)


class BankSpecTests(unittest.TestCase):
    def test_nested_banks_are_ordered_small_medium_large(self) -> None:
        self.assertEqual(NESTED_BANKS, (A_SMALL, A_MEDIUM, A_LARGE))

    def test_a_small_matches_historical_bank(self) -> None:
        self.assertEqual(A_SMALL.max_positive, 5)
        self.assertEqual(A_SMALL.max_negative, 4)
        self.assertEqual(A_SMALL.max_calls, 3)

    def test_a_medium_limits(self) -> None:
        self.assertEqual(A_MEDIUM.max_positive, 20)
        self.assertEqual(A_MEDIUM.max_negative, 10)
        self.assertEqual(A_MEDIUM.max_calls, 5)

    def test_a_large_limits(self) -> None:
        self.assertEqual(A_LARGE.max_positive, 100)
        self.assertEqual(A_LARGE.max_negative, 30)
        self.assertEqual(A_LARGE.max_calls, 20)

    def test_nesting_is_monotonic(self) -> None:
        # A_small ⊂ A_medium ⊂ A_large (limits are monotonically non-decreasing).
        self.assertLessEqual(A_SMALL.max_positive, A_MEDIUM.max_positive)
        self.assertLessEqual(A_MEDIUM.max_positive, A_LARGE.max_positive)
        self.assertLessEqual(A_SMALL.max_negative, A_MEDIUM.max_negative)
        self.assertLessEqual(A_MEDIUM.max_negative, A_LARGE.max_negative)
        self.assertLessEqual(A_SMALL.max_calls, A_MEDIUM.max_calls)
        self.assertLessEqual(A_MEDIUM.max_calls, A_LARGE.max_calls)


class ActionSetForBankTests(unittest.TestCase):
    def test_a_small_includes_one_group_k5(self) -> None:
        actions = _action_set_for_bank(A_SMALL)
        k5_one_group = [
            a for a in actions
            if a.k_positive == 5 and a.grouping == "G_ONE_GROUP"
        ]
        self.assertGreater(len(k5_one_group), 0)

    def test_a_small_excludes_k20(self) -> None:
        actions = _action_set_for_bank(A_SMALL)
        for a in actions:
            self.assertLessEqual(a.k_positive, A_SMALL.max_positive)

    def test_a_medium_includes_k20_one_group(self) -> None:
        actions = _action_set_for_bank(A_MEDIUM)
        k20 = [
            a for a in actions
            if a.k_positive == 20 and a.grouping == "G_ONE_GROUP"
        ]
        self.assertGreater(len(k20), 0)

    def test_a_large_includes_k100(self) -> None:
        actions = _action_set_for_bank(A_LARGE)
        k100 = [a for a in actions if a.k_positive == 100]
        self.assertGreater(len(k100), 0)

    def test_a_small_excludes_kmeans_20(self) -> None:
        # G_KMEANS_20 has 20 calls > A_small.max_calls=3.
        actions = _action_set_for_bank(A_SMALL)
        for a in actions:
            self.assertNotEqual(a.grouping, "G_KMEANS_20")
            self.assertNotEqual(a.grouping, "G_KMEANS_10")

    def test_a_medium_includes_kmeans_5(self) -> None:
        actions = _action_set_for_bank(A_MEDIUM)
        kmeans_5 = [a for a in actions if a.grouping == "G_KMEANS_5"]
        self.assertGreater(len(kmeans_5), 0)

    def test_a_large_includes_kmeans_20(self) -> None:
        actions = _action_set_for_bank(A_LARGE)
        kmeans_20 = [a for a in actions if a.grouping == "G_KMEANS_20"]
        self.assertGreater(len(kmeans_20), 0)

    def test_n_none_with_k_negative_zero_only(self) -> None:
        # N_NONE should only appear with k_negative=0.
        for bank in NESTED_BANKS:
            actions = _action_set_for_bank(bank)
            for a in actions:
                if a.selector == "N_NONE":
                    self.assertEqual(a.k_negative, 0)

    def test_n_bottomk_with_k_negative_positive_only(self) -> None:
        # N_BOTTOMK should only appear with k_negative > 0.
        for bank in NESTED_BANKS:
            actions = _action_set_for_bank(bank)
            for a in actions:
                if a.selector == "N_BOTTOMK":
                    self.assertGreater(a.k_negative, 0)

    def test_all_actions_have_unique_ids(self) -> None:
        for bank in NESTED_BANKS:
            actions = _action_set_for_bank(bank)
            ids = [a.action_id for a in actions]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in {bank.name}")


class BankKGridTests(unittest.TestCase):
    def test_a_small_k_positive_grid(self) -> None:
        self.assertEqual(_bank_k_positive_grid(A_SMALL), (1, 3, 5))

    def test_a_medium_k_positive_grid(self) -> None:
        self.assertEqual(_bank_k_positive_grid(A_MEDIUM), (5, 10, 20))

    def test_a_large_k_positive_grid(self) -> None:
        self.assertEqual(_bank_k_positive_grid(A_LARGE), (20, 50, 100))

    def test_a_small_k_negative_grid(self) -> None:
        self.assertEqual(_bank_k_negative_grid(A_SMALL), (0, 1, 3))

    def test_grids_within_bank_limits(self) -> None:
        for bank in NESTED_BANKS:
            for k in _bank_k_positive_grid(bank):
                self.assertLessEqual(k, bank.max_positive)
            for k in _bank_k_negative_grid(bank):
                self.assertLessEqual(k, bank.max_negative)


class BankSaturationTests(unittest.TestCase):
    def _record(
        self,
        episode_id: str,
        action_id: str,
        bank: str,
        dice: float,
        sam_calls: int = 1,
        runtime_ms: float = 25.0,
    ) -> dict:
        return {
            "episode_id": episode_id,
            "action_id": action_id,
            "bank": bank,
            "k_positive": 5,
            "k_negative": 0,
            "grouping": "G_ONE_GROUP",
            "selector": "N_NONE",
            "sam_calls": sam_calls,
            "actual_negative_count": 0,
            "dice": dice,
            "iou": 0.3,
            "aji": 0.3,
            "component_recall": 0.9,
            "target_component_count": 3,
            "hit_components": 3,
            "retained_masks": 1,
            "runtime_ms": runtime_ms,
            "status": "OK",
        }

    def test_empty_records_returns_zeros(self) -> None:
        s = _bank_saturation([], "A_small")
        self.assertEqual(s["bank"], "A_small")
        self.assertEqual(s["action_count"], 0)
        self.assertIsNone(s["best_fixed_mean_dice"])
        self.assertIsNone(s["oracle_mean_dice"])
        self.assertIsNone(s["oracle_gap"])

    def test_single_action_best_fixed_equals_oracle(self) -> None:
        records = [
            self._record("ep1", "a1", "A_small", 0.5),
            self._record("ep2", "a1", "A_small", 0.6),
        ]
        s = _bank_saturation(records, "A_small")
        self.assertEqual(s["best_fixed_action_id"], "a1")
        self.assertAlmostEqual(s["best_fixed_mean_dice"], 0.55)
        self.assertAlmostEqual(s["oracle_mean_dice"], 0.55)
        self.assertAlmostEqual(s["oracle_gap"], 0.0)

    def test_two_actions_oracle_exceeds_best_fixed(self) -> None:
        records = [
            self._record("ep1", "a1", "A_small", 0.4),
            self._record("ep1", "a2", "A_small", 0.6),
            self._record("ep2", "a1", "A_small", 0.7),
            self._record("ep2", "a2", "A_small", 0.3),
        ]
        s = _bank_saturation(records, "A_small")
        # a1 mean = (0.4+0.7)/2 = 0.55; a2 mean = (0.6+0.3)/2 = 0.45.
        # best fixed = a1 (0.55). oracle = (0.6 + 0.7)/2 = 0.65.
        self.assertEqual(s["best_fixed_action_id"], "a1")
        self.assertAlmostEqual(s["best_fixed_mean_dice"], 0.55)
        self.assertAlmostEqual(s["oracle_mean_dice"], 0.65)
        self.assertAlmostEqual(s["oracle_gap"], 0.10)

    def test_win_tie_loss(self) -> None:
        records = [
            self._record("ep1", "a1", "A_small", 0.4),
            self._record("ep1", "a2", "A_small", 0.6),  # oracle wins
            self._record("ep2", "a1", "A_small", 0.7),  # fixed wins (oracle = 0.7, tied)
            self._record("ep2", "a2", "A_small", 0.3),
        ]
        s = _bank_saturation(records, "A_small")
        # ep1: oracle=0.6, fixed=0.4 -> win
        # ep2: oracle=0.7, fixed=0.7 -> tie
        self.assertEqual(s["win_tie_loss"]["win"], 1)
        self.assertEqual(s["win_tie_loss"]["tie"], 1)
        self.assertEqual(s["win_tie_loss"]["loss"], 0)


class GateX4StatusTests(unittest.TestCase):
    def test_old_action_bank_inadequate(self) -> None:
        saturations = [
            {"bank": "A_small", "oracle_gap": 0.003, "best_fixed_mean_dice": 0.35, "oracle_mean_dice": 0.353},
            {"bank": "A_medium", "oracle_gap": 0.05, "best_fixed_mean_dice": 0.50, "oracle_mean_dice": 0.55},
            {"bank": "A_large", "oracle_gap": 0.04, "best_fixed_mean_dice": 0.55, "oracle_mean_dice": 0.59},
        ]
        status, _ = _gate_x4_status(saturations)
        # A_large best_fixed (0.55) - A_small best_fixed (0.35) = 0.20 >= 0.05.
        self.assertEqual(status, "OLD_ACTION_BANK_INADEQUATE")

    def test_action_space_saturated(self) -> None:
        saturations = [
            {"bank": "A_small", "oracle_gap": 0.003, "best_fixed_mean_dice": 0.50, "oracle_mean_dice": 0.503},
            {"bank": "A_medium", "oracle_gap": 0.002, "best_fixed_mean_dice": 0.52, "oracle_mean_dice": 0.522},
            {"bank": "A_large", "oracle_gap": 0.001, "best_fixed_mean_dice": 0.53, "oracle_mean_dice": 0.531},
        ]
        status, _ = _gate_x4_status(saturations)
        # fixed_gain = 0.53 - 0.50 = 0.03 < 0.05, A_large gap = 0.001 < 0.005.
        self.assertEqual(status, "ACTION_SPACE_SATURATED")

    def test_action_space_not_saturated(self) -> None:
        saturations = [
            {"bank": "A_small", "oracle_gap": 0.003, "best_fixed_mean_dice": 0.50, "oracle_mean_dice": 0.503},
            {"bank": "A_medium", "oracle_gap": 0.02, "best_fixed_mean_dice": 0.52, "oracle_mean_dice": 0.54},
            {"bank": "A_large", "oracle_gap": 0.02, "best_fixed_mean_dice": 0.53, "oracle_mean_dice": 0.55},
        ]
        status, _ = _gate_x4_status(saturations)
        # fixed_gain = 0.03 < 0.05, A_large gap = 0.02 >= 0.005.
        self.assertEqual(status, "ACTION_SPACE_NOT_SATURATED")

    def test_insufficient_data_no_saturations(self) -> None:
        status, _ = _gate_x4_status([])
        self.assertEqual(status, "ACTION_SPACE_INSUFFICIENT_DATA")

    def test_insufficient_data_missing_small(self) -> None:
        saturations = [
            {"bank": "A_medium", "oracle_gap": 0.02, "best_fixed_mean_dice": 0.52},
            {"bank": "A_large", "oracle_gap": 0.02, "best_fixed_mean_dice": 0.53},
        ]
        status, _ = _gate_x4_status(saturations)
        self.assertEqual(status, "ACTION_SPACE_INSUFFICIENT_DATA")

    def test_insufficient_data_none_gap(self) -> None:
        saturations = [
            {"bank": "A_small", "oracle_gap": None, "best_fixed_mean_dice": 0.50},
            {"bank": "A_large", "oracle_gap": None, "best_fixed_mean_dice": 0.53},
        ]
        status, _ = _gate_x4_status(saturations)
        self.assertEqual(status, "ACTION_SPACE_INSUFFICIENT_DATA")


class ActionSpecValidationTests(unittest.TestCase):
    def test_n_none_with_k_negative_zero_is_valid(self) -> None:
        ActionSpec(
            action_id="test",
            bank="A_small",
            k_positive=5,
            k_negative=0,
            grouping="G_ONE_GROUP",
            selector="N_NONE",
        )

    def test_n_none_with_k_negative_positive_raises(self) -> None:
        with self.assertRaises(ValueError):
            ActionSpec(
                action_id="test",
                bank="A_small",
                k_positive=5,
                k_negative=3,
                grouping="G_ONE_GROUP",
                selector="N_NONE",
            )

    def test_zero_k_positive_raises(self) -> None:
        with self.assertRaises(ValueError):
            ActionSpec(
                action_id="test",
                bank="A_small",
                k_positive=0,
                k_negative=0,
                grouping="G_ONE_GROUP",
                selector="N_NONE",
            )

    def test_negative_k_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            ActionSpec(
                action_id="test",
                bank="A_small",
                k_positive=5,
                k_negative=-1,
                grouping="G_ONE_GROUP",
                selector="N_NONE",
            )

    def test_empty_action_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            ActionSpec(
                action_id="",
                bank="A_small",
                k_positive=5,
                k_negative=0,
                grouping="G_ONE_GROUP",
                selector="N_NONE",
            )


if __name__ == "__main__":
    unittest.main()
