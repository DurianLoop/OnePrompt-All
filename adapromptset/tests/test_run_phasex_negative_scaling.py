"""Tests for Phase X Stage S4: negative scaling."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_negative_scaling import (
    DEFAULT_GROUPING,
    DEFAULT_K_NEGATIVE_GRID,
    DEFAULT_K_POSITIVE,
    NEGATIVE_SELECTORS_AVAILABLE,
    _assign_negatives_to_groups,
    _gate_x4_negative_status,
    _select_negatives,
)


class SelectNegativesTests(unittest.TestCase):
    def test_n_none_returns_empty_regardless_of_k(self) -> None:
        negs = [(0, 0), (10, 10), (20, 20)]
        self.assertEqual(_select_negatives("N_NONE", 0, negs), [])
        self.assertEqual(_select_negatives("N_NONE", 3, negs), [])

    def test_k_zero_returns_empty(self) -> None:
        negs = [(0, 0), (10, 10)]
        self.assertEqual(_select_negatives("N_BOTTOMK", 0, negs), [])
        self.assertEqual(_select_negatives("N_HIGH_BACKGROUND", 0, negs), [])

    def test_n_bottomk_returns_first_k(self) -> None:
        negs = [(0, 0), (10, 10), (20, 20), (30, 30)]
        self.assertEqual(_select_negatives("N_BOTTOMK", 2, negs), [(0, 0), (10, 10)])

    def test_n_high_background_returns_last_k(self) -> None:
        negs = [(0, 0), (10, 10), (20, 20), (30, 30)]
        self.assertEqual(
            _select_negatives("N_HIGH_BACKGROUND", 2, negs),
            [(20, 20), (30, 30)],
        )

    def test_k_exceeding_pool_size_clamps(self) -> None:
        negs = [(0, 0), (10, 10)]
        self.assertEqual(_select_negatives("N_BOTTOMK", 5, negs), [(0, 0), (10, 10)])
        self.assertEqual(
            _select_negatives("N_HIGH_BACKGROUND", 5, negs), [(0, 0), (10, 10)]
        )

    def test_unknown_selector_raises(self) -> None:
        with self.assertRaises(ValueError):
            _select_negatives("N_NONSENSE", 1, [(0, 0)])


class AssignNegativesToGroupsTests(unittest.TestCase):
    def test_single_group_gets_all_negatives(self) -> None:
        groups = [(0, 1, 2)]
        positive_coords = [(0, 0), (5, 5), (10, 10)]
        negative_coords = [(100, 100), (200, 200)]
        assigned = _assign_negatives_to_groups(groups, positive_coords, negative_coords)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0], [(100, 100), (200, 200)])

    def test_two_groups_split_by_nearest(self) -> None:
        groups = [(0,), (1,)]
        positive_coords = [(0, 0), (100, 100)]
        negative_coords = [(5, 5), (95, 95)]
        assigned = _assign_negatives_to_groups(groups, positive_coords, negative_coords)
        self.assertEqual(assigned[0], [(5, 5)])
        self.assertEqual(assigned[1], [(95, 95)])

    def test_empty_negatives_returns_empty_per_group(self) -> None:
        groups = [(0,), (1,)]
        positive_coords = [(0, 0), (10, 10)]
        assigned = _assign_negatives_to_groups(groups, positive_coords, [])
        self.assertEqual(assigned, [[], []])

    def test_no_groups_returns_empty_list(self) -> None:
        assigned = _assign_negatives_to_groups([], [], [(0, 0)])
        self.assertEqual(assigned, [])

    def test_tie_breaks_to_first_group(self) -> None:
        # Two positives equidistant from a negative: should go to group 0.
        groups = [(0,), (1,)]
        positive_coords = [(0, 10), (20, 10)]
        negative_coords = [(10, 10)]  # exactly between the two positives
        assigned = _assign_negatives_to_groups(groups, positive_coords, negative_coords)
        self.assertEqual(assigned[0], [(10, 10)])
        self.assertEqual(assigned[1], [])


class GateX4NegativeStatusTests(unittest.TestCase):
    def _row(
        self,
        k_neg: int,
        selector: str,
        mean_dice: float,
    ) -> dict:
        return {
            "k_negative": k_neg,
            "selector": selector,
            "episode_count": 10,
            "mean_dice": mean_dice,
            "mean_iou": 0.3,
            "mean_aji": 0.3,
            "mean_component_recall": 0.9,
            "mean_sam_calls": 1,
            "mean_runtime_ms": 25.0,
            "mean_actual_negative_count": float(k_neg),
            "mean_retained_masks": 1,
        }

    def test_best_beats_none(self) -> None:
        curve = [
            self._row(0, "N_NONE", 0.500),
            self._row(3, "N_BOTTOMK", 0.520),
            self._row(5, "N_HIGH_BACKGROUND", 0.515),
        ]
        status, _ = _gate_x4_negative_status(curve)
        self.assertEqual(status, "NEGATIVE_BEST_BEATS_NONE")

    def test_none_dominates(self) -> None:
        curve = [
            self._row(0, "N_NONE", 0.500),
            self._row(3, "N_BOTTOMK", 0.505),  # within 0.01
            self._row(5, "N_HIGH_BACKGROUND", 0.498),
        ]
        status, _ = _gate_x4_negative_status(curve)
        self.assertEqual(status, "NEGATIVE_NONE_DOMINATES")

    def test_insufficient_data_missing_baseline(self) -> None:
        curve = [
            self._row(3, "N_BOTTOMK", 0.520),
        ]
        status, _ = _gate_x4_negative_status(curve)
        self.assertEqual(status, "NEGATIVE_INSUFFICIENT_DATA")

    def test_insufficient_data_baseline_none_dice(self) -> None:
        curve = [
            {"k_negative": 0, "selector": "N_NONE", "mean_dice": None},
            self._row(3, "N_BOTTOMK", 0.520),
        ]
        status, _ = _gate_x4_negative_status(curve)
        self.assertEqual(status, "NEGATIVE_INSUFFICIENT_DATA")

    def test_insufficient_data_no_k_positive_data(self) -> None:
        curve = [
            self._row(0, "N_NONE", 0.500),
        ]
        status, _ = _gate_x4_negative_status(curve)
        self.assertEqual(status, "NEGATIVE_INSUFFICIENT_DATA")


class DefaultsTests(unittest.TestCase):
    def test_default_k_positive_is_s2_peak(self) -> None:
        self.assertEqual(DEFAULT_K_POSITIVE, 20)

    def test_default_grouping_is_one_group(self) -> None:
        self.assertEqual(DEFAULT_GROUPING, "G_ONE_GROUP")

    def test_default_k_negative_grid_includes_zero(self) -> None:
        self.assertIn(0, DEFAULT_K_NEGATIVE_GRID)

    def test_default_k_negative_grid_excludes_30_and_50(self) -> None:
        # Deferred to keep CPU runtime manageable.
        self.assertNotIn(30, DEFAULT_K_NEGATIVE_GRID)
        self.assertNotIn(50, DEFAULT_K_NEGATIVE_GRID)

    def test_negative_selectors_available_excludes_deferred(self) -> None:
        # Deferred selectors require additional candidate generation logic.
        self.assertNotIn("N_BOUNDARY_HARD", NEGATIVE_SELECTORS_AVAILABLE)
        self.assertNotIn("N_LOCAL_CONFUSION", NEGATIVE_SELECTORS_AVAILABLE)
        self.assertNotIn("N_DIVERSITY", NEGATIVE_SELECTORS_AVAILABLE)

    def test_negative_selectors_available_includes_none(self) -> None:
        self.assertIn("N_NONE", NEGATIVE_SELECTORS_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
