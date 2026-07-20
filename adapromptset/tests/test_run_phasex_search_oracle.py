"""Tests for Phase X: GT-guided search oracle (section 13)."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_search_oracle import (
    DEFAULT_B_CALL,
    DEFAULT_B_POINT,
    DEFAULT_K_NEGATIVE_START,
    DEFAULT_K_POSITIVE_START,
    DEFAULT_NEGATIVE_SELECTOR,
    GREEDY_ADD_CANDIDATE_LIMIT,
    MIN_IMPROVEMENT,
    SKIP_GREEDY_SWAP,
    SearchState,
    _gate_x5_status,
    _predict_one,
)


class SearchConfigTests(unittest.TestCase):
    """Verify frozen search configuration constants."""

    def test_default_start_uses_gate_x4_best_fixed(self) -> None:
        """Search must start from the Gate X4 best fixed action."""
        self.assertEqual(DEFAULT_K_POSITIVE_START, 20)
        self.assertEqual(DEFAULT_K_NEGATIVE_START, 1)
        self.assertEqual(DEFAULT_NEGATIVE_SELECTOR, "N_BOTTOMK")

    def test_default_b_point_matches_frozen_grid_max(self) -> None:
        """B_point must allow exploring the full 100-positive pool."""
        self.assertEqual(DEFAULT_B_POINT, 100)

    def test_default_b_call_is_one(self) -> None:
        """B_call=1 because Stage S3 showed G_ONE_GROUP dominates."""
        self.assertEqual(DEFAULT_B_CALL, 1)

    def test_min_improvement_is_positive(self) -> None:
        """Minimum improvement threshold must be positive."""
        self.assertGreater(MIN_IMPROVEMENT, 0.0)

    def test_cpu_optimizations_are_set(self) -> None:
        """CPU-only optimizations must be enabled."""
        self.assertGreater(GREEDY_ADD_CANDIDATE_LIMIT, 0)
        self.assertTrue(SKIP_GREEDY_SWAP)


class SearchStateTests(unittest.TestCase):
    """Verify SearchState dataclass behavior."""

    def test_default_state_is_empty(self) -> None:
        state = SearchState()
        self.assertEqual(state.positive_indices, [])
        self.assertEqual(state.negative_indices, [])
        self.assertEqual(state.dice, 0.0)
        self.assertEqual(state.sam_calls, 0)

    def test_state_is_mutable(self) -> None:
        state = SearchState()
        state.positive_indices.append(5)
        state.dice = 0.5
        state.sam_calls = 10
        self.assertEqual(state.positive_indices, [5])
        self.assertEqual(state.dice, 0.5)
        self.assertEqual(state.sam_calls, 10)


class GateX5StatusTests(unittest.TestCase):
    """Verify Gate X5 status logic."""

    def test_exceeds_bank_oracle(self) -> None:
        """Search - bank >= 0.01 → SEARCH_ORACLE_EXCEEDS_BANK_ORACLE."""
        status, _ = _gate_x5_status(
            search_mean_dice=0.70,
            bank_oracle_mean_dice=0.60,
            best_fixed_mean_dice=0.55,
        )
        self.assertEqual(status, "SEARCH_ORACLE_EXCEEDS_BANK_ORACLE")

    def test_close_to_bank_oracle(self) -> None:
        """Search - bank < 0.01 and search >= 0.4 → SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE."""
        status, _ = _gate_x5_status(
            search_mean_dice=0.60,
            bank_oracle_mean_dice=0.595,
            best_fixed_mean_dice=0.55,
        )
        self.assertEqual(status, "SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE")

    def test_absolute_performance_low(self) -> None:
        """Search < 0.4 and search - bank < 0.01 → SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW."""
        # search=0.35, bank=0.345 → search-bank=0.005 < 0.01, and search < 0.4
        status, _ = _gate_x5_status(
            search_mean_dice=0.35,
            bank_oracle_mean_dice=0.345,
            best_fixed_mean_dice=0.30,
        )
        self.assertEqual(status, "SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW")

    def test_exceeds_takes_priority_over_low_absolute(self) -> None:
        """If search - bank >= 0.01, EXCEEDS takes priority over ABSOLUTE_LOW."""
        # search=0.35, bank=0.30 → search-bank=0.05 >= 0.01, even though search < 0.4
        status, _ = _gate_x5_status(
            search_mean_dice=0.35,
            bank_oracle_mean_dice=0.30,
            best_fixed_mean_dice=0.25,
        )
        self.assertEqual(status, "SEARCH_ORACLE_EXCEEDS_BANK_ORACLE")

    def test_insufficient_data_on_none_search(self) -> None:
        """None search → INSUFFICIENT_DATA."""
        status, _ = _gate_x5_status(
            search_mean_dice=None,
            bank_oracle_mean_dice=0.60,
            best_fixed_mean_dice=0.55,
        )
        self.assertEqual(status, "SEARCH_ORACLE_INSUFFICIENT_DATA")

    def test_insufficient_data_on_none_bank(self) -> None:
        """None bank → INSUFFICIENT_DATA."""
        status, _ = _gate_x5_status(
            search_mean_dice=0.70,
            bank_oracle_mean_dice=None,
            best_fixed_mean_dice=0.55,
        )
        self.assertEqual(status, "SEARCH_ORACLE_INSUFFICIENT_DATA")

    def test_boundary_at_exactly_0_01(self) -> None:
        """Search - bank == 0.01 exactly → EXCEEDS (>= threshold)."""
        status, _ = _gate_x5_status(
            search_mean_dice=0.61,
            bank_oracle_mean_dice=0.60,
            best_fixed_mean_dice=0.55,
        )
        self.assertEqual(status, "SEARCH_ORACLE_EXCEEDS_BANK_ORACLE")


class PredictOneTests(unittest.TestCase):
    """Verify _predict_one input validation."""

    def test_predict_one_raises_on_empty_points(self) -> None:
        """_predict_one must raise ValueError when no points are provided."""
        with self.assertRaises(ValueError):
            _predict_one(predictor=None, positive_points=[], negative_points=[])


if __name__ == "__main__":
    unittest.main()
