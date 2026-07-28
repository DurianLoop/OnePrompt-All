"""Tests for Phase X Stage S2: single-variable K+ scaling."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_single_variable_kpos import (
    _component_labels,
    _component_recall,
    _gate_x3_status,
    _semantic_instance_labels,
)


class ComponentRecallTests(unittest.TestCase):
    def test_component_labels_single_component(self) -> None:
        mask = [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
        labels = _component_labels(mask)
        self.assertEqual(len(set(labels.values())), 1)

    def test_component_labels_two_components(self) -> None:
        mask = [
            [True, True, False, True],
            [True, True, False, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
        labels = _component_labels(mask)
        unique = set(labels.values())
        self.assertEqual(len(unique), 2)

    def test_component_recall_all_hit(self) -> None:
        mask = [
            [True, True, False],
            [True, True, False],
        ]
        labels = _component_labels(mask)
        recall, hit, total = _component_recall(mask, labels)
        self.assertEqual(total, 1)
        self.assertEqual(hit, 1)
        self.assertAlmostEqual(recall, 1.0)

    def test_component_recall_none_hit(self) -> None:
        gt = [
            [True, True, False],
            [True, True, False],
        ]
        pred = [
            [False, False, True],
            [False, False, True],
        ]
        labels = _component_labels(gt)
        recall, hit, total = _component_recall(pred, labels)
        self.assertEqual(total, 1)
        self.assertEqual(hit, 0)
        self.assertAlmostEqual(recall, 0.0)

    def test_component_recall_partial_hit_two_components(self) -> None:
        gt = [
            [True, True, False, True],
            [True, True, False, True],
        ]
        pred = [
            [True, False, False, False],
            [False, False, False, False],
        ]
        labels = _component_labels(gt)
        recall, hit, total = _component_recall(pred, labels)
        self.assertEqual(total, 2)
        self.assertEqual(hit, 1)
        self.assertAlmostEqual(recall, 0.5)

    def test_component_recall_empty_target(self) -> None:
        pred = [[False, False], [False, False]]
        labels: dict[tuple[int, int], int] = {}
        recall, hit, total = _component_recall(pred, labels)
        self.assertEqual(total, 0)
        self.assertEqual(hit, 0)
        self.assertAlmostEqual(recall, 0.0)


class GateX3StatusTests(unittest.TestCase):
    def test_viability_established(self) -> None:
        curve = [
            {"k_positive": 1, "episode_count": 10, "mean_dice": 0.4, "mean_iou": 0.3, "mean_aji": 0.3, "mean_component_recall": 0.5, "mean_runtime_ms": 100.0},
            {"k_positive": 3, "episode_count": 10, "mean_dice": 0.45, "mean_iou": 0.35, "mean_aji": 0.35, "mean_component_recall": 0.55, "mean_runtime_ms": 110.0},
            {"k_positive": 5, "episode_count": 10, "mean_dice": 0.5, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.6, "mean_runtime_ms": 120.0},
            {"k_positive": 10, "episode_count": 10, "mean_dice": 0.55, "mean_iou": 0.45, "mean_aji": 0.45, "mean_component_recall": 0.7, "mean_runtime_ms": 130.0},
            {"k_positive": 20, "episode_count": 10, "mean_dice": 0.58, "mean_iou": 0.48, "mean_aji": 0.48, "mean_component_recall": 0.75, "mean_runtime_ms": 140.0},
            {"k_positive": 30, "episode_count": 10, "mean_dice": 0.6, "mean_iou": 0.5, "mean_aji": 0.5, "mean_component_recall": 0.8, "mean_runtime_ms": 150.0},
            {"k_positive": 50, "episode_count": 10, "mean_dice": 0.62, "mean_iou": 0.52, "mean_aji": 0.52, "mean_component_recall": 0.85, "mean_runtime_ms": 160.0},
            {"k_positive": 75, "episode_count": 10, "mean_dice": 0.64, "mean_iou": 0.54, "mean_aji": 0.54, "mean_component_recall": 0.9, "mean_runtime_ms": 170.0},
            {"k_positive": 100, "episode_count": 10, "mean_dice": 0.65, "mean_iou": 0.55, "mean_aji": 0.55, "mean_component_recall": 0.92, "mean_runtime_ms": 180.0},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_VIABILITY_ESTABLISHED")

    def test_no_gain(self) -> None:
        curve = [
            {"k_positive": 5, "episode_count": 10, "mean_dice": 0.5, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.6, "mean_runtime_ms": 120.0},
            {"k_positive": 100, "episode_count": 10, "mean_dice": 0.505, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.62, "mean_runtime_ms": 180.0},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_NO_GAIN")

    def test_cost_only_gain(self) -> None:
        curve = [
            {"k_positive": 5, "episode_count": 10, "mean_dice": 0.5, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.6, "mean_runtime_ms": 120.0},
            {"k_positive": 100, "episode_count": 10, "mean_dice": 0.505, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.7, "mean_runtime_ms": 180.0},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_COST_ONLY_GAIN")

    def test_absolute_performance_low(self) -> None:
        curve = [
            {"k_positive": 5, "episode_count": 10, "mean_dice": 0.15, "mean_iou": 0.1, "mean_aji": 0.1, "mean_component_recall": 0.3, "mean_runtime_ms": 120.0},
            {"k_positive": 100, "episode_count": 10, "mean_dice": 0.25, "mean_iou": 0.2, "mean_aji": 0.2, "mean_component_recall": 0.5, "mean_runtime_ms": 180.0},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW")

    def test_insufficient_data_missing_k100(self) -> None:
        curve = [
            {"k_positive": 5, "episode_count": 10, "mean_dice": 0.5, "mean_iou": 0.4, "mean_aji": 0.4, "mean_component_recall": 0.6, "mean_runtime_ms": 120.0},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_INSUFFICIENT_DATA")

    def test_insufficient_data_none_dice(self) -> None:
        curve = [
            {"k_positive": 5, "episode_count": 10, "mean_dice": None, "mean_iou": None, "mean_aji": None, "mean_component_recall": None, "mean_runtime_ms": None},
            {"k_positive": 100, "episode_count": 0, "mean_dice": None, "mean_iou": None, "mean_aji": None, "mean_component_recall": None, "mean_runtime_ms": None},
        ]
        status, _ = _gate_x3_status(curve)
        self.assertEqual(status, "DENSE_PROMPT_INSUFFICIENT_DATA")


class SemanticInstanceLabelsTests(unittest.TestCase):
    def test_converts_bool_to_int(self) -> None:
        mask = [[True, False], [False, True]]
        labels = _semantic_instance_labels(mask)
        self.assertEqual(labels, [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
