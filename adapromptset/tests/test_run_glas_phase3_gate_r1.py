from __future__ import annotations

# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_glas_phase3_gate_r1 import _box_iou, _random_box_like, _summary


class RunGlasPhase3GateR1Tests(unittest.TestCase):
    def test_box_iou_known_case(self) -> None:
        self.assertAlmostEqual(_box_iou((0, 0, 9, 9), (5, 5, 14, 14)), 25 / 175)
        self.assertEqual(_box_iou(None, (0, 0, 1, 1)), 0.0)

    def test_random_box_preserves_size(self) -> None:
        box = (10, 20, 29, 49)
        random_box = _random_box_like(box, "seed")
        self.assertIsNotNone(random_box)
        assert random_box is not None
        self.assertEqual(random_box[2] - random_box[0], box[2] - box[0])
        self.assertEqual(random_box[3] - random_box[1], box[3] - box[1])

    def test_summary_applies_box_gate(self) -> None:
        rows = [
            {
                "improvement": 0.2,
                "win": True,
                "foreground_recall": 0.8,
                "empty": False,
                "metric_value": 0.5,
                "control_value": 0.3,
                "precision": 0.6,
                "component_coverage": 0.7,
                "area_ratio": 0.4,
            }
            for _ in range(10)
        ]
        result = _summary(rows, "R03_automatic_union_box")
        self.assertEqual(result["status"], "PASS")

    def test_summary_applies_mask_gate(self) -> None:
        rows = [
            {
                "improvement": 0.1,
                "win": True,
                "foreground_recall": 0.7,
                "empty": False,
                "metric_value": 0.25,
                "control_value": 0.15,
                "precision": 0.5,
                "component_coverage": 0.6,
                "area_ratio": 0.2,
            }
            for _ in range(10)
        ]
        result = _summary(rows, "R00_direct_matching_mask")
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
