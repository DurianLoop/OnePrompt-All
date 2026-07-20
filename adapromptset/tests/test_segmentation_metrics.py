from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.evaluation.metrics import aggregated_jaccard_index, dice_score, iou_score


class SegmentationMetricTests(unittest.TestCase):
    def test_dice_and_iou_known_case(self) -> None:
        pred = [
            [True, True, False],
            [False, True, False],
        ]
        target = [
            [True, False, False],
            [False, True, True],
        ]

        self.assertAlmostEqual(dice_score(pred, target), 4 / 6)
        self.assertAlmostEqual(iou_score(pred, target), 2 / 4)

    def test_empty_masks_are_perfectly_equal(self) -> None:
        pred = [[False, False]]
        target = [[False, False]]

        self.assertEqual(dice_score(pred, target), 1.0)
        self.assertEqual(iou_score(pred, target), 1.0)

    def test_aji_known_case_matches_manual_aggregation(self) -> None:
        prediction = [
            [1, 1, 0, 3],
            [1, 0, 2, 3],
            [0, 0, 2, 0],
        ]
        target = [
            [1, 1, 0, 0],
            [1, 0, 2, 2],
            [0, 0, 2, 0],
        ]

        self.assertAlmostEqual(aggregated_jaccard_index(prediction, target), 5 / 8)


if __name__ == "__main__":
    unittest.main()
