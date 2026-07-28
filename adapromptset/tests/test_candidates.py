from __future__ import annotations

# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.candidates.points import (
    Candidate,
    build_candidate_pools,
    component_coverage,
    point_precision,
    validate_candidate_pools,
)


class CandidateTests(unittest.TestCase):
    def test_topk_candidates_use_contrast_and_valid_mask(self) -> None:
        foreground = [
            [0.0, 9.0, 8.0],
            [1.0, 7.0, 2.0],
            [3.0, 4.0, 5.0],
        ]
        background = [
            [9.0, 0.0, 0.0],
            [8.0, 0.0, 7.0],
            [6.0, 5.0, 4.0],
        ]
        valid_mask = [
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ]

        pools = build_candidate_pools(
            foreground,
            background,
            image_size=(30, 30),
            positive_limit=2,
            negative_limit=2,
            valid_mask=valid_mask,
            positive_layout="topk",
        )

        self.assertEqual([item.grid_index for item in pools.positives], [(0, 2), (1, 1)])
        self.assertNotIn((0, 1), [item.grid_index for item in pools.positives])
        self.assertEqual(len({item.grid_index for item in pools.positives + pools.negatives}), 4)

    def test_spread_layout_keeps_positive_points_apart_when_possible(self) -> None:
        foreground = [
            [10.0, 9.0, 8.0, 1.0],
            [9.0, 8.0, 7.0, 1.0],
            [1.0, 1.0, 1.0, 9.0],
            [1.0, 1.0, 1.0, 8.0],
        ]
        background = [[0.0] * 4 for _ in range(4)]

        pools = build_candidate_pools(
            foreground,
            background,
            image_size=(40, 40),
            positive_limit=2,
            negative_limit=0,
            positive_layout="spread",
            min_positive_distance=2.0,
        )

        self.assertEqual(pools.positives[0].grid_index, (0, 0))
        self.assertEqual(pools.positives[1].grid_index, (2, 3))

    def test_local_peak_layout_prefers_separate_score_peaks(self) -> None:
        foreground = [
            [10.0, 9.0, 8.0, 1.0, 1.0],
            [9.0, 8.0, 7.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 9.5, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 8.5, 1.0, 1.0, 8.0],
        ]
        background = [[0.0] * 5 for _ in range(5)]

        pools = build_candidate_pools(
            foreground,
            background,
            image_size=(50, 50),
            positive_limit=4,
            negative_limit=0,
            positive_layout="local_peaks",
            min_positive_distance=1.0,
        )

        self.assertEqual([item.grid_index for item in pools.positives[:3]], [(0, 0), (2, 3), (4, 1)])

    def test_candidate_validation_rejects_out_of_bounds_and_duplicates(self) -> None:
        candidates = [
            Candidate(kind="positive", grid_index=(0, 0), coordinate=(5, 5), score=1.0),
            Candidate(kind="negative", grid_index=(0, 0), coordinate=(5, 5), score=0.5),
            Candidate(kind="positive", grid_index=(1, 1), coordinate=(50, 5), score=0.1),
        ]

        issues = validate_candidate_pools(candidates, image_size=(20, 20))

        self.assertTrue(any("duplicate" in issue for issue in issues))
        self.assertTrue(any("out of bounds" in issue for issue in issues))

    def test_precision_and_component_coverage_use_mask_components(self) -> None:
        mask = [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, False, True, True],
            [False, False, False, True, True],
        ]
        candidates = [
            Candidate(kind="positive", grid_index=(0, 0), coordinate=(0, 0), score=1.0),
            Candidate(kind="positive", grid_index=(2, 3), coordinate=(3, 2), score=0.9),
            Candidate(kind="positive", grid_index=(0, 4), coordinate=(4, 0), score=0.1),
        ]

        self.assertAlmostEqual(point_precision(candidates, mask), 2 / 3)
        self.assertAlmostEqual(component_coverage(candidates, mask), 1.0)


if __name__ == "__main__":
    unittest.main()
