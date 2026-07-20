from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from adapromptset.matching.evidence import (
    build_evidence_metadata,
    contrast_scores,
    matching_scores_from_features,
    patch_weights_from_mask,
    score_map_global_features,
)


class MatchingEvidenceTests(unittest.TestCase):
    def test_patch_weights_sum_separately_for_foreground_and_background(self) -> None:
        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        fg = patch_weights_from_mask(mask, patch_size=2, foreground=True)
        bg = patch_weights_from_mask(mask, patch_size=2, foreground=False)

        self.assertAlmostEqual(float(fg.sum()), 1.0)
        self.assertAlmostEqual(float(bg.sum()), 1.0)
        self.assertEqual(fg.argmax().item(), 0)
        self.assertNotEqual(bg.argmax().item(), 0)

    def test_matching_scores_from_features_returns_fg_bg_ctr_maps(self) -> None:
        support_features = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 0.0],
                ]
            ),
            dim=1,
        )
        query_features = support_features.clone()
        fg_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        bg_weights = torch.tensor([0.0, 0.0, 0.0, 1.0])

        maps = matching_scores_from_features(
            support_features=support_features,
            query_features=query_features,
            foreground_weights=fg_weights,
            background_weights=bg_weights,
            grid_size=2,
        )

        self.assertEqual(tuple(maps["s_fg"].shape), (2, 2))
        self.assertEqual(tuple(maps["s_bg"].shape), (2, 2))
        self.assertEqual(tuple(maps["s_ctr"].shape), (2, 2))
        self.assertGreater(float(maps["s_ctr"][0, 0]), float(maps["s_ctr"][1, 1]))

    def test_contrast_scores_matches_fg_minus_bg(self) -> None:
        s_fg = torch.tensor([[2.0, 1.0]])
        s_bg = torch.tensor([[0.5, 2.0]])

        result = contrast_scores(s_fg, s_bg, lambda_bg=0.5)

        self.assertTrue(torch.allclose(result, torch.tensor([[1.75, 0.0]])))

    def test_evidence_metadata_excludes_query_ground_truth(self) -> None:
        metadata = build_evidence_metadata(
            dataset="MoNuSAC",
            episode_id="episode",
            support_case_id="support",
            query_case_id="query",
            support_image_path="support.png",
            support_mask_path="support_mask.png",
            query_image_path="query.png",
            matcher_checkpoint_sha256="abc",
            feature_grid_size=37,
        )

        serialized = repr(metadata)
        self.assertIn("support_mask_path", serialized)
        self.assertNotIn("query_mask", serialized)
        self.assertNotIn("query_gt", serialized)

    def test_score_map_global_features_capture_entropy_margin_and_area(self) -> None:
        scores = torch.tensor(
            [
                [4.0, 1.0],
                [0.0, -1.0],
            ]
        )

        features = score_map_global_features(scores, high_score_quantile=0.75)

        self.assertEqual(features["score_count"], 4)
        self.assertEqual(features["top_index_row"], 0)
        self.assertEqual(features["top_index_col"], 0)
        self.assertAlmostEqual(features["top_score"], 4.0)
        self.assertAlmostEqual(features["second_score"], 1.0)
        self.assertAlmostEqual(features["top_margin"], 3.0)
        self.assertAlmostEqual(features["high_score_area_fraction"], 0.25)
        self.assertGreater(features["matching_entropy"], 0.0)
        self.assertLess(features["matching_entropy"], 1.0)

    def test_score_map_global_features_support_valid_mask(self) -> None:
        scores = torch.tensor(
            [
                [10.0, 1.0],
                [2.0, 3.0],
            ]
        )
        valid_mask = torch.tensor(
            [
                [False, True],
                [True, True],
            ]
        )

        features = score_map_global_features(scores, valid_mask=valid_mask, high_score_threshold=2.5)

        self.assertEqual(features["score_count"], 3)
        self.assertEqual(features["top_index_row"], 1)
        self.assertEqual(features["top_index_col"], 1)
        self.assertAlmostEqual(features["top_score"], 3.0)
        self.assertAlmostEqual(features["top_margin"], 1.0)
        self.assertAlmostEqual(features["high_score_area_fraction"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
