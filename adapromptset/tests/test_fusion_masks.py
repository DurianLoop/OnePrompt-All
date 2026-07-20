from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.fusion.masks import (
    FusionConfig,
    MaskCandidate,
    filter_masks_by_area,
    load_fusion_config,
    mask_iou,
    nms_masks,
    semantic_union,
)


class FusionMaskTests(unittest.TestCase):
    def test_filter_masks_by_area_keeps_only_large_enough_masks(self) -> None:
        small = MaskCandidate(mask=[[True, False], [False, False]], score=0.9, source_id="small")
        large = MaskCandidate(mask=[[True, True], [False, True]], score=0.8, source_id="large")

        kept = filter_masks_by_area([small, large], min_area=2)

        self.assertEqual([candidate.source_id for candidate in kept], ["large"])

    def test_mask_iou_and_nms_keep_highest_scored_duplicate(self) -> None:
        lower_score = MaskCandidate(
            mask=[[True, True, False], [False, False, False]],
            score=0.5,
            source_id="lower",
        )
        higher_score = MaskCandidate(
            mask=[[True, True, False], [False, False, False]],
            score=0.9,
            source_id="higher",
        )
        distinct = MaskCandidate(
            mask=[[False, False, False], [False, True, True]],
            score=0.4,
            source_id="distinct",
        )

        self.assertEqual(mask_iou(lower_score.mask, higher_score.mask), 1.0)
        kept = nms_masks([lower_score, higher_score, distinct], iou_threshold=0.5)

        self.assertEqual([candidate.source_id for candidate in kept], ["higher", "distinct"])

    def test_semantic_union_combines_retained_masks(self) -> None:
        masks = [
            MaskCandidate(mask=[[True, False], [False, False]], score=0.9),
            MaskCandidate(mask=[[False, False], [True, True]], score=0.8),
        ]

        self.assertEqual(semantic_union(masks), [[True, False], [True, True]])

    def test_fusion_config_is_hashable_and_stable(self) -> None:
        config = FusionConfig(min_area=2, iou_nms_threshold=0.5)

        self.assertEqual(config.content_hash(), "58592D2E6010C2FBB41F6A931AFFB00CDDDBCD44C77032E1768DC503CCC01ABA")

    def test_load_fusion_config_matches_locked_mvp_file(self) -> None:
        config = load_fusion_config(PROJECT_ROOT / "configs" / "segmenter" / "fusion_mvp.json")

        self.assertEqual(config, FusionConfig(min_area=1, iou_nms_threshold=0.95))
        self.assertEqual(config.content_hash(), "5F15975FBE2BBF0E16B2A06DAF0A6F67F0BF7E50824F97CE6B3B66C0A743A8F4")


if __name__ == "__main__":
    unittest.main()
