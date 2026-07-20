from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from adapromptset.evaluation.metrics import BoolMask, iou_score


@dataclass(frozen=True)
class FusionConfig:
    min_area: int = 1
    iou_nms_threshold: float = 0.95

    def content_hash(self) -> str:
        payload = {
            "min_area": self.min_area,
            "iou_nms_threshold": self.iou_nms_threshold,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


@dataclass(frozen=True)
class MaskCandidate:
    mask: BoolMask
    score: float = 1.0
    source_id: str = ""


def mask_area(mask: BoolMask) -> int:
    return sum(1 for row in mask for value in row if value)


def mask_iou(left: BoolMask, right: BoolMask) -> float:
    return iou_score(left, right)


def filter_masks_by_area(candidates: Sequence[MaskCandidate], *, min_area: int) -> list[MaskCandidate]:
    return [candidate for candidate in candidates if mask_area(candidate.mask) >= min_area]


def nms_masks(candidates: Sequence[MaskCandidate], *, iou_threshold: float) -> list[MaskCandidate]:
    retained: list[MaskCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(mask_iou(candidate.mask, kept.mask) <= iou_threshold for kept in retained):
            retained.append(candidate)
    return retained


def semantic_union(candidates: Sequence[MaskCandidate]) -> list[list[bool]]:
    if not candidates:
        raise ValueError("semantic union requires at least one mask")
    height = len(candidates[0].mask)
    width = len(candidates[0].mask[0])
    union = [[False for _ in range(width)] for _ in range(height)]
    for candidate in candidates:
        if len(candidate.mask) != height or len(candidate.mask[0]) != width:
            raise ValueError("all masks must have the same size")
        for y, row in enumerate(candidate.mask):
            if len(row) != width:
                raise ValueError("mask must be rectangular")
            for x, value in enumerate(row):
                union[y][x] = union[y][x] or bool(value)
    return union


def fuse_masks(candidates: Sequence[MaskCandidate], config: FusionConfig) -> tuple[list[MaskCandidate], list[list[bool]]]:
    area_filtered = filter_masks_by_area(candidates, min_area=config.min_area)
    retained = nms_masks(area_filtered, iou_threshold=config.iou_nms_threshold)
    return retained, semantic_union(retained)


def load_fusion_config(path: Path) -> FusionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FusionConfig(
        min_area=int(payload["min_area"]),
        iou_nms_threshold=float(payload["iou_nms_threshold"]),
    )
