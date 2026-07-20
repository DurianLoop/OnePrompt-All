from __future__ import annotations

import time
from typing import Any, Callable

from adapromptset.actions.bank import (
    PromptAction,
    build_prompt_arrays,
    group_cache_key,
    sam_call_count,
)
from adapromptset.evaluation.metrics import BoolMask, aggregated_jaccard_index, dice_score, iou_score
from adapromptset.fusion.masks import FusionConfig, MaskCandidate, fuse_masks


SegmenterFn = Callable[[list[tuple[int, int]], list[int]], tuple[list[list[bool]], float]]


def _semantic_instance_labels(mask: BoolMask) -> list[list[int]]:
    return [[1 if bool(value) else 0 for value in row] for row in mask]


def _invalid_record(episode_id: str, action: PromptAction) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "action_id": action.action_id,
        "action_family": action.family,
        "valid": False,
        "invalid_reason": action.invalid_reason,
        "sam_calls": sam_call_count(action),
        "decoder_calls": 0,
        "cache_hits": 0,
        "num_groups": 0,
        "num_positive": 0,
        "num_negative": 0,
        "dice": None,
        "iou": None,
        "aji": None,
        "retained_masks": 0,
        "runtime_ms": 0.0,
    }


def execute_action(
    *,
    episode_id: str,
    action: PromptAction,
    coordinates: list[tuple[int, int]],
    target_mask: BoolMask,
    segmenter: SegmenterFn,
    fusion_config: FusionConfig,
    cache: dict[str, tuple[list[list[bool]], float]] | None = None,
) -> dict[str, Any]:
    if not action.valid:
        return _invalid_record(episode_id, action)
    cache = cache if cache is not None else {}
    start = time.perf_counter()
    candidates: list[MaskCandidate] = []
    decoder_calls = 0
    cache_hits = 0
    num_positive = 0
    num_negative = 0
    for index, group in enumerate(action.groups):
        num_positive += len(group.positive_indices)
        num_negative += len(group.negative_indices)
        key = group_cache_key(group, coordinates)
        if key in cache:
            mask, score = cache[key]
            cache_hits += 1
        else:
            points, labels = build_prompt_arrays(group, coordinates)
            mask, score = segmenter(points, labels)
            cache[key] = (mask, score)
            decoder_calls += 1
        candidates.append(MaskCandidate(mask=mask, score=score, source_id=f"{action.action_id}:group{index}"))
    retained, semantic = fuse_masks(candidates, fusion_config)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return {
        "episode_id": episode_id,
        "action_id": action.action_id,
        "action_family": action.family,
        "valid": True,
        "invalid_reason": "",
        "sam_calls": sam_call_count(action),
        "decoder_calls": decoder_calls,
        "cache_hits": cache_hits,
        "num_groups": len(action.groups),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "dice": dice_score(semantic, target_mask),
        "iou": iou_score(semantic, target_mask),
        "aji": aggregated_jaccard_index(_semantic_instance_labels(semantic), _semantic_instance_labels(target_mask)),
        "retained_masks": len(retained),
        "runtime_ms": round(runtime_ms, 3),
    }
