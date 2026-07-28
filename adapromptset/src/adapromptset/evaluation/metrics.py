from __future__ import annotations

from typing import Sequence


BoolMask = Sequence[Sequence[bool]]
LabelMask = Sequence[Sequence[int]]


def _flat(mask: BoolMask) -> list[bool]:
    if not mask or not mask[0]:
        raise ValueError("mask must be non-empty")
    width = len(mask[0])
    if any(len(row) != width for row in mask):
        raise ValueError("mask must be rectangular")
    return [bool(value) for row in mask for value in row]


def dice_score(prediction: BoolMask, target: BoolMask) -> float:
    pred = _flat(prediction)
    tgt = _flat(target)
    if len(pred) != len(tgt):
        raise ValueError("prediction and target must have the same size")
    pred_sum = sum(pred)
    tgt_sum = sum(tgt)
    if pred_sum + tgt_sum == 0:
        return 1.0
    intersection = sum(p and t for p, t in zip(pred, tgt))
    return 2.0 * intersection / (pred_sum + tgt_sum)


def iou_score(prediction: BoolMask, target: BoolMask) -> float:
    pred = _flat(prediction)
    tgt = _flat(target)
    if len(pred) != len(tgt):
        raise ValueError("prediction and target must have the same size")
    union = sum(p or t for p, t in zip(pred, tgt))
    if union == 0:
        return 1.0
    intersection = sum(p and t for p, t in zip(pred, tgt))
    return intersection / union


def _flat_labels(mask: LabelMask) -> list[int]:
    if not mask or not mask[0]:
        raise ValueError("mask must be non-empty")
    width = len(mask[0])
    if any(len(row) != width for row in mask):
        raise ValueError("mask must be rectangular")
    return [int(value) for row in mask for value in row]


def aggregated_jaccard_index(prediction: LabelMask, target: LabelMask) -> float:
    pred = _flat_labels(prediction)
    tgt = _flat_labels(target)
    if len(pred) != len(tgt):
        raise ValueError("prediction and target must have the same size")
    pred_ids = sorted({value for value in pred if value > 0})
    target_ids = sorted({value for value in tgt if value > 0})
    if not pred_ids and not target_ids:
        return 1.0
    pred_pixels = {instance_id: {idx for idx, value in enumerate(pred) if value == instance_id} for instance_id in pred_ids}
    target_pixels = {instance_id: {idx for idx, value in enumerate(tgt) if value == instance_id} for instance_id in target_ids}
    pair_scores: list[tuple[float, int, int, int, int]] = []
    for target_id, target_set in target_pixels.items():
        for pred_id, pred_set in pred_pixels.items():
            intersection = len(target_set & pred_set)
            if intersection == 0:
                continue
            union = len(target_set | pred_set)
            pair_scores.append((intersection / union, intersection, union, target_id, pred_id))
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    intersection_sum = 0
    union_sum = 0
    for _, intersection, union, target_id, pred_id in sorted(pair_scores, reverse=True):
        if target_id in matched_targets or pred_id in matched_predictions:
            continue
        matched_targets.add(target_id)
        matched_predictions.add(pred_id)
        intersection_sum += intersection
        union_sum += union
    for target_id, pixels in target_pixels.items():
        if target_id not in matched_targets:
            union_sum += len(pixels)
    for pred_id, pixels in pred_pixels.items():
        if pred_id not in matched_predictions:
            union_sum += len(pixels)
    if union_sum == 0:
        return 1.0
    return intersection_sum / union_sum
