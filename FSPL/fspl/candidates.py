from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    coordinate: tuple[int, int]
    score: float


@dataclass(frozen=True)
class CandidatePools:
    positives: tuple[Candidate, ...]
    negatives: tuple[Candidate, ...]


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / std


def contrast_map(fg_scores: np.ndarray, bg_scores: np.ndarray, lambda_bg: float) -> np.ndarray:
    foreground = np.asarray(fg_scores, dtype=np.float32)
    background = np.asarray(bg_scores, dtype=np.float32)
    if foreground.shape != background.shape or foreground.ndim != 2:
        raise ValueError("foreground and background scores must be same-shape 2D arrays")
    if lambda_bg < 0.0:
        raise ValueError("lambda_bg must be non-negative")
    return _zscore(foreground) - float(lambda_bg) * _zscore(background)


def _ordered_indices(scores: np.ndarray, descending: bool) -> list[int]:
    flat_scores = np.asarray(scores, dtype=np.float32).ravel()
    indices = np.arange(flat_scores.size)
    primary = -flat_scores if descending else flat_scores
    return list(np.lexsort((indices, primary)))


def _candidate(index: int, scores: np.ndarray, width: int, patch_size: int) -> Candidate:
    row, column = divmod(index, width)
    coordinate = (column * patch_size + patch_size // 2, row * patch_size + patch_size // 2)
    return Candidate(coordinate=coordinate, score=float(scores.ravel()[index]))


def _grid_distance(first: int, second: int, width: int) -> float:
    first_row, first_column = divmod(first, width)
    second_row, second_column = divmod(second, width)
    return float(np.hypot(first_row - second_row, first_column - second_column))


def _select_positive_indices(
    scores: np.ndarray,
    *,
    positive_limit: int,
    width: int,
    positive_layout: str,
    min_positive_distance_patches: float,
) -> list[int]:
    ranked = _ordered_indices(scores, descending=True)
    if positive_layout == "topk":
        return ranked[:positive_limit]
    selected: list[int] = []
    for index in ranked:
        if len(selected) >= positive_limit:
            break
        if all(_grid_distance(index, accepted, width) >= min_positive_distance_patches for accepted in selected):
            selected.append(index)
    for index in ranked:
        if len(selected) >= positive_limit:
            break
        if index not in selected:
            selected.append(index)
    return selected


def build_candidate_pools(
    fg_scores: np.ndarray,
    bg_scores: np.ndarray,
    patch_size: int,
    positive_limit: int,
    negative_limit: int,
    lambda_bg: float = 1.0,
    positive_layout: str = "topk",
    min_positive_distance_patches: float = 3.0,
) -> CandidatePools:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if positive_limit < 0 or negative_limit < 0:
        raise ValueError("candidate limits must be non-negative")
    if positive_layout not in {"topk", "spread"}:
        raise ValueError("positive_layout must be 'topk' or 'spread'")
    if min_positive_distance_patches < 0.0:
        raise ValueError("min_positive_distance_patches must be non-negative")

    contrast = contrast_map(fg_scores, bg_scores, lambda_bg)
    negative_evidence = -contrast
    width = contrast.shape[1]
    selected: set[int] = set()
    positives: list[Candidate] = []
    negatives: list[Candidate] = []

    for index in _select_positive_indices(
        contrast,
        positive_limit=positive_limit,
        width=width,
        positive_layout=positive_layout,
        min_positive_distance_patches=min_positive_distance_patches,
    ):
        selected.add(index)
        positives.append(_candidate(index, contrast, width, patch_size))

    for index in _ordered_indices(negative_evidence, descending=True):
        if len(negatives) >= negative_limit:
            break
        if index in selected:
            continue
        selected.add(index)
        negatives.append(_candidate(index, negative_evidence, width, patch_size))

    return CandidatePools(positives=tuple(positives), negatives=tuple(negatives))
