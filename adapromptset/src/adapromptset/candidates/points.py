from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Grid = Sequence[Sequence[float]]
BoolGrid = Sequence[Sequence[bool]]


@dataclass(frozen=True)
class Candidate:
    kind: str
    grid_index: tuple[int, int]
    coordinate: tuple[int, int]
    score: float


@dataclass(frozen=True)
class CandidatePools:
    positives: tuple[Candidate, ...]
    negatives: tuple[Candidate, ...]


def _shape(grid: Sequence[Sequence[object]]) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("score grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("score grid must be rectangular")
    return len(grid), width


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _zscore_grid(grid: Grid) -> list[list[float]]:
    values = [float(value) for row in grid for value in row]
    mean, std = _mean_std(values)
    if std == 0.0:
        return [[0.0 for _ in row] for row in grid]
    return [[(float(value) - mean) / std for value in row] for row in grid]


def _contrast_grid(foreground: Grid, background: Grid, lambda_bg: float) -> list[list[float]]:
    rows, cols = _shape(foreground)
    if _shape(background) != (rows, cols):
        raise ValueError("foreground and background grids must have the same shape")
    if lambda_bg < 0.0:
        raise ValueError("lambda_bg must be non-negative")
    fg = _zscore_grid(foreground)
    bg = _zscore_grid(background)
    return [[fg[r][c] - lambda_bg * bg[r][c] for c in range(cols)] for r in range(rows)]


def _coordinate(row: int, col: int, rows: int, cols: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    x = min(width - 1, max(0, int((col + 0.5) * width / cols)))
    y = min(height - 1, max(0, int((row + 0.5) * height / rows)))
    return x, y


def _ranked_indices(scores: list[list[float]], valid_mask: BoolGrid | None, descending: bool) -> list[tuple[int, int]]:
    rows, cols = _shape(scores)
    if valid_mask is not None and _shape(valid_mask) != (rows, cols):
        raise ValueError("valid_mask must match score grid shape")
    indices: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if valid_mask is not None and not valid_mask[row][col]:
                continue
            indices.append((row, col))
    return sorted(indices, key=lambda item: ((-1 if descending else 1) * scores[item[0]][item[1]], item[0], item[1]))


def _grid_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _is_local_peak(scores: list[list[float]], row: int, col: int) -> bool:
    rows, cols = _shape(scores)
    value = scores[row][col]
    for nr in range(max(0, row - 1), min(rows, row + 2)):
        for nc in range(max(0, col - 1), min(cols, col + 2)):
            if (nr, nc) != (row, col) and scores[nr][nc] > value:
                return False
    return True


def _local_peak_ranked_indices(
    scores: list[list[float]],
    ranked: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    peaks = [index for index in ranked if _is_local_peak(scores, index[0], index[1])]
    peak_set = set(peaks)
    return peaks + [index for index in ranked if index not in peak_set]


def _select_positives(
    scores: list[list[float]],
    ranked: list[tuple[int, int]],
    *,
    positive_limit: int,
    positive_layout: str,
    min_positive_distance: float,
) -> list[tuple[int, int]]:
    if positive_layout == "topk":
        return ranked[:positive_limit]
    if positive_layout == "local_peaks":
        ranked = _local_peak_ranked_indices(scores, ranked)
    elif positive_layout != "spread":
        raise ValueError("positive_layout must be 'topk', 'spread', or 'local_peaks'")
    selected: list[tuple[int, int]] = []
    for index in ranked:
        if len(selected) >= positive_limit:
            break
        if all(_grid_distance(index, accepted) >= min_positive_distance for accepted in selected):
            selected.append(index)
    for index in ranked:
        if len(selected) >= positive_limit:
            break
        if index not in selected:
            selected.append(index)
    return selected


def build_candidate_pools(
    foreground: Grid,
    background: Grid,
    *,
    image_size: tuple[int, int],
    positive_limit: int,
    negative_limit: int,
    valid_mask: BoolGrid | None = None,
    lambda_bg: float = 1.0,
    positive_layout: str = "topk",
    min_positive_distance: float = 3.0,
) -> CandidatePools:
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size values must be positive")
    if positive_limit < 0 or negative_limit < 0:
        raise ValueError("candidate limits must be non-negative")
    scores = _contrast_grid(foreground, background, lambda_bg)
    rows, cols = _shape(scores)
    ranked_positive = _ranked_indices(scores, valid_mask, descending=True)
    selected_positive = _select_positives(
        scores,
        ranked_positive,
        positive_limit=positive_limit,
        positive_layout=positive_layout,
        min_positive_distance=min_positive_distance,
    )
    selected = set(selected_positive)
    positives = tuple(
        Candidate(
            kind="positive",
            grid_index=(row, col),
            coordinate=_coordinate(row, col, rows, cols, image_size),
            score=scores[row][col],
        )
        for row, col in selected_positive
    )
    negatives: list[Candidate] = []
    for row, col in _ranked_indices(scores, valid_mask, descending=False):
        if len(negatives) >= negative_limit:
            break
        if (row, col) in selected:
            continue
        selected.add((row, col))
        negatives.append(
            Candidate(
                kind="negative",
                grid_index=(row, col),
                coordinate=_coordinate(row, col, rows, cols, image_size),
                score=-scores[row][col],
            )
        )
    return CandidatePools(positives=positives, negatives=tuple(negatives))


def validate_candidate_pools(candidates: Iterable[Candidate], *, image_size: tuple[int, int]) -> list[str]:
    width, height = image_size
    seen: set[tuple[int, int]] = set()
    issues: list[str] = []
    for candidate in candidates:
        if candidate.grid_index in seen:
            issues.append(f"duplicate grid candidate at {candidate.grid_index}")
        seen.add(candidate.grid_index)
        x, y = candidate.coordinate
        if not (0 <= x < width and 0 <= y < height):
            issues.append(f"candidate out of bounds at {candidate.coordinate}")
        if candidate.kind not in {"positive", "negative"}:
            issues.append(f"unknown candidate kind: {candidate.kind}")
    return issues


def _mask_value(mask: BoolGrid, coordinate: tuple[int, int]) -> bool:
    x, y = coordinate
    return 0 <= y < len(mask) and 0 <= x < len(mask[0]) and bool(mask[y][x])


def point_precision(candidates: Iterable[Candidate], mask: BoolGrid) -> float:
    positives = [candidate for candidate in candidates if candidate.kind == "positive"]
    if not positives:
        return 0.0
    hits = sum(1 for candidate in positives if _mask_value(mask, candidate.coordinate))
    return hits / len(positives)


def _component_labels(mask: BoolGrid) -> dict[tuple[int, int], int]:
    rows, cols = _shape(mask)
    labels: dict[tuple[int, int], int] = {}
    current = 0
    for row in range(rows):
        for col in range(cols):
            if not mask[row][col] or (row, col) in labels:
                continue
            current += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[(row, col)] = current
            while queue:
                r, c = queue.popleft()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr][nc] and (nr, nc) not in labels:
                        labels[(nr, nc)] = current
                        queue.append((nr, nc))
    return labels


def component_coverage(candidates: Iterable[Candidate], mask: BoolGrid) -> float:
    labels = _component_labels(mask)
    all_components = set(labels.values())
    if not all_components:
        return 0.0
    hit_components: set[int] = set()
    for candidate in candidates:
        if candidate.kind != "positive":
            continue
        x, y = candidate.coordinate
        label = labels.get((y, x))
        if label is not None:
            hit_components.add(label)
    return len(hit_components) / len(all_components)
