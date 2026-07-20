"""Phase X Stage S1 / Gate X2: Candidate Coverage Scaling.

Implements section 7 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.
This is a CPU-only audit of the candidate pool run *before* SAM inference.
For each K+ in the frozen grid, it takes the top-K+ positives from the
candidate pool, computes coverage/quality metrics against the query GT mask,
and reports whether the candidate universe supports dense target coverage.

If the supplied candidate pool has fewer than 100 positives per episode
(the historical pilot only provides 5), the script records the shortfall
and reports saturation at the actual maximum.  No fake points are inserted.

Outputs (section 7 layout)::

    reports/phaseX_candidate_scaling/
        positive_scaling_query_metrics.csv
        negative_scaling_query_metrics.csv
        positive_coverage_curve.csv
        negative_quality_curve.csv
        component_coverage_by_k.csv
        spatial_dispersion_by_k.csv
        scaling_summary.md
        scaling_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from adapromptset.scripts.build_phasex_preregistration import (
    NEGATIVE_K_GRID,
    POSITIVE_K_GRID,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 37
COVERAGE_RADIUS = float(IMAGE_SIZE // GRID_SIZE)  # ~14 px, one patch width
BOUNDARY_BAND = float(IMAGE_SIZE // GRID_SIZE)  # distance to count as "hard" background


# ---------------------------------------------------------------------------
# Episode path resolution (torch-free standalone copy of the project helper)
# ---------------------------------------------------------------------------


def resolve_episode_paths(row: dict[str, str], workspace_root: Path) -> dict[str, Path]:
    """Resolve episode image/mask paths relative to the dataset root.

    This is a torch-free copy of
    ``adapromptset.scripts.run_real_matcher_smoke.resolve_episode_paths`` so the
    candidate coverage audit can run without importing torch (which triggers an
    OpenMP runtime conflict on this environment).
    """
    dataset = row.get("dataset", "MoNuSAC")
    dataset_roots = {
        "MoNuSAC": workspace_root / "code" / "datasets" / "MoNuSAC",
        "MoNuSeg": workspace_root / "code" / "datasets" / "MonuSeg" / "official",
        "GlaS": workspace_root / "external" / "research_data" / "extracted" / "GlaS",
    }
    dataset_root = dataset_roots[dataset]

    def resolve_path(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path
        dataset_relative = dataset_root / path
        if dataset_relative.exists():
            return dataset_relative
        return workspace_root / path

    return {
        "support_image": resolve_path(row["support_image_path"]),
        "support_mask": resolve_path(row["support_mask_path"]),
        "query_image": resolve_path(row["query_image_path"]),
        "query_mask": resolve_path(row["query_mask_path"]),
    }


def load_mask_bool(path: Path, *, size: tuple[int, int]) -> list[list[bool]]:
    """Load a mask image as a 2D boolean grid (torch-free)."""
    image = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    width, height = size
    pixels = list(image.getdata())
    return [[bool(pixels[row * width + col]) for col in range(width)] for row in range(height)]


# ---------------------------------------------------------------------------
# Candidate pool loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePoint:
    kind: str
    grid_index: tuple[int, int]
    coordinate: tuple[int, int]
    score: float


def _read_candidate_pool(path: Path) -> dict[str, dict[str, list[CandidatePoint]]]:
    """Return {episode_id: {"positive": [...], "negative": [...]}}.

    Positives are sorted by score descending (highest contrast first).
    Negatives are sorted by score descending (the candidate pool stores
    ``-contrast`` for negatives so higher = stronger negative).
    """
    by_episode: dict[str, dict[str, list[CandidatePoint]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = row["episode_id"]
            kind = row["kind"]
            point = CandidatePoint(
                kind=kind,
                grid_index=(int(row["grid_row"]), int(row["grid_col"])),
                coordinate=(int(row["x"]), int(row["y"])),
                score=float(row["score"]),
            )
            entry = by_episode.setdefault(episode_id, {"positive": [], "negative": []})
            entry[kind].append(point)
    for entry in by_episode.values():
        entry["positive"].sort(key=lambda p: p.score, reverse=True)
        entry["negative"].sort(key=lambda p: p.score, reverse=True)
    return by_episode


def _read_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


BoolGrid = Sequence[Sequence[bool]]


def _component_labels(mask: BoolGrid) -> dict[tuple[int, int], int]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
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
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and mask[nr][nc]
                        and (nr, nc) not in labels
                    ):
                        labels[(nr, nc)] = current
                        queue.append((nr, nc))
    return labels


def _boundary_pixels(mask: BoolGrid) -> set[tuple[int, int]]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    boundary: set[tuple[int, int]] = set()
    for row in range(rows):
        for col in range(cols):
            if not mask[row][col]:
                continue
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr][nc]:
                    boundary.add((row, col))
                    break
    return boundary


def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _min_distance_to_set(point: tuple[int, int], points: set[tuple[int, int]]) -> float:
    if not points:
        return float("inf")
    return min(_euclidean(point, other) for other in points)


# ---------------------------------------------------------------------------
# Positive-side metrics (section 7)
# ---------------------------------------------------------------------------


def _point_precision(positives: list[CandidatePoint], mask: BoolGrid) -> float:
    if not positives:
        return 0.0
    hits = 0
    for p in positives:
        x, y = p.coordinate
        if 0 <= y < len(mask) and 0 <= x < len(mask[0]) and mask[y][x]:
            hits += 1
    return hits / len(positives)


def _component_coverage(positives: list[CandidatePoint], labels: dict[tuple[int, int], int]) -> float:
    all_components = set(labels.values())
    if not all_components:
        return 0.0
    hit: set[int] = set()
    for p in positives:
        x, y = p.coordinate
        label = labels.get((y, x))
        if label is not None:
            hit.add(label)
    return len(hit) / len(all_components)


def _missed_component_count(positives: list[CandidatePoint], labels: dict[tuple[int, int], int]) -> int:
    all_components = set(labels.values())
    if not all_components:
        return 0
    hit: set[int] = set()
    for p in positives:
        x, y = p.coordinate
        label = labels.get((y, x))
        if label is not None:
            hit.add(label)
    return len(all_components) - len(hit)


def _foreground_area_coverage_proxy(
    positives: list[CandidatePoint],
    mask: BoolGrid,
    radius: float,
) -> float:
    """Fraction of GT foreground pixels within ``radius`` of any selected positive."""
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    foreground_pixels = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if mask[row][col]
    ]
    if not foreground_pixels or not positives:
        return 0.0
    int_radius = int(math.ceil(radius))
    positive_coords = [p.coordinate for p in positives]
    covered = 0
    for prow, pcol in foreground_pixels:
        for px, py in positive_coords:
            if abs(prow - py) <= int_radius and abs(pcol - px) <= int_radius:
                dx = pcol - px
                dy = prow - py
                if dx * dx + dy * dy <= radius * radius:
                    covered += 1
                    break
    return covered / len(foreground_pixels)


def _spatial_dispersion(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            total += _euclidean(a, b)
            count += 1
    return total / count if count else 0.0


def _nearest_neighbor_distance(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i, a in enumerate(points):
        nn = min(_euclidean(a, b) for j, b in enumerate(points) if j != i)
        total += nn
    return total / len(points)


def _redundancy(points: list[tuple[int, int]], radius: float) -> float:
    if len(points) < 2:
        return 0.0
    redundant = 0
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            if _euclidean(a, b) < radius:
                redundant += 1
                break
    return redundant / len(points)


def _candidate_entropy(positives: list[CandidatePoint]) -> float:
    if len(positives) < 2:
        return 0.0
    scores = [max(p.score, 0.0) for p in positives]
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for s in scores:
        p = s / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Negative-side metrics (section 7)
# ---------------------------------------------------------------------------


def _background_precision(negatives: list[CandidatePoint], mask: BoolGrid) -> float:
    if not negatives:
        return 0.0
    hits = 0
    for p in negatives:
        x, y = p.coordinate
        if 0 <= y < len(mask) and 0 <= x < len(mask[0]) and not mask[y][x]:
            hits += 1
    return hits / len(negatives)


def _false_negative_risk(negatives: list[CandidatePoint], mask: BoolGrid) -> float:
    if not negatives:
        return 0.0
    hits = 0
    for p in negatives:
        x, y = p.coordinate
        if 0 <= y < len(mask) and 0 <= x < len(mask[0]) and mask[y][x]:
            hits += 1
    return hits / len(negatives)


def _boundary_proximity(negatives: list[CandidatePoint], boundary: set[tuple[int, int]]) -> float:
    if not negatives:
        return 0.0
    if not boundary:
        return float("inf")
    total = 0.0
    for p in negatives:
        total += _min_distance_to_set(p.coordinate, boundary)
    return total / len(negatives)


def _hard_background_rate(
    negatives: list[CandidatePoint],
    boundary: set[tuple[int, int]],
    band: float,
) -> float:
    if not negatives:
        return 0.0
    if not boundary:
        return 0.0
    hard = 0
    for p in negatives:
        if _min_distance_to_set(p.coordinate, boundary) <= band:
            hard += 1
    return hard / len(negatives)


def _easy_background_fraction(
    negatives: list[CandidatePoint],
    boundary: set[tuple[int, int]],
    band: float,
) -> float:
    if not negatives:
        return 0.0
    if not boundary:
        return 1.0
    easy = 0
    for p in negatives:
        if _min_distance_to_set(p.coordinate, boundary) > 2 * band:
            easy += 1
    return easy / len(negatives)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def _compute_positive_metrics(
    positives: list[CandidatePoint],
    mask: BoolGrid,
    labels: dict[tuple[int, int], int],
    k: int,
) -> dict[str, Any]:
    selected = positives[:k]
    coords = [p.coordinate for p in selected]
    return {
        "k_positive": k,
        "actual_positive_count": len(selected),
        "shortfall": max(0, k - len(selected)),
        "point_precision": _point_precision(selected, mask),
        "component_coverage": _component_coverage(selected, labels),
        "missed_component_count": _missed_component_count(selected, labels),
        "foreground_area_coverage_proxy": _foreground_area_coverage_proxy(selected, mask, COVERAGE_RADIUS),
        "spatial_dispersion": _spatial_dispersion(coords),
        "nearest_neighbor_distance": _nearest_neighbor_distance(coords),
        "redundancy": _redundancy(coords, COVERAGE_RADIUS),
        "candidate_entropy": _candidate_entropy(selected),
    }


def _compute_negative_metrics(
    negatives: list[CandidatePoint],
    mask: BoolGrid,
    boundary: set[tuple[int, int]],
    k: int,
) -> dict[str, Any]:
    selected = negatives[:k]
    coords = [p.coordinate for p in selected]
    return {
        "k_negative": k,
        "actual_negative_count": len(selected),
        "shortfall": max(0, k - len(selected)),
        "background_precision": _background_precision(selected, mask),
        "false_negative_risk": _false_negative_risk(selected, mask),
        "boundary_proximity": _boundary_proximity(selected, boundary),
        "hard_background_rate": _hard_background_rate(selected, boundary, BOUNDARY_BAND),
        "easy_background_fraction": _easy_background_fraction(selected, boundary, BOUNDARY_BAND),
        "spatial_dispersion": _spatial_dispersion(coords),
    }


def _gate_x2_status(
    coverage_curve: dict[int, float],
    *,
    pool_max_positive: int | None = None,
) -> tuple[str, str]:
    """Return (status_code, rationale).

    ``pool_max_positive`` is the maximum number of positive candidates actually
    available per episode.  If not provided, it is inferred from the largest
    K+ key in ``coverage_curve`` (the historical pilot only has 5 positives per
    episode, so coverage maxes out at K=5).  If the pool maxes out below 100,
    the K+=100 coverage value is not a true K+=100 measurement and the
    saturation conclusion cannot be drawn.
    """
    if not coverage_curve:
        return "CANDIDATE_SCALING_INSUFFICIENT_DATA", "no coverage data available"

    inferred_max = max(coverage_curve.keys())
    if pool_max_positive is None:
        pool_max_positive = inferred_max
    else:
        # The actual pool can be smaller than the inferred max if the curve
        # only includes the frozen grid (e.g. pool=5 but curve includes K=100
        # because the same 5 positives were reused).  Take the smaller of the
        # two to be safe.
        pool_max_positive = min(pool_max_positive, inferred_max)

    cov_5 = coverage_curve.get(5)
    cov_50 = coverage_curve.get(50)
    cov_100 = coverage_curve.get(100)

    # If the pool never reached 100 distinct positives, we cannot draw a
    # saturation conclusion at K+=100.  Look up coverage at the smallest grid
    # K+ that is >= pool_max_positive (that is the max coverage the pool can
    # actually deliver; smaller K values underestimate the pool).
    if pool_max_positive < 100:
        ceiling_ks = [k for k in coverage_curve.keys() if k >= pool_max_positive]
        if ceiling_ks:
            k_at_max = min(ceiling_ks)
            cov_max = coverage_curve[k_at_max]
            k_label = f"K+={k_at_max} (grid ceiling for pool max {pool_max_positive})"
        else:
            # All grid K values are below pool_max_positive — use the largest.
            k_at_max = max(coverage_curve.keys())
            cov_max = coverage_curve[k_at_max]
            k_label = f"K+={k_at_max} (max grid K)"
        if cov_max < 0.5:
            return (
                "CANDIDATE_UNIVERSE_INADEQUATE",
                f"component coverage at {k_label} is {cov_max:.3f} < 0.5; "
                "candidate pool too small to reach K+=100 and already inadequate",
            )
        return (
            "CANDIDATE_SCALING_INSUFFICIENT_DATA",
            f"candidate pool maxes out at K+={pool_max_positive}; cannot test "
            "K+=100 with 100 distinct positives; regenerate pool with "
            "positive_limit>=100 for full Gate X2",
        )

    # We have K+=100 data backed by >=100 distinct positives.
    cov_5_val = cov_5 if cov_5 is not None else 0.0
    cov_100_val = cov_100 if cov_100 is not None else 0.0

    if cov_100_val < 0.5:
        return (
            "CANDIDATE_UNIVERSE_INADEQUATE",
            f"even at K+=100, component coverage is {cov_100_val:.3f} < 0.5; "
            "problem is matcher/candidate generator, not policy",
        )

    _ = cov_50  # reserved for future use
    if cov_5_val < 0.5 and cov_100_val > cov_5_val + 0.1:
        return (
            "OLD_SMALL_K_BANK_INADEQUATE",
            f"K+=5 coverage {cov_5_val:.3f} is low but K+=100 reaches {cov_100_val:.3f}; "
            "historical 5-positive bank was inadequate",
        )

    if cov_100_val - cov_5_val < 0.05:
        return (
            "CANDIDATE_SCALING_SATURATES_SMALL_K",
            f"coverage saturates by K+=5 ({cov_5_val:.3f} -> {cov_100_val:.3f}); "
            "candidate universe is adequate at small K",
        )

    return (
        "CANDIDATE_SCALING_SUPPORTS_LARGE_K",
        f"K+=5 coverage {cov_5_val:.3f}, K+=100 coverage {cov_100_val:.3f}; "
        "candidate universe supports dense target coverage",
    )


def run_candidate_scaling(
    *,
    project_root: Path = PROJECT_ROOT,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    workspace_root = project_root.parent
    candidates_csv = candidates_csv or (project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv")
    episodes_csv = episodes_csv or (project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv")
    out_dir = out_dir or (project_root / "reports" / "phaseX_candidate_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_CANDIDATE_POOL_MISSING: {candidates_csv}")
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_EPISODES_MISSING: {episodes_csv}")

    pool = _read_candidate_pool(candidates_csv)
    episodes = _read_episodes(episodes_csv)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    positive_query_rows: list[dict[str, Any]] = []
    negative_query_rows: list[dict[str, Any]] = []
    coverage_by_k: dict[int, list[float]] = {k: [] for k in POSITIVE_K_GRID}
    dispersion_by_k: dict[int, list[float]] = {k: [] for k in POSITIVE_K_GRID}
    precision_by_k: dict[int, list[float]] = {k: [] for k in POSITIVE_K_GRID}
    bg_precision_by_k: dict[int, list[float]] = {k: [] for k in NEGATIVE_K_GRID}
    fn_risk_by_k: dict[int, list[float]] = {k: [] for k in NEGATIVE_K_GRID}

    start = time.perf_counter()
    episode_count = 0
    pool_max_positive = 0
    pool_max_negative = 0

    for episode in episodes:
        episode_id = episode["episode_id"]
        if episode_id not in pool:
            continue
        episode_count += 1
        paths = resolve_episode_paths(episode, workspace_root)
        mask_bool = load_mask_bool(paths["query_mask"], size=(IMAGE_SIZE, IMAGE_SIZE))
        labels = _component_labels(mask_bool)
        boundary = _boundary_pixels(mask_bool)

        positives = pool[episode_id]["positive"]
        negatives = pool[episode_id]["negative"]
        pool_max_positive = max(pool_max_positive, len(positives))
        pool_max_negative = max(pool_max_negative, len(negatives))

        for k in POSITIVE_K_GRID:
            metrics = _compute_positive_metrics(positives, mask_bool, labels, k)
            row = {"episode_id": episode_id, **metrics}
            positive_query_rows.append(row)
            coverage_by_k[k].append(metrics["component_coverage"])
            dispersion_by_k[k].append(metrics["spatial_dispersion"])
            precision_by_k[k].append(metrics["point_precision"])

        for k in NEGATIVE_K_GRID:
            if k == 0:
                continue
            metrics = _compute_negative_metrics(negatives, mask_bool, boundary, k)
            row = {"episode_id": episode_id, **metrics}
            negative_query_rows.append(row)
            bg_precision_by_k[k].append(metrics["background_precision"])
            fn_risk_by_k[k].append(metrics["false_negative_risk"])

    elapsed = time.perf_counter() - start

    # Aggregate curves.
    positive_coverage_curve = []
    for k in POSITIVE_K_GRID:
        values = coverage_by_k[k]
        positive_coverage_curve.append({
            "k_positive": k,
            "episode_count": len(values),
            "mean_component_coverage": sum(values) / len(values) if values else 0.0,
            "mean_point_precision": sum(precision_by_k[k]) / len(precision_by_k[k]) if precision_by_k[k] else 0.0,
            "mean_spatial_dispersion": sum(dispersion_by_k[k]) / len(dispersion_by_k[k]) if dispersion_by_k[k] else 0.0,
        })

    negative_quality_curve = []
    for k in NEGATIVE_K_GRID:
        if k == 0:
            continue
        values = bg_precision_by_k[k]
        negative_quality_curve.append({
            "k_negative": k,
            "episode_count": len(values),
            "mean_background_precision": sum(values) / len(values) if values else 0.0,
            "mean_false_negative_risk": sum(fn_risk_by_k[k]) / len(fn_risk_by_k[k]) if fn_risk_by_k[k] else 0.0,
        })

    coverage_curve_map = {
        row["k_positive"]: row["mean_component_coverage"]
        for row in positive_coverage_curve
    }
    gate_status, gate_rationale = _gate_x2_status(coverage_curve_map, pool_max_positive=pool_max_positive)

    # --- Write CSVs ---
    _write_csv(
        out_dir / "positive_scaling_query_metrics.csv",
        positive_query_rows,
    )
    _write_csv(
        out_dir / "negative_scaling_query_metrics.csv",
        negative_query_rows,
    )
    _write_csv(
        out_dir / "positive_coverage_curve.csv",
        positive_coverage_curve,
    )
    _write_csv(
        out_dir / "negative_quality_curve.csv",
        negative_quality_curve,
    )
    _write_csv(
        out_dir / "component_coverage_by_k.csv",
        [
            {
                "k_positive": row["k_positive"],
                "episode_count": row["episode_count"],
                "mean_component_coverage": row["mean_component_coverage"],
            }
            for row in positive_coverage_curve
        ],
    )
    _write_csv(
        out_dir / "spatial_dispersion_by_k.csv",
        [
            {
                "k_positive": row["k_positive"],
                "episode_count": row["episode_count"],
                "mean_spatial_dispersion": row["mean_spatial_dispersion"],
            }
            for row in positive_coverage_curve
        ],
    )

    summary = {
        "phase": "phaseX",
        "stage": "candidate_scaling",
        "gate": "X2",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "episode_count": episode_count,
        "pool_max_positive": pool_max_positive,
        "pool_max_negative": pool_max_negative,
        "positive_k_grid": list(POSITIVE_K_GRID),
        "negative_k_grid": list(NEGATIVE_K_GRID),
        "gate_x2_status": gate_status,
        "gate_x2_rationale": gate_rationale,
        "coverage_curve": {
            str(row["k_positive"]): row["mean_component_coverage"]
            for row in positive_coverage_curve
        },
        "precision_curve": {
            str(row["k_positive"]): row["mean_point_precision"]
            for row in positive_coverage_curve
        },
        "elapsed_seconds": round(elapsed, 3),
        "cpu_only": True,
        "sam_inference_run": False,
    }
    (out_dir / "scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "scaling_summary.md").write_text(
        _build_summary_md(summary, positive_coverage_curve, negative_quality_curve),
        encoding="utf-8",
    )

    return summary


def _build_summary_md(
    summary: dict[str, Any],
    positive_curve: list[dict[str, Any]],
    negative_curve: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase X Candidate Coverage Scaling Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: {summary['stage']} (Gate X2)",
        f"Episodes: {summary['episode_count']}",
        f"Pool max positives: {summary['pool_max_positive']}",
        f"Pool max negatives: {summary['pool_max_negative']}",
        f"CPU-only: {summary['cpu_only']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        "",
        "## Gate X2 Status",
        "",
        f"**{summary['gate_x2_status']}**",
        "",
        summary["gate_x2_rationale"],
        "",
        "## Positive coverage curve",
        "",
        "| K+ | Episodes | Mean component coverage | Mean point precision | Mean spatial dispersion |",
        "|---|---|---|---|---|",
    ]
    for row in positive_curve:
        lines.append(
            f"| {row['k_positive']} | {row['episode_count']} | "
            f"{row['mean_component_coverage']:.4f} | "
            f"{row['mean_point_precision']:.4f} | "
            f"{row['mean_spatial_dispersion']:.2f} |"
        )
    lines += [
        "",
        "## Negative quality curve",
        "",
        "| K- | Episodes | Mean background precision | Mean false-negative risk |",
        "|---|---|---|---|",
    ]
    for row in negative_curve:
        lines.append(
            f"| {row['k_negative']} | {row['episode_count']} | "
            f"{row['mean_background_precision']:.4f} | "
            f"{row['mean_false_negative_risk']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If `OLD_SMALL_K_BANK_INADEQUATE`: the historical 5-positive bank was too small.",
        "- If `CANDIDATE_UNIVERSE_INADEQUATE`: the matcher/candidate generator is the bottleneck.",
        "- If `CANDIDATE_SCALING_SUPPORTS_LARGE_K`: dense prompts are viable.",
        "- If `CANDIDATE_SCALING_SATURATES_SMALL_K`: small K+ is sufficient.",
        "- If `CANDIDATE_SCALING_INSUFFICIENT_DATA`: regenerate the pool with positive_limit>=100.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase X Stage S1 / Gate X2 candidate coverage scaling (CPU-only)."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    summary = run_candidate_scaling(
        project_root=args.project_root,
        candidates_csv=args.candidates_csv,
        episodes_csv=args.episodes_csv,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
