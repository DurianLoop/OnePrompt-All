"""Phase X Stage S3: Grouping Scaling.

Implements section 8 Stage S2 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.

For each K+ in a representative subset of the S1 K+ grid (directive specifies
{10, 30, 50, 100}; we also include 20 since Stage S2 found K=20 was the Dice
peak) and each of the 12 frozen grouping families, run SAM-B on the top K+
positive candidates with the grouping rule applied. Negative prompts are
held at N_NONE (Stage S3 isolates the grouping axis). For each group, SAM is
called once with that group's positive points. The resulting per-group masks
are fused via the standard NMS + semantic union pipeline (FusionConfig from
``configs/segmenter/fusion_mvp.json``).

Per section 8.6/8.7 fairness rules, every (K+, grouping) combo records both
the actual SAM-call count and total point count, so comparisons can be made
at matched call budget OR matched point budget OR on a Pareto curve.

Outputs (section 8 Stage S2 layout)::

    reports/phaseX_grouping_scaling/
        grouping_scaling_query_metrics.csv
        grouping_scaling_curve.csv
        grouping_scaling_summary.md
        grouping_scaling_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adapromptset.evaluation.metrics import (
    aggregated_jaccard_index,
    dice_score,
    iou_score,
)
from adapromptset.fusion.masks import (
    FusionConfig,
    MaskCandidate,
    fuse_masks,
    load_fusion_config,
)
from adapromptset.matching.runtime import checkpoint_record
from adapromptset.scripts.build_action_bank import _candidate_pools_from_csv
from adapromptset.scripts.build_phasex_preregistration import GROUPINGS
from adapromptset.scripts.run_phasex_single_variable_kpos import (
    _component_labels,
    _component_recall,
    _semantic_instance_labels,
)
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518

# Representative K+ subset per directive section 8 Stage S2.
# Directive specifies {10, 30, 50, 100}; we add 20 because Stage S2 found
# K=20 to be the Dice peak (0.524) — testing the peak directly is essential.
REPRESENTATIVE_K_POSITIVE: tuple[int, ...] = (10, 20, 30, 50, 100)

# Distance thresholds (in SAM input-image pixels, IMAGE_SIZE=518) for the
# graph-based grouping families. These are frozen at preregistration time
# and recorded in the summary so that the grouping rule is reproducible.
G_GRAPH_DISTANCE_THRESHOLD = 80.0
G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD = 30.0

# G_SPATIAL_GRID uses a 2x2 grid (4 cells) over the IMAGE_SIZE x IMAGE_SIZE
# input. This is a coarse split suitable for the validation subset.
G_SPATIAL_GRID_CELLS_PER_AXIS = 2


def _read_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_ledger(project_root: Path, gpu_hours: float) -> None:
    ledger_path = project_root / "reports" / "phaseX_compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "command", "gpu_hours", "notes"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "phaseX-stage-S3-grouping-scaling",
                "command": "scripts/run_phasex_grouping_scaling.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Phase X Stage S3: grouping scaling with SAM-B on MoNuSAC validation.",
            }
        )


# ---------------------------------------------------------------------------
# Grouping rules: each takes the top-k positive coordinates and returns a
# list of index tuples (into coords[:k]) defining the groups.
# ---------------------------------------------------------------------------


def _kmeans_assign(
    points: list[tuple[float, float]],
    centroids: list[tuple[float, float]],
) -> list[int]:
    assignments: list[int] = []
    for px, py in points:
        best_cluster = 0
        best_dist = math.hypot(px - centroids[0][0], py - centroids[0][1])
        for ci in range(1, len(centroids)):
            d = math.hypot(px - centroids[ci][0], py - centroids[ci][1])
            if d < best_dist:
                best_dist = d
                best_cluster = ci
        assignments.append(best_cluster)
    return assignments


def _kmeans_groups(
    coords: list[tuple[int, int]],
    k_clusters: int,
) -> list[tuple[int, ...]]:
    """Plain Lloyd's algorithm on (x, y). Deterministic init: first k points.

    If k_clusters >= len(coords), each point becomes its own cluster (we
    degrade gracefully — the resulting group count is len(coords)).
    """
    n = len(coords)
    if n == 0:
        return []
    k_clusters = min(k_clusters, n)
    if k_clusters <= 1:
        return [tuple(range(n))]
    points: list[tuple[float, float]] = [
        (float(x), float(y)) for x, y in coords
    ]
    # Deterministic init: evenly spaced points across the sorted-by-index list.
    centroids = [
        points[(i * n) // k_clusters] for i in range(k_clusters)
    ]
    for _ in range(10):  # fixed 10 iterations
        assignments = _kmeans_assign(points, centroids)
        new_centroids: list[tuple[float, float]] = []
        for ci in range(k_clusters):
            members = [
                points[i] for i in range(n) if assignments[i] == ci
            ]
            if not members:
                # Keep the old centroid if cluster emptied.
                new_centroids.append(centroids[ci])
                continue
            cx = sum(m[0] for m in members) / len(members)
            cy = sum(m[1] for m in members) / len(members)
            new_centroids.append((cx, cy))
        if new_centroids == centroids:
            break
        centroids = new_centroids
    assignments = _kmeans_assign(points, centroids)
    groups: list[tuple[int, ...]] = []
    for ci in range(k_clusters):
        members = tuple(i for i in range(n) if assignments[i] == ci)
        if members:
            groups.append(members)
    return groups


def _connected_component_groups(
    coords: list[tuple[int, int]],
    threshold: float,
) -> list[tuple[int, ...]]:
    """Union-find: connect two points if their Euclidean distance <= threshold."""
    n = len(coords)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            if math.hypot(xi - xj, yi - yj) <= threshold:
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [tuple(members) for members in clusters.values()]


def _spatial_grid_groups(
    coords: list[tuple[int, int]],
    cells_per_axis: int,
    image_size: int,
) -> list[tuple[int, ...]]:
    """Group points by which grid cell they fall into.

    cells_per_axis=2 with image_size=518 gives a 2x2 grid of 4 cells. Points
    on cell boundaries are assigned deterministically by floor division.
    """
    n = len(coords)
    if n == 0:
        return []
    cell_size = image_size / cells_per_axis
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(coords):
        cx = min(cells_per_axis - 1, int(x // cell_size))
        cy = min(cells_per_axis - 1, int(y // cell_size))
        buckets.setdefault((cx, cy), []).append(i)
    # Sort by (cx, cy) for deterministic output ordering.
    return [
        tuple(buckets[key])
        for key in sorted(buckets.keys())
    ]


def split_groups(
    grouping: str,
    k: int,
    coords: list[tuple[int, int]],
) -> list[tuple[int, ...]]:
    """Return a list of positive-index tuples for the given grouping family.

    Indices refer to positions 0..k-1 in ``coords`` (i.e. the top-k positive
    candidates from the pool).
    """
    indices = list(range(k))
    if grouping == "G_ONE_GROUP":
        return [tuple(indices)]
    if grouping == "G_ONE_POINT_PER_GROUP":
        return [(i,) for i in indices]
    if grouping.startswith("G_FIXED_CHUNK_"):
        n = int(grouping.rsplit("_", 1)[-1])
        return [tuple(indices[i : i + n]) for i in range(0, k, n)]
    if grouping.startswith("G_KMEANS_"):
        n_clusters = int(grouping.rsplit("_", 1)[-1])
        return _kmeans_groups(coords[:k], n_clusters)
    if grouping == "G_SPATIAL_GRID":
        return _spatial_grid_groups(
            coords[:k], G_SPATIAL_GRID_CELLS_PER_AXIS, IMAGE_SIZE
        )
    if grouping == "G_GRAPH":
        return _connected_component_groups(
            coords[:k], G_GRAPH_DISTANCE_THRESHOLD
        )
    if grouping == "G_CANDIDATE_COMPONENT":
        return _connected_component_groups(
            coords[:k], G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD
        )
    raise ValueError(f"unknown grouping: {grouping}")


# ---------------------------------------------------------------------------
# SAM inference + fusion
# ---------------------------------------------------------------------------


def _predict_group(
    predictor: Any,
    points: list[tuple[int, int]],
) -> tuple[list[list[bool]], float]:
    labels = [1] * len(points)
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def _run_episode_grouping(
    predictor: Any,
    coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    k: int,
    grouping: str,
    fusion_config: FusionConfig,
    episode_id: str,
) -> dict[str, Any]:
    """Run one (episode, K+, grouping) combo. Returns a metrics record."""
    groups = split_groups(grouping, k, coords)
    if not groups:
        return {
            "episode_id": episode_id,
            "k_positive": k,
            "grouping": grouping,
            "actual_group_count": 0,
            "sam_calls": 0,
            "dice": None,
            "iou": None,
            "aji": None,
            "component_recall": None,
            "target_component_count": target_component_count,
            "hit_components": None,
            "retained_masks": 0,
            "runtime_ms": 0.0,
            "mean_group_size": None,
            "max_group_size": None,
            "status": "SKIPPED_NO_GROUPS",
        }

    t0 = time.perf_counter()
    candidates: list[MaskCandidate] = []
    for gi, indices in enumerate(groups):
        points = [coords[i] for i in indices]
        mask, score = _predict_group(predictor, points)
        candidates.append(
            MaskCandidate(mask=mask, score=score, source_id=f"{grouping}:g{gi}")
        )
    retained, semantic = fuse_masks(candidates, fusion_config)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    dice = dice_score(semantic, target)
    iou = iou_score(semantic, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(semantic),
        _semantic_instance_labels(target),
    )
    recall, hit_count, total_count = _component_recall(semantic, target_labels)
    group_sizes = [len(g) for g in groups]

    return {
        "episode_id": episode_id,
        "k_positive": k,
        "grouping": grouping,
        "actual_group_count": len(groups),
        "sam_calls": len(groups),
        "dice": dice,
        "iou": iou,
        "aji": aji,
        "component_recall": recall,
        "target_component_count": total_count,
        "hit_components": hit_count,
        "retained_masks": len(retained),
        "runtime_ms": round(runtime_ms, 3),
        "mean_group_size": sum(group_sizes) / len(group_sizes),
        "max_group_size": max(group_sizes),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Gate X4 (preliminary): grouping saturation status
# ---------------------------------------------------------------------------


def _gate_x4_grouping_status(
    curve: list[dict[str, Any]],
) -> tuple[str, str]:
    """Assess grouping scaling.

    Per directive section 17 Gate X4 statuses:
      - ACTION_SPACE_NOT_SATURATED
      - ACTION_SPACE_SATURATED
      - OLD_ACTION_BANK_INADEQUATE

    This is a *preliminary* Gate X4 read on the grouping axis only. The full
    Gate X4 verdict requires nested action bank saturation (section 11) and
    search-oracle adequacy (section 13). Here we report whether the best
    grouping at the S2 Dice peak (K=20) significantly beats G_ONE_GROUP
    (the historical grouping), which is the relevant question for stage S3.

    The full set of statuses we may return here:
      - GROUPING_BEST_BEATS_ONE_GROUP: a non-trivial grouping beats G_ONE_GROUP
        at the S2 peak by >= 0.01 Dice.
      - GROUPING_ONE_GROUP_DOMINATES: G_ONE_GROUP is within 0.01 Dice of the
        best grouping at the S2 peak.
      - GROUPING_INSUFFICIENT_DATA: missing data at the S2 peak.
    """
    # Use K=20 (the S2 Dice peak) as the primary comparison point. Fall back
    # to K=10 if K=20 is not in the curve.
    target_k = 20
    k_rows = [c for c in curve if c["k_positive"] == target_k]
    if not k_rows:
        target_k = 10
        k_rows = [c for c in curve if c["k_positive"] == target_k]
    if not k_rows:
        return (
            "GROUPING_INSUFFICIENT_DATA",
            "no data at K=20 or K=10 for grouping comparison",
        )

    valid = [c for c in k_rows if c["mean_dice"] is not None]
    if not valid:
        return (
            "GROUPING_INSUFFICIENT_DATA",
            f"no valid mean_dice at K={target_k}",
        )

    one_group = next(
        (c for c in valid if c["grouping"] == "G_ONE_GROUP"), None
    )
    if one_group is None:
        return (
            "GROUPING_INSUFFICIENT_DATA",
            f"G_ONE_GROUP missing at K={target_k}",
        )

    best = max(valid, key=lambda c: c["mean_dice"])
    gain = best["mean_dice"] - one_group["mean_dice"]
    if gain >= 0.01:
        return (
            "GROUPING_BEST_BEATS_ONE_GROUP",
            f"K={target_k} best grouping={best['grouping']} "
            f"mean_dice={best['mean_dice']:.4f} vs G_ONE_GROUP "
            f"mean_dice={one_group['mean_dice']:.4f}; gain={gain:.4f} >= 0.010",
        )
    return (
        "GROUPING_ONE_GROUP_DOMINATES",
        f"K={target_k} best grouping={best['grouping']} "
        f"mean_dice={best['mean_dice']:.4f} vs G_ONE_GROUP "
        f"mean_dice={one_group['mean_dice']:.4f}; gain={gain:.4f} < 0.010",
    )


def _format_or_na(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "N/A"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "N/A"


def _write_summary_md(
    path: Path,
    summary: dict[str, Any],
    curve: list[dict[str, Any]],
) -> None:
    # Group curve by K+ for printing.
    by_k: dict[int, list[dict[str, Any]]] = {}
    for c in curve:
        by_k.setdefault(c["k_positive"], []).append(c)

    lines = [
        "# Phase X Stage S3: Grouping Scaling Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: grouping_scaling (Gate X4 preliminary)",
        f"Episodes: {summary['episode_count']}",
        f"Positive limit: {summary['positive_limit']}",
        f"Device: {summary['device']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        f"Elapsed seconds: {summary['elapsed_seconds']}",
        f"GPU-hours: {summary['gpu_hours']}",
        "",
        "## Grouping rule parameters (frozen)",
        "",
        f"- G_GRAPH_DISTANCE_THRESHOLD = {summary['g_graph_distance_threshold']}",
        f"- G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD = {summary['g_candidate_component_distance_threshold']}",
        f"- G_SPATIAL_GRID_CELLS_PER_AXIS = {summary['g_spatial_grid_cells_per_axis']}",
        f"- G_SPATIAL_GRID total cells = {summary['g_spatial_grid_cells_per_axis'] ** 2}",
        "",
        "## Gate X4 (preliminary, grouping axis only)",
        "",
        f"**{summary['gate_x4_grouping_status']}**",
        "",
        summary["gate_x4_grouping_rationale"],
        "",
        "## Grouping scaling curve",
        "",
        "| K+ | Grouping | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean calls | Mean runtime (ms) | Mean group size |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(by_k.keys()):
        for c in by_k[k]:
            lines.append(
                f"| {c['k_positive']} | {c['grouping']} | {c['episode_count']} | "
                f"{_format_or_na(c['mean_dice'])} | "
                f"{_format_or_na(c['mean_iou'])} | "
                f"{_format_or_na(c['mean_aji'])} | "
                f"{_format_or_na(c['mean_component_recall'])} | "
                f"{_format_or_na(c['mean_sam_calls'], '.1f')} | "
                f"{_format_or_na(c['mean_runtime_ms'], '.1f')} | "
                f"{_format_or_na(c['mean_group_size'], '.1f')} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `GROUPING_BEST_BEATS_ONE_GROUP`: a non-trivial grouping beats G_ONE_GROUP by >= 0.01 Dice at the S2 peak.",
            "- `GROUPING_ONE_GROUP_DOMINATES`: G_ONE_GROUP is within 0.01 Dice of the best grouping at the S2 peak.",
            "- `GROUPING_INSUFFICIENT_DATA`: missing data at the S2 peak.",
            "",
            "Note: this is a *preliminary* read on the grouping axis only. The full Gate X4",
            "verdict requires nested action bank saturation (section 11) and search-oracle",
            "adequacy (section 13).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grouping_scaling(
    *,
    project_root: Path = PROJECT_ROOT,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    device: str = "cpu",
    positive_limit: int = 100,
    k_values: Sequence[int] | None = None,
    groupings: Sequence[str] | None = None,
) -> dict[str, Any]:
    workspace_root = project_root.parent
    candidates_csv = candidates_csv or (
        project_root / "reports" / "phaseX_candidate_pilot_100" / "candidate_pilot_candidates.csv"
    )
    episodes_csv = episodes_csv or (
        project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv"
    )
    out_dir = out_dir or (project_root / "reports" / "phaseX_grouping_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_CANDIDATE_POOL_MISSING: {candidates_csv}")
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_EPISODES_MISSING: {episodes_csv}")

    episodes = _read_episodes(episodes_csv)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    k_grid = tuple(k_values) if k_values is not None else REPRESENTATIVE_K_POSITIVE
    grouping_grid = tuple(groupings) if groupings is not None else GROUPINGS

    # Load fusion config
    fusion_config = load_fusion_config(
        project_root / "configs" / "segmenter" / "fusion_mvp.json"
    )

    # Load SAM predictor
    sam_checkpoint = workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth"
    segmenter_record = checkpoint_record(sam_checkpoint, role="segmenter")
    predictor = load_sam_predictor(sam_checkpoint, workspace_root, device)

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    episode_count = 0

    for episode in episodes:
        episode_id = episode["episode_id"]
        paths = resolve_episode_paths(episode, workspace_root)
        image = load_rgb(paths["query_image"])
        target = load_mask(paths["query_mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        try:
            pools = _candidate_pools_from_csv(
                candidates_csv,
                episode_id=episode_id,
                positive_limit=positive_limit,
            )
        except ValueError:
            continue
        positive_coords = [c.coordinate for c in pools.positives]
        if len(positive_coords) < max(k_grid):
            # Skip episode if it doesn't have enough positives for the largest K.
            continue

        # Set image once per episode (expensive on CPU).
        predictor.set_image(np.asarray(image))
        episode_count += 1

        for k in k_grid:
            if k > len(positive_coords):
                continue
            for grouping in grouping_grid:
                record = _run_episode_grouping(
                    predictor,
                    positive_coords,
                    target,
                    target_labels,
                    target_component_count,
                    k=k,
                    grouping=grouping,
                    fusion_config=fusion_config,
                    episode_id=episode_id,
                )
                records.append(record)

    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0

    # Aggregate by (K+, grouping)
    curve: list[dict[str, Any]] = []
    for k in k_grid:
        for grouping in grouping_grid:
            k_rows = [
                r for r in records
                if r["k_positive"] == k
                and r["grouping"] == grouping
                and r["status"] == "OK"
            ]
            if not k_rows:
                curve.append(
                    {
                        "k_positive": k,
                        "grouping": grouping,
                        "episode_count": 0,
                        "mean_dice": None,
                        "mean_iou": None,
                        "mean_aji": None,
                        "mean_component_recall": None,
                        "mean_sam_calls": None,
                        "mean_runtime_ms": None,
                        "mean_group_size": None,
                        "mean_retained_masks": None,
                    }
                )
                continue
            curve.append(
                {
                    "k_positive": k,
                    "grouping": grouping,
                    "episode_count": len(k_rows),
                    "mean_dice": sum(r["dice"] for r in k_rows) / len(k_rows),
                    "mean_iou": sum(r["iou"] for r in k_rows) / len(k_rows),
                    "mean_aji": sum(r["aji"] for r in k_rows) / len(k_rows),
                    "mean_component_recall": sum(r["component_recall"] for r in k_rows) / len(k_rows),
                    "mean_sam_calls": sum(r["sam_calls"] for r in k_rows) / len(k_rows),
                    "mean_runtime_ms": sum(r["runtime_ms"] for r in k_rows) / len(k_rows),
                    "mean_group_size": sum(r["mean_group_size"] for r in k_rows) / len(k_rows),
                    "mean_retained_masks": sum(r["retained_masks"] for r in k_rows) / len(k_rows),
                }
            )

    gate_status, gate_rationale = _gate_x4_grouping_status(curve)

    _write_csv(out_dir / "grouping_scaling_query_metrics.csv", records)
    _write_csv(out_dir / "grouping_scaling_curve.csv", curve)

    summary = {
        "phase": "phaseX",
        "stage": "grouping_scaling",
        "gate": "X4_preliminary",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "episode_count": episode_count,
        "positive_limit": positive_limit,
        "k_values": list(k_grid),
        "groupings": list(grouping_grid),
        "gate_x4_grouping_status": gate_status,
        "gate_x4_grouping_rationale": gate_rationale,
        "g_graph_distance_threshold": G_GRAPH_DISTANCE_THRESHOLD,
        "g_candidate_component_distance_threshold": G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD,
        "g_spatial_grid_cells_per_axis": G_SPATIAL_GRID_CELLS_PER_AXIS,
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours": round(gpu_hours, 6),
        "device": device,
        "sam_inference_run": True,
        "sam_checkpoint": segmenter_record,
        "fusion_config": {
            "min_area": fusion_config.min_area,
            "iou_nms_threshold": fusion_config.iou_nms_threshold,
        },
        "candidates_csv": str(candidates_csv),
        "best_per_k": {
            str(k): max(
                (c for c in curve if c["k_positive"] == k and c["mean_dice"] is not None),
                key=lambda c: c["mean_dice"],
                default=None,
            )
            for k in k_grid
        },
    }
    (out_dir / "grouping_scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_summary_md(out_dir / "grouping_scaling_summary.md", summary, curve)
    _append_ledger(project_root, gpu_hours)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase X Stage S3: grouping scaling with SAM."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=100)
    args = parser.parse_args()
    summary = run_grouping_scaling(
        project_root=args.project_root,
        candidates_csv=args.candidates_csv,
        episodes_csv=args.episodes_csv,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        device=args.device,
        positive_limit=args.positive_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
