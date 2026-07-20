"""Phase X Stage S4: Negative Scaling.

Implements section 8 Stage S3 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.

Fix the best positive/grouping configuration from Stage S3 (default: K+=20,
``G_ONE_GROUP`` — Stage S2's Dice peak and Stage S3 smoke-test winner). Sweep
K- in the frozen negative K grid and a subset of negative selector
strategies that can be implemented with the existing candidate pool:

- ``N_NONE``: no negatives (K-=0 baseline).
- ``N_BOTTOMK``: take the first K- negatives from the pool (most confident
  background points — lowest foreground score).
- ``N_HIGH_BACKGROUND``: take the last K- negatives from the pool (boundary
  cases — highest-score negatives, closest to the foreground/background
  boundary).

The other three negative selectors (``N_BOUNDARY_HARD``,
``N_LOCAL_CONFUSION``, ``N_DIVERSITY``) require additional candidate
generation logic that is not available in the current pool and are deferred
to a later stage.

For the ``G_ONE_GROUP`` grouping, all K+ positives and K- negatives go in
one SAM call. For multi-group groupings, each negative is assigned to the
group containing its nearest positive (round-robin on ties).

Outputs (section 8 Stage S3 layout)::

    reports/phaseX_negative_scaling/
        negative_scaling_query_metrics.csv
        negative_scaling_curve.csv
        negative_scaling_summary.md
        negative_scaling_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
from adapromptset.scripts.build_phasex_preregistration import NEGATIVE_K_GRID
from adapromptset.scripts.run_phasex_grouping_scaling import split_groups
from adapromptset.scripts.run_phasex_single_variable_kpos import (
    _component_labels,
    _component_recall,
    _semantic_instance_labels,
)
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default positive config from Stage S2 peak (K=20) and Stage S3 smoke test
# (G_ONE_GROUP won at K=20). These can be overridden via CLI after S3
# completes if a different config wins on the full 10-episode aggregate.
DEFAULT_K_POSITIVE = 20
DEFAULT_GROUPING = "G_ONE_GROUP"

# Subset of NEGATIVE_K_GRID that is practical for CPU-only inference.
# Full grid is {0, 1, 3, 5, 10, 20, 30, 50}; we skip 30 and 50 to keep
# runtime manageable (each K- value × 3 strategies × 10 episodes).
DEFAULT_K_NEGATIVE_GRID: tuple[int, ...] = (0, 1, 3, 5, 10, 20)

# Negative selectors implementable with the existing candidate pool.
NEGATIVE_SELECTORS_AVAILABLE: tuple[str, ...] = (
    "N_NONE",
    "N_BOTTOMK",
    "N_HIGH_BACKGROUND",
)


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
                "phase": "phaseX-stage-S4-negative-scaling",
                "command": "scripts/run_phasex_negative_scaling.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Phase X Stage S4: negative scaling with SAM-B on MoNuSAC validation.",
            }
        )


def _select_negatives(
    selector: str,
    k_negative: int,
    negative_coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the K- negative coordinates for the given selector.

    The candidate pool orders negatives by ascending foreground score
    (i.e. first = most confident background, last = boundary case).
    """
    if k_negative == 0 or selector == "N_NONE":
        return []
    if k_negative > len(negative_coords):
        k_negative = len(negative_coords)
    if selector == "N_BOTTOMK":
        return negative_coords[:k_negative]
    if selector == "N_HIGH_BACKGROUND":
        return negative_coords[-k_negative:]
    raise ValueError(f"unknown negative selector: {selector}")


def _assign_negatives_to_groups(
    groups: list[tuple[int, ...]],
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Assign each negative to the group containing its nearest positive.

    Returns a list (parallel to ``groups``) of negative coordinate lists.
    Each negative is assigned to exactly one group.
    """
    if not negative_coords:
        return [[] for _ in groups]
    # Build a flat list of (positive_coord, group_index) for nearest lookup.
    positive_index: list[tuple[tuple[int, int], int]] = []
    for gi, indices in enumerate(groups):
        for pi in indices:
            positive_index.append((positive_coords[pi], gi))
    if not positive_index:
        return [[] for _ in groups]
    assigned: list[list[tuple[int, int]]] = [[] for _ in groups]
    for nx, ny in negative_coords:
        best_gi = 0
        best_dist = math.hypot(
            nx - positive_index[0][0][0], ny - positive_index[0][0][1]
        )
        for i in range(1, len(positive_index)):
            d = math.hypot(
                nx - positive_index[i][0][0], ny - positive_index[i][0][1]
            )
            if d < best_dist:
                best_dist = d
                best_gi = positive_index[i][1]
        assigned[best_gi].append((nx, ny))
    return assigned


def _predict_with_negatives(
    predictor: Any,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
) -> tuple[list[list[bool]], float]:
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    if not points:
        raise ValueError("predict_with_negatives requires at least one point")
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def _run_episode_negative(
    predictor: Any,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    k_positive: int,
    grouping: str,
    k_negative: int,
    selector: str,
    fusion_config: FusionConfig,
    episode_id: str,
) -> dict[str, Any]:
    """Run one (episode, K+, grouping, K-, selector) combo."""
    groups = split_groups(grouping, k_positive, positive_coords)
    if not groups:
        return {
            "episode_id": episode_id,
            "k_positive": k_positive,
            "grouping": grouping,
            "k_negative": k_negative,
            "selector": selector,
            "actual_group_count": 0,
            "sam_calls": 0,
            "actual_negative_count": 0,
            "dice": None,
            "iou": None,
            "aji": None,
            "component_recall": None,
            "target_component_count": target_component_count,
            "hit_components": None,
            "retained_masks": 0,
            "runtime_ms": 0.0,
            "status": "SKIPPED_NO_GROUPS",
        }

    selected_negatives = _select_negatives(selector, k_negative, negative_coords)

    t0 = time.perf_counter()
    candidates: list[MaskCandidate] = []
    if len(groups) == 1:
        # G_ONE_GROUP: all positives + all negatives in one call.
        pos_points = [positive_coords[i] for i in groups[0]]
        mask, score = _predict_with_negatives(
            predictor, pos_points, selected_negatives
        )
        candidates.append(
            MaskCandidate(mask=mask, score=score, source_id=f"{selector}:g0")
        )
    else:
        # Multi-group: assign each negative to nearest positive's group.
        assigned = _assign_negatives_to_groups(
            groups, positive_coords, selected_negatives
        )
        for gi, indices in enumerate(groups):
            pos_points = [positive_coords[i] for i in indices]
            mask, score = _predict_with_negatives(
                predictor, pos_points, assigned[gi]
            )
            candidates.append(
                MaskCandidate(
                    mask=mask, score=score, source_id=f"{selector}:g{gi}"
                )
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

    return {
        "episode_id": episode_id,
        "k_positive": k_positive,
        "grouping": grouping,
        "k_negative": k_negative,
        "selector": selector,
        "actual_group_count": len(groups),
        "sam_calls": len(groups),
        "actual_negative_count": len(selected_negatives),
        "dice": dice,
        "iou": iou,
        "aji": aji,
        "component_recall": recall,
        "target_component_count": total_count,
        "hit_components": hit_count,
        "retained_masks": len(retained),
        "runtime_ms": round(runtime_ms, 3),
        "status": "OK",
    }


def _gate_x4_negative_status(
    curve: list[dict[str, Any]],
) -> tuple[str, str]:
    """Assess negative scaling.

    Compare best (K-, selector) vs N_NONE baseline (K-=0). A gain >= 0.01
    Dice from negatives indicates negative prompts help.
    """
    # Find the N_NONE baseline (K-=0, any selector — but N_NONE is canonical).
    baseline = next(
        (c for c in curve if c["k_negative"] == 0 and c["selector"] == "N_NONE"),
        None,
    )
    if baseline is None or baseline["mean_dice"] is None:
        return (
            "NEGATIVE_INSUFFICIENT_DATA",
            "missing N_NONE baseline (K-=0)",
        )
    valid = [c for c in curve if c["mean_dice"] is not None and c["k_negative"] > 0]
    if not valid:
        return (
            "NEGATIVE_INSUFFICIENT_DATA",
            "no valid K->0 data points",
        )
    best = max(valid, key=lambda c: c["mean_dice"])
    gain = best["mean_dice"] - baseline["mean_dice"]
    if gain >= 0.01:
        return (
            "NEGATIVE_BEST_BEATS_NONE",
            f"best K-={best['k_negative']} selector={best['selector']} "
            f"mean_dice={best['mean_dice']:.4f} vs N_NONE baseline "
            f"mean_dice={baseline['mean_dice']:.4f}; gain={gain:.4f} >= 0.010",
        )
    return (
        "NEGATIVE_NONE_DOMINATES",
        f"best K-={best['k_negative']} selector={best['selector']} "
        f"mean_dice={best['mean_dice']:.4f} vs N_NONE baseline "
        f"mean_dice={baseline['mean_dice']:.4f}; gain={gain:.4f} < 0.010",
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
    lines = [
        "# Phase X Stage S4: Negative Scaling Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: negative_scaling (Gate X4 preliminary, negative axis)",
        f"Episodes: {summary['episode_count']}",
        f"Positive limit: {summary['positive_limit']}",
        f"K+ (fixed): {summary['k_positive']}",
        f"Grouping (fixed): {summary['grouping']}",
        f"Device: {summary['device']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        f"Elapsed seconds: {summary['elapsed_seconds']}",
        f"GPU-hours: {summary['gpu_hours']}",
        "",
        "## Negative selectors (frozen)",
        "",
        f"- Available: {', '.join(summary['negative_selectors'])}",
        "- Deferred (require candidate generation logic): "
        "N_BOUNDARY_HARD, N_LOCAL_CONFUSION, N_DIVERSITY",
        "",
        "## Gate X4 (preliminary, negative axis only)",
        "",
        f"**{summary['gate_x4_negative_status']}**",
        "",
        summary["gate_x4_negative_rationale"],
        "",
        "## Negative scaling curve",
        "",
        "| K- | Selector | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean calls | Mean runtime (ms) | Actual negatives |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in curve:
        lines.append(
            f"| {c['k_negative']} | {c['selector']} | {c['episode_count']} | "
            f"{_format_or_na(c['mean_dice'])} | "
            f"{_format_or_na(c['mean_iou'])} | "
            f"{_format_or_na(c['mean_aji'])} | "
            f"{_format_or_na(c['mean_component_recall'])} | "
            f"{_format_or_na(c['mean_sam_calls'], '.1f')} | "
            f"{_format_or_na(c['mean_runtime_ms'], '.1f')} | "
            f"{_format_or_na(c['mean_actual_negative_count'], '.1f')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `NEGATIVE_BEST_BEATS_NONE`: a (K-, selector) combo beats N_NONE by >= 0.01 Dice.",
            "- `NEGATIVE_NONE_DOMINATES`: N_NONE baseline is within 0.01 Dice of the best negative config.",
            "- `NEGATIVE_INSUFFICIENT_DATA`: missing N_NONE baseline or no valid K->0 data.",
            "",
            "Note: this is a *preliminary* read on the negative axis only. The full Gate X4",
            "verdict requires nested action bank saturation (section 11) and search-oracle",
            "adequacy (section 13).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_negative_scaling(
    *,
    project_root: Path = PROJECT_ROOT,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    device: str = "cpu",
    positive_limit: int = 100,
    k_positive: int = DEFAULT_K_POSITIVE,
    grouping: str = DEFAULT_GROUPING,
    k_negative_grid: Sequence[int] | None = None,
    selectors: Sequence[str] | None = None,
) -> dict[str, Any]:
    workspace_root = project_root.parent
    candidates_csv = candidates_csv or (
        project_root / "reports" / "phaseX_candidate_pilot_100" / "candidate_pilot_candidates.csv"
    )
    episodes_csv = episodes_csv or (
        project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv"
    )
    out_dir = out_dir or (project_root / "reports" / "phaseX_negative_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_CANDIDATE_POOL_MISSING: {candidates_csv}")
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_EPISODES_MISSING: {episodes_csv}")

    episodes = _read_episodes(episodes_csv)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    k_neg_grid = (
        tuple(k_negative_grid) if k_negative_grid is not None else DEFAULT_K_NEGATIVE_GRID
    )
    selector_grid = (
        tuple(selectors) if selectors is not None else NEGATIVE_SELECTORS_AVAILABLE
    )

    fusion_config = load_fusion_config(
        project_root / "configs" / "segmenter" / "fusion_mvp.json"
    )

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
        negative_coords = [c.coordinate for c in pools.negatives]
        if len(positive_coords) < k_positive:
            continue

        predictor.set_image(np.asarray(image))
        episode_count += 1

        for selector in selector_grid:
            for k_neg in k_neg_grid:
                # N_NONE with K->0 is the same regardless of selector; skip
                # duplicates to avoid wasted calls.
                if k_neg == 0 and selector != "N_NONE":
                    continue
                record = _run_episode_negative(
                    predictor,
                    positive_coords,
                    negative_coords,
                    target,
                    target_labels,
                    target_component_count,
                    k_positive=k_positive,
                    grouping=grouping,
                    k_negative=k_neg,
                    selector=selector,
                    fusion_config=fusion_config,
                    episode_id=episode_id,
                )
                records.append(record)

    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0

    # Aggregate by (K-, selector)
    curve: list[dict[str, Any]] = []
    for selector in selector_grid:
        for k_neg in k_neg_grid:
            if k_neg == 0 and selector != "N_NONE":
                continue
            k_rows = [
                r for r in records
                if r["selector"] == selector
                and r["k_negative"] == k_neg
                and r["status"] == "OK"
            ]
            if not k_rows:
                curve.append(
                    {
                        "k_negative": k_neg,
                        "selector": selector,
                        "episode_count": 0,
                        "mean_dice": None,
                        "mean_iou": None,
                        "mean_aji": None,
                        "mean_component_recall": None,
                        "mean_sam_calls": None,
                        "mean_runtime_ms": None,
                        "mean_actual_negative_count": None,
                        "mean_retained_masks": None,
                    }
                )
                continue
            curve.append(
                {
                    "k_negative": k_neg,
                    "selector": selector,
                    "episode_count": len(k_rows),
                    "mean_dice": sum(r["dice"] for r in k_rows) / len(k_rows),
                    "mean_iou": sum(r["iou"] for r in k_rows) / len(k_rows),
                    "mean_aji": sum(r["aji"] for r in k_rows) / len(k_rows),
                    "mean_component_recall": sum(r["component_recall"] for r in k_rows) / len(k_rows),
                    "mean_sam_calls": sum(r["sam_calls"] for r in k_rows) / len(k_rows),
                    "mean_runtime_ms": sum(r["runtime_ms"] for r in k_rows) / len(k_rows),
                    "mean_actual_negative_count": sum(r["actual_negative_count"] for r in k_rows) / len(k_rows),
                    "mean_retained_masks": sum(r["retained_masks"] for r in k_rows) / len(k_rows),
                }
            )

    gate_status, gate_rationale = _gate_x4_negative_status(curve)

    _write_csv(out_dir / "negative_scaling_query_metrics.csv", records)
    _write_csv(out_dir / "negative_scaling_curve.csv", curve)

    summary = {
        "phase": "phaseX",
        "stage": "negative_scaling",
        "gate": "X4_preliminary",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "episode_count": episode_count,
        "positive_limit": positive_limit,
        "k_positive": k_positive,
        "grouping": grouping,
        "k_negative_grid": list(k_neg_grid),
        "negative_selectors": list(selector_grid),
        "gate_x4_negative_status": gate_status,
        "gate_x4_negative_rationale": gate_rationale,
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
        "best_per_k_neg": {
            f"{c['k_negative']}_{c['selector']}": c
            for c in curve
            if c["mean_dice"] is not None
        },
    }
    (out_dir / "negative_scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_summary_md(out_dir / "negative_scaling_summary.md", summary, curve)
    _append_ledger(project_root, gpu_hours)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase X Stage S4: negative scaling with SAM."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=100)
    parser.add_argument("--k-positive", type=int, default=DEFAULT_K_POSITIVE)
    parser.add_argument("--grouping", default=DEFAULT_GROUPING)
    args = parser.parse_args()
    summary = run_negative_scaling(
        project_root=args.project_root,
        candidates_csv=args.candidates_csv,
        episodes_csv=args.episodes_csv,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        device=args.device,
        positive_limit=args.positive_limit,
        k_positive=args.k_positive,
        grouping=args.grouping,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
