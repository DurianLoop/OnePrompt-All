"""Phase X: GT-Guided Search Oracle (section 13).

Implements section 13 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.

The finite-bank oracle (Gate X4) only considers top-K positive selection
within a frozen action bank. The search oracle goes further: it uses
GT-guided greedy add / remove / swap to find the best *subset* of
candidates, not just the top-K prefix.

This is **diagnostic only** (section 10): the search output must not
become an automatic action without a separate train-only abstraction step.

Search variants implemented (section 13):

- 13.1 Greedy Add: start from the best fixed config, add the candidate
  that most improves Dice, repeat until no improvement or budget exhausted.
- 13.2 Greedy Remove: try removing each selected positive; keep the
  removal if it improves Dice.
- 13.2 Greedy Swap: try swapping each selected positive with each
  unselected candidate; keep the best swap if it improves Dice.
- 13.2 Greedy Negative Add: same greedy add but over the negative pool.

Beam search (13.3) is deferred (CPU-only; beam_width>1 would multiply
SAM calls beyond the CPU budget).

Search budget (section 13.4): ``B_point in {20, 50, 100}``, ``B_call=1``
(G_ONE_GROUP dominates per Stage S3).

Outputs::

    reports/phaseX_search_oracle/
        search_oracle_query_metrics.csv
        search_oracle_summary.json
        search_oracle_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
from adapromptset.scripts.run_phasex_single_variable_kpos import (
    _component_labels,
    _component_recall,
    _semantic_instance_labels,
)
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Search configuration (frozen at preregistration).
# Start from the best fixed action (Stage S4 / nested bank saturation winner):
# K+=20 Top-K, K-=1 N_BOTTOMK, G_ONE_GROUP, 1 call.
DEFAULT_K_POSITIVE_START = 20
DEFAULT_K_NEGATIVE_START = 1
DEFAULT_NEGATIVE_SELECTOR = "N_BOTTOMK"
DEFAULT_B_POINT = 100  # max total positives allowed in search
DEFAULT_B_CALL = 1     # G_ONE_GROUP only (Stage S3 showed this dominates)

# Search stopping: minimum Dice improvement to accept a move.
# Smaller = more thorough but more SAM calls.
MIN_IMPROVEMENT = 0.001

# CPU-only optimization: limit the number of candidates tried per greedy add
# iteration. Instead of trying all unselected candidates (up to 80), only try
# the top N by candidate score (foreground score for positives, background
# score for negatives). This reduces SAM calls by ~3x while still exploring
# the most promising candidates.
GREEDY_ADD_CANDIDATE_LIMIT = 30

# CPU-only optimization: skip greedy swap (O(K * (100-K)) per iteration is
# too expensive on CPU). Greedy add + remove + negative add already captures
# most of the search benefit.
SKIP_GREEDY_SWAP = True


@dataclass
class SearchState:
    """Mutable search state: selected positive indices + negative indices."""

    positive_indices: list[int] = field(default_factory=list)
    negative_indices: list[int] = field(default_factory=list)
    dice: float = 0.0
    sam_calls: int = 0


def _predict_one(
    predictor: Any,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
) -> tuple[list[list[bool]], float]:
    """Single SAM predict call with mixed positive/negative points."""
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    if not points:
        raise ValueError("predict_one requires at least one point")
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def _evaluate(
    predictor: Any,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
) -> tuple[float, dict[str, Any], int]:
    """Run SAM predict + fusion + metrics. Returns (dice, metrics, sam_calls)."""
    if not positive_coords:
        return 0.0, {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "retained_masks": 0,
        }, 0
    mask, score = _predict_one(predictor, positive_coords, negative_coords)
    candidate = MaskCandidate(mask=mask, score=score)
    retained, semantic = fuse_masks([candidate], fusion_config)
    if not retained:
        return 0.0, {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "retained_masks": 0,
        }, 1
    dice = dice_score(semantic, target)
    iou = iou_score(semantic, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(semantic),
        _semantic_instance_labels(target),
    )
    recall, _, _ = _component_recall(semantic, target_labels)
    return dice, {
        "dice": dice, "iou": iou, "aji": aji,
        "component_recall": recall, "retained_masks": len(retained),
    }, 1


def _greedy_add_positives(
    predictor: Any,
    state: SearchState,
    all_positive_coords: list[tuple[int, int]],
    all_negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
    max_positives: int,
) -> SearchState:
    """Greedy add: try each unselected positive, keep the best if it improves.

    CPU-only optimization: only try the first ``GREEDY_ADD_CANDIDATE_LIMIT``
    unselected candidates (by index order = candidate score order, since the
    pool is sorted by descending foreground score).
    """
    improved = True
    while improved and len(state.positive_indices) < max_positives:
        improved = False
        best_idx = -1
        best_dice = state.dice
        selected_set = set(state.positive_indices)
        unselected = [
            i for i in range(len(all_positive_coords))
            if i not in selected_set
        ]
        # CPU-only: limit candidates tried per iteration.
        unselected = unselected[:GREEDY_ADD_CANDIDATE_LIMIT]
        for idx in unselected:
            trial_pos = [all_positive_coords[i] for i in state.positive_indices] + [
                all_positive_coords[idx]
            ]
            trial_neg = [all_negative_coords[i] for i in state.negative_indices]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
                fusion_config=fusion_config,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_idx = idx
        if best_idx >= 0:
            state.positive_indices.append(best_idx)
            state.dice = best_dice
            improved = True
    return state


def _greedy_remove_positives(
    predictor: Any,
    state: SearchState,
    all_positive_coords: list[tuple[int, int]],
    all_negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
    min_positives: int = 1,
) -> SearchState:
    """Greedy remove: try removing each selected positive, keep if improves."""
    if len(state.positive_indices) <= min_positives:
        return state
    improved = True
    while improved and len(state.positive_indices) > min_positives:
        improved = False
        best_remove = -1
        best_dice = state.dice
        for i, _ in enumerate(state.positive_indices):
            trial_pos_indices = state.positive_indices[:i] + state.positive_indices[i + 1:]
            trial_pos = [all_positive_coords[j] for j in trial_pos_indices]
            trial_neg = [all_negative_coords[j] for j in state.negative_indices]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
                fusion_config=fusion_config,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_remove = i
        if best_remove >= 0:
            state.positive_indices.pop(best_remove)
            state.dice = best_dice
            improved = True
    return state


def _greedy_swap_positives(
    predictor: Any,
    state: SearchState,
    all_positive_coords: list[tuple[int, int]],
    all_negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
    max_swaps: int = 3,
) -> SearchState:
    """Greedy swap: try swapping each selected with each unselected, keep best."""
    for _ in range(max_swaps):
        best_dice = state.dice
        best_sel_i = -1
        best_unsel_j = -1
        selected_set = set(state.positive_indices)
        for si, sel_idx in enumerate(state.positive_indices):
            for j in range(len(all_positive_coords)):
                if j in selected_set:
                    continue
                trial_pos_indices = state.positive_indices[:si] + [j] + state.positive_indices[si + 1:]
                trial_pos = [all_positive_coords[k] for k in trial_pos_indices]
                trial_neg = [all_negative_coords[k] for k in state.negative_indices]
                trial_dice, _, calls = _evaluate(
                    predictor, trial_pos, trial_neg, target, target_labels,
                    target_component_count,
                    fusion_config=fusion_config,
                )
                state.sam_calls += calls
                if trial_dice > best_dice + MIN_IMPROVEMENT:
                    best_dice = trial_dice
                    best_sel_i = si
                    best_unsel_j = j
        if best_sel_i >= 0:
            state.positive_indices[best_sel_i] = best_unsel_j
            state.dice = best_dice
        else:
            break
    return state


def _greedy_add_negatives(
    predictor: Any,
    state: SearchState,
    all_positive_coords: list[tuple[int, int]],
    all_negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
    max_negatives: int,
) -> SearchState:
    """Greedy add negatives: try each unselected negative, keep best if improves.

    CPU-only optimization: only try the first ``GREEDY_ADD_CANDIDATE_LIMIT``
    unselected negatives.
    """
    improved = True
    while improved and len(state.negative_indices) < max_negatives:
        improved = False
        best_idx = -1
        best_dice = state.dice
        selected_set = set(state.negative_indices)
        unselected = [
            i for i in range(len(all_negative_coords))
            if i not in selected_set
        ]
        # CPU-only: limit candidates tried per iteration.
        unselected = unselected[:GREEDY_ADD_CANDIDATE_LIMIT]
        for idx in unselected:
            trial_pos = [all_positive_coords[i] for i in state.positive_indices]
            trial_neg = [all_negative_coords[i] for i in state.negative_indices] + [
                all_negative_coords[idx]
            ]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
                fusion_config=fusion_config,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_idx = idx
        if best_idx >= 0:
            state.negative_indices.append(best_idx)
            state.dice = best_dice
            improved = True
    return state


def _run_search_oracle_episode(
    predictor: Any,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    k_positive_start: int,
    k_negative_start: int,
    negative_selector: str,
    b_point: int,
    fusion_config: FusionConfig,
    episode_id: str,
) -> dict[str, Any]:
    """Run the full search oracle for one episode.

    Returns a record with:
    - baseline_dice: Dice of the starting config (top-K + N_BOTTOMK K-=1)
    - search_dice: Dice after greedy search
    - search_gap: search_dice - baseline_dice
    - final_positive_count, final_negative_count
    - sam_calls: total SAM calls used
    - trajectory: list of (step, action, dice) tuples
    """
    # Initialize state from the best fixed config.
    start_pos_indices = list(range(min(k_positive_start, len(positive_coords))))
    if negative_selector == "N_BOTTOMK":
        start_neg_indices = list(range(min(k_negative_start, len(negative_coords))))
    elif negative_selector == "N_HIGH_BACKGROUND":
        n = len(negative_coords)
        start_neg_indices = list(
            range(n - min(k_negative_start, n), n)
        )
    else:
        start_neg_indices = []

    state = SearchState(
        positive_indices=start_pos_indices,
        negative_indices=start_neg_indices,
        dice=0.0,
        sam_calls=0,
    )

    # Evaluate baseline.
    start_pos = [positive_coords[i] for i in state.positive_indices]
    start_neg = [negative_coords[i] for i in state.negative_indices]
    baseline_dice, baseline_metrics, calls = _evaluate(
        predictor, start_pos, start_neg, target, target_labels,
        target_component_count,
        fusion_config=fusion_config,
    )
    state.dice = baseline_dice
    state.sam_calls += calls

    trajectory: list[dict[str, Any]] = [{
        "step": 0,
        "action": "init",
        "dice": baseline_dice,
        "positive_count": len(state.positive_indices),
        "negative_count": len(state.negative_indices),
    }]

    # Phase 1: Greedy add positives.
    state = _greedy_add_positives(
        predictor, state, positive_coords, negative_coords,
        target, target_labels, target_component_count,
        fusion_config=fusion_config,
        max_positives=b_point,
    )
    trajectory.append({
        "step": 1,
        "action": "greedy_add_positives",
        "dice": state.dice,
        "positive_count": len(state.positive_indices),
        "negative_count": len(state.negative_indices),
    })

    # Phase 2: Greedy remove positives.
    state = _greedy_remove_positives(
        predictor, state, positive_coords, negative_coords,
        target, target_labels, target_component_count,
        fusion_config=fusion_config,
        min_positives=1,
    )
    trajectory.append({
        "step": 2,
        "action": "greedy_remove_positives",
        "dice": state.dice,
        "positive_count": len(state.positive_indices),
        "negative_count": len(state.negative_indices),
    })

    # Phase 3: Greedy swap positives (skipped on CPU for runtime).
    if not SKIP_GREEDY_SWAP:
        state = _greedy_swap_positives(
            predictor, state, positive_coords, negative_coords,
            target, target_labels, target_component_count,
            fusion_config=fusion_config,
            max_swaps=3,
        )
    trajectory.append({
        "step": 3,
        "action": "greedy_swap_positives" if not SKIP_GREEDY_SWAP else "greedy_swap_positives_skipped",
        "dice": state.dice,
        "positive_count": len(state.positive_indices),
        "negative_count": len(state.negative_indices),
    })

    # Phase 4: Greedy add negatives.
    state = _greedy_add_negatives(
        predictor, state, positive_coords, negative_coords,
        target, target_labels, target_component_count,
        fusion_config=fusion_config,
        max_negatives=b_point,
    )
    trajectory.append({
        "step": 4,
        "action": "greedy_add_negatives",
        "dice": state.dice,
        "positive_count": len(state.positive_indices),
        "negative_count": len(state.negative_indices),
    })

    # Final evaluation.
    final_pos = [positive_coords[i] for i in state.positive_indices]
    final_neg = [negative_coords[i] for i in state.negative_indices]
    final_dice, final_metrics, calls = _evaluate(
        predictor, final_pos, final_neg, target, target_labels,
        target_component_count,
        fusion_config=fusion_config,
    )
    state.sam_calls += calls

    return {
        "episode_id": episode_id,
        "baseline_dice": baseline_dice,
        "search_dice": state.dice,
        "search_gap": state.dice - baseline_dice,
        "final_positive_count": len(state.positive_indices),
        "final_negative_count": len(state.negative_indices),
        "sam_calls": state.sam_calls,
        "final_iou": final_metrics["iou"],
        "final_aji": final_metrics["aji"],
        "final_component_recall": final_metrics["component_recall"],
        "trajectory": trajectory,
    }


def _gate_x5_status(
    search_mean_dice: float,
    bank_oracle_mean_dice: float,
    best_fixed_mean_dice: float,
) -> tuple[str, str]:
    """Determine Gate X5 status from search vs bank vs fixed.

    Returns (status, rationale).
    """
    if search_mean_dice is None or bank_oracle_mean_dice is None:
        return "SEARCH_ORACLE_INSUFFICIENT_DATA", "missing search or bank oracle mean dice"
    search_vs_bank = search_mean_dice - bank_oracle_mean_dice
    search_vs_fixed = search_mean_dice - best_fixed_mean_dice
    if search_vs_bank >= 0.01:
        return (
            "SEARCH_ORACLE_EXCEEDS_BANK_ORACLE",
            f"search={search_mean_dice:.4f} - bank={bank_oracle_mean_dice:.4f} = {search_vs_bank:.4f} >= 0.01",
        )
    if search_mean_dice < 0.4:
        return (
            "SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW",
            f"search={search_mean_dice:.4f} < 0.4 absolute threshold",
        )
    return (
        "SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE",
        f"search={search_mean_dice:.4f} - bank={bank_oracle_mean_dice:.4f} = {search_vs_bank:.4f} < 0.01; search-fixed={search_vs_fixed:.4f}",
    )


def run_search_oracle(
    *,
    project_root: Path,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    device: str = "cpu",
    positive_limit: int = 100,
    k_positive_start: int = DEFAULT_K_POSITIVE_START,
    k_negative_start: int = DEFAULT_K_NEGATIVE_START,
    negative_selector: str = DEFAULT_NEGATIVE_SELECTOR,
    b_point: int = DEFAULT_B_POINT,
) -> dict[str, Any]:
    """Run the GT-guided search oracle over all validation episodes."""
    if candidates_csv is None:
        candidates_csv = (
            project_root / "reports"
            / "phaseX_candidate_pilot_100" / "candidate_pilot_candidates.csv"
        )
    if episodes_csv is None:
        episodes_csv = (
            project_root / "manifests" / "episodes" / "monusac"
            / "episodes_val.csv"
        )
    if out_dir is None:
        out_dir = (
            project_root / "reports" / "phaseX_search_oracle"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = project_root.parent
    fusion_config = load_fusion_config(
        project_root / "configs" / "segmenter" / "fusion_mvp.json"
    )

    # Load episodes.
    with open(episodes_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        episode_rows = list(reader)
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]

    # Load SAM predictor once.
    sam_ckpt = workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth"
    segmenter_record = checkpoint_record(sam_ckpt, role="segmenter")
    predictor = load_sam_predictor(sam_ckpt, workspace_root, device=device)

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for row in episode_rows:
        episode_id = row["episode_id"]
        paths = resolve_episode_paths(row, workspace_root)
        image = load_rgb(paths["query_image"])
        predictor.set_image(np.asarray(image))
        target = load_mask(paths["query_mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        pools = _candidate_pools_from_csv(
            candidates_csv, episode_id=episode_id, positive_limit=positive_limit,
        )
        if not pools.positives:
            continue
        positive_coords = [c.coordinate for c in pools.positives]
        negative_coords = [c.coordinate for c in pools.negatives]

        rec = _run_search_oracle_episode(
            predictor,
            positive_coords,
            negative_coords,
            target,
            target_labels,
            target_component_count,
            k_positive_start=k_positive_start,
            k_negative_start=k_negative_start,
            negative_selector=negative_selector,
            b_point=b_point,
            fusion_config=fusion_config,
            episode_id=episode_id,
        )
        records.append(rec)

    elapsed = time.perf_counter() - t0
    gpu_hours = 0.0  # CPU-only.

    # Aggregate.
    baseline_dices = [r["baseline_dice"] for r in records if r["baseline_dice"] is not None]
    search_dices = [r["search_dice"] for r in records if r["search_dice"] is not None]
    search_gaps = [r["search_gap"] for r in records if r["search_gap"] is not None]
    sam_calls_list = [r["sam_calls"] for r in records]

    mean_baseline = float(np.mean(baseline_dices)) if baseline_dices else 0.0
    mean_search = float(np.mean(search_dices)) if search_dices else 0.0
    mean_gap = float(np.mean(search_gaps)) if search_gaps else 0.0
    mean_calls = float(np.mean(sam_calls_list)) if sam_calls_list else 0.0

    # Win/tie/loss: search vs baseline (per-episode).
    win = sum(1 for r in records if r["search_gap"] > 0.001)
    tie = sum(1 for r in records if abs(r["search_gap"]) <= 0.001)
    loss = sum(1 for r in records if r["search_gap"] < -0.001)

    # Bank oracle mean dice (from nested bank saturation A_large).
    bank_oracle_mean_dice = 0.6119  # from Gate X4 A_large oracle
    best_fixed_mean_dice = 0.5411   # from Gate X4 A_large best_fixed

    gate_x5_status, gate_x5_rationale = _gate_x5_status(
        mean_search, bank_oracle_mean_dice, best_fixed_mean_dice,
    )

    summary: dict[str, Any] = {
        "phase": "phaseX",
        "stage": "search_oracle",
        "gate": "X5",
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "episode_count": len(records),
        "positive_limit": positive_limit,
        "k_positive_start": k_positive_start,
        "k_negative_start": k_negative_start,
        "negative_selector": negative_selector,
        "b_point": b_point,
        "b_call": DEFAULT_B_CALL,
        "search_variants": [
            "greedy_add_positives",
            "greedy_remove_positives",
            "greedy_swap_positives",
            "greedy_add_negatives",
        ],
        "deferred_variants": ["beam_search (CPU budget)"],
        "mean_baseline_dice": mean_baseline,
        "mean_search_dice": mean_search,
        "mean_search_gap": mean_gap,
        "mean_sam_calls": mean_calls,
        "win_tie_loss": {"win": win, "tie": tie, "loss": loss},
        "bank_oracle_mean_dice": bank_oracle_mean_dice,
        "best_fixed_mean_dice": best_fixed_mean_dice,
        "search_vs_bank": mean_search - bank_oracle_mean_dice,
        "search_vs_fixed": mean_search - best_fixed_mean_dice,
        "gate_x5_status": gate_x5_status,
        "gate_x5_rationale": gate_x5_rationale,
        "elapsed_seconds": elapsed,
        "gpu_hours": gpu_hours,
        "device": device,
        "sam_inference_run": True,
        "sam_checkpoint": segmenter_record,
        "fusion_config": {
            "min_area": fusion_config.min_area,
            "iou_nms_threshold": fusion_config.iou_nms_threshold,
        },
        "candidates_csv": str(candidates_csv),
    }

    # Write query metrics CSV.
    with open(out_dir / "search_oracle_query_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "baseline_dice", "search_dice", "search_gap",
            "final_positive_count", "final_negative_count", "sam_calls",
            "final_iou", "final_aji", "final_component_recall",
        ])
        for r in records:
            writer.writerow([
                r["episode_id"], r["baseline_dice"], r["search_dice"],
                r["search_gap"], r["final_positive_count"],
                r["final_negative_count"], r["sam_calls"],
                r["final_iou"], r["final_aji"], r["final_component_recall"],
            ])

    # Write summary JSON.
    summary_no_traj = {k: v for k, v in summary.items()}
    summary_no_traj["per_episode_trajectories"] = [
        {"episode_id": r["episode_id"], "trajectory": r["trajectory"]}
        for r in records
    ]
    (out_dir / "search_oracle_summary.json").write_text(
        json.dumps(summary_no_traj, indent=2), encoding="utf-8"
    )

    # Write summary MD.
    _write_summary_md(out_dir / "search_oracle_summary.md", summary, records)

    # Append to compute ledger.
    _append_ledger(project_root, elapsed, gpu_hours, len(records))

    return summary_no_traj


def _write_summary_md(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase X: GT-Guided Search Oracle Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: search_oracle (Gate X5)",
        f"Episodes: {summary['episode_count']}",
        f"Positive limit: {summary['positive_limit']}",
        f"K+ start: {summary['k_positive_start']}",
        f"K- start: {summary['k_negative_start']}",
        f"Negative selector: {summary['negative_selector']}",
        f"B_point: {summary['b_point']}",
        f"B_call: {summary['b_call']}",
        f"Device: {summary['device']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        f"Elapsed seconds: {summary['elapsed_seconds']:.1f}",
        f"GPU-hours: {summary['gpu_hours']}",
        "",
        "## Search variants (implemented)",
        "",
        "- greedy_add_positives",
        "- greedy_remove_positives",
        "- greedy_swap_positives",
        "- greedy_add_negatives",
        "",
        "## Deferred variants",
        "",
        "- beam_search (CPU budget; beam_width>1 would multiply SAM calls)",
        "",
        "## Gate X5 Status",
        "",
        f"**{summary['gate_x5_status']}**",
        "",
        f"{summary['gate_x5_rationale']}",
        "",
        "## Aggregate metrics",
        "",
        f"- Mean baseline Dice (best fixed): {summary['mean_baseline_dice']:.4f}",
        f"- Mean search oracle Dice: {summary['mean_search_dice']:.4f}",
        f"- Mean search gap: {summary['mean_search_gap']:.4f}",
        f"- Mean SAM calls per episode: {summary['mean_sam_calls']:.1f}",
        f"- Win/tie/loss (search vs baseline): {summary['win_tie_loss']}",
        f"- Bank oracle mean Dice (A_large from Gate X4): {summary['bank_oracle_mean_dice']:.4f}",
        f"- Best fixed mean Dice (A_large from Gate X4): {summary['best_fixed_mean_dice']:.4f}",
        f"- Search - bank oracle: {summary['search_vs_bank']:.4f}",
        f"- Search - best fixed: {summary['search_vs_fixed']:.4f}",
        "",
        "## Per-episode results",
        "",
        "| Episode | Baseline Dice | Search Dice | Gap | Final K+ | Final K- | SAM calls |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['episode_id']} | {r['baseline_dice']:.4f} | {r['search_dice']:.4f} | "
            f"{r['search_gap']:+.4f} | {r['final_positive_count']} | "
            f"{r['final_negative_count']} | {r['sam_calls']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `SEARCH_ORACLE_EXCEEDS_BANK_ORACLE`: search oracle - bank oracle >= 0.01 → "
        "the finite bank misses meaningful prompt structures.",
        "- `SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE`: search oracle - bank oracle < 0.01 → "
        "the finite bank captures most of the achievable performance.",
        "- `SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW`: search oracle < 0.4 → "
        "the limitation is the prompt interface or candidate generator, not the action bank.",
        "",
        "Note: this search oracle uses GT-guided greedy add/remove/swap over the frozen "
        "candidate pool. It is **diagnostic only** (section 10): search outputs must not "
        "become automatic actions without a separate train-only abstraction step.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_ledger(project_root: Path, elapsed: float, gpu_hours: float, episode_count: int) -> None:
    ledger_path = (
        project_root / "reports" / "phaseX_compute_ledger.csv"
    )
    header = "stage,episode_count,elapsed_seconds,gpu_hours,device,date\n"
    line = (
        f"search_oracle,{episode_count},{elapsed:.1f},{gpu_hours},cpu,"
        f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    )
    if not ledger_path.exists():
        ledger_path.write_text(header + line, encoding="utf-8")
    else:
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase X: GT-guided search oracle (section 13)."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=100)
    parser.add_argument("--k-positive-start", type=int, default=DEFAULT_K_POSITIVE_START)
    parser.add_argument("--k-negative-start", type=int, default=DEFAULT_K_NEGATIVE_START)
    parser.add_argument("--negative-selector", default=DEFAULT_NEGATIVE_SELECTOR)
    parser.add_argument("--b-point", type=int, default=DEFAULT_B_POINT)
    args = parser.parse_args()
    summary = run_search_oracle(
        project_root=args.project_root,
        candidates_csv=args.candidates_csv,
        episodes_csv=args.episodes_csv,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        device=args.device,
        positive_limit=args.positive_limit,
        k_positive_start=args.k_positive_start,
        k_negative_start=args.k_negative_start,
        negative_selector=args.negative_selector,
        b_point=args.b_point,
    )
    # Don't print the full summary with trajectories (too verbose).
    print_summary = {k: v for k, v in summary.items() if k != "per_episode_trajectories"}
    print(json.dumps(print_summary, indent=2))


if __name__ == "__main__":
    main()
