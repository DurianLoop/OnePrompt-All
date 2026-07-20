"""Phase X: Nested Action Bank Saturation (Section 11).

Implements section 11 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.

Builds three nested action banks ``A_small ⊂ A_medium ⊂ A_large`` and
computes the fixed-baseline saturation curve and finite-bank oracle
saturation curve for each. The key output is the oracle gap per bank,
which answers the central Phase X question: does the oracle gap shrink
as the action space grows (saturation), or does it stay large (the old
bank was inadequate)?

Bank constraints (frozen at preregistration):

- ``A_small``: K+ ≤ 5, K- ≤ 4, calls ≤ 3 (matches historical 12-action bank)
- ``A_medium``: K+ ≤ 20, K- ≤ 10, calls ≤ 5
- ``A_large``: K+ ≤ 100, K- ≤ 30, calls ≤ 20 (includes dense grouping)

To keep CPU-only runtime manageable, a *representative* action subset is
used per bank rather than the full Cartesian product. The subset is
designed to:

1. Include the historical 12-action bank's strongest configs (K+ ≤ 5,
   G_ONE_GROUP).
2. Include the Stage S2 Dice peak (K+=20, G_ONE_GROUP).
3. Include a spread of groupings to capture grouping-driven oracle gains.
4. Include both N_NONE and N_BOTTOMK to capture negative-driven gains.

Outputs (section 11 layout)::

    reports/phaseX_nested_bank_saturation/
        nested_bank_query_metrics.csv
        nested_bank_saturation_curve.csv
        nested_bank_summary.md
        nested_bank_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
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
from adapromptset.scripts.run_phasex_grouping_scaling import split_groups
from adapromptset.scripts.run_phasex_negative_scaling import (
    _assign_negatives_to_groups,
    _predict_with_negatives,
    _select_negatives,
)
from adapromptset.scripts.run_phasex_single_variable_kpos import (
    _component_labels,
    _component_recall,
    _semantic_instance_labels,
)
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Bank definitions (frozen at preregistration per section 10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankSpec:
    name: str
    max_positive: int
    max_negative: int
    max_calls: int


A_SMALL = BankSpec("A_small", max_positive=5, max_negative=4, max_calls=3)
A_MEDIUM = BankSpec("A_medium", max_positive=20, max_negative=10, max_calls=5)
A_LARGE = BankSpec("A_large", max_positive=100, max_negative=30, max_calls=20)
NESTED_BANKS: tuple[BankSpec, ...] = (A_SMALL, A_MEDIUM, A_LARGE)


# ---------------------------------------------------------------------------
# Representative action set per bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    bank: str
    k_positive: int
    k_negative: int
    grouping: str
    selector: str

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id required")
        if self.k_positive <= 0:
            raise ValueError("k_positive must be positive")
        if self.k_negative < 0:
            raise ValueError("k_negative must be non-negative")
        if self.selector == "N_NONE" and self.k_negative > 0:
            raise ValueError("N_NONE with k_negative > 0 is inconsistent")


def _action_set_for_bank(bank: BankSpec) -> list[ActionSpec]:
    """Return a representative action subset for the given bank.

    The subset is designed to cover:
    - The historical strongest configs (K+ ≤ 5, G_ONE_GROUP).
    - The Stage S2 Dice peak (K+=20, G_ONE_GROUP).
    - A spread of groupings to capture grouping-driven oracle gains.
    - Both N_NONE and N_BOTTOMK to capture negative-driven gains.

    Only groupings whose call count fits within ``bank.max_calls`` are
    included. For example, G_KMEANS_20 (20 calls) is only in A_large.
    """
    actions: list[ActionSpec] = []
    # G_ONE_GROUP (1 call) fits all banks.
    k_pos_grid = _bank_k_positive_grid(bank)
    k_neg_grid = _bank_k_negative_grid(bank)
    for k_pos in k_pos_grid:
        for k_neg in k_neg_grid:
            for selector in ("N_NONE", "N_BOTTOMK"):
                if selector == "N_NONE" and k_neg > 0:
                    continue
                if selector == "N_BOTTOMK" and k_neg == 0:
                    continue
                actions.append(
                    ActionSpec(
                        action_id=f"{bank.name}__G_ONE_GROUP__K{ k_pos}__N{ k_neg}__{selector}",
                        bank=bank.name,
                        k_positive=k_pos,
                        k_negative=k_neg,
                        grouping="G_ONE_GROUP",
                        selector=selector,
                    )
                )
    # Add multi-group configs that fit within the bank's call budget.
    # G_FIXED_CHUNK_5 has ceil(K+/5) calls. G_KMEANS_k has k calls.
    multi_groupings = [
        ("G_FIXED_CHUNK_5", 5),  # ceil(K+/5) calls
        ("G_KMEANS_3", 3),
        ("G_KMEANS_5", 5),
        ("G_KMEANS_10", 10),
        ("G_KMEANS_20", 20),
        ("G_SPATIAL_GRID", 4),  # 2x2 grid = 4 cells
    ]
    for grouping, calls_per_grouping in multi_groupings:
        if calls_per_grouping > bank.max_calls:
            continue
        # For each K+, check if the grouping's call count fits.
        for k_pos in k_pos_grid:
            if grouping.startswith("G_FIXED_CHUNK_"):
                chunk_size = int(grouping.rsplit("_", 1)[-1])
                actual_calls = (k_pos + chunk_size - 1) // chunk_size
            else:
                actual_calls = calls_per_grouping
            if actual_calls > bank.max_calls:
                continue
            # Only add N_NONE for multi-group (negatives add complexity).
            actions.append(
                ActionSpec(
                    action_id=f"{bank.name}__{grouping}__K{ k_pos}__N0__N_NONE",
                    bank=bank.name,
                    k_positive=k_pos,
                    k_negative=0,
                    grouping=grouping,
                    selector="N_NONE",
                )
            )
    return actions


def _bank_k_positive_grid(bank: BankSpec) -> tuple[int, ...]:
    """Return the K+ grid for the bank (within bank.max_positive).

    Grids are cumulative across banks to preserve the strict nesting property
    A_small ⊂ A_medium ⊂ A_large: every K+ valid in a smaller bank is also
    represented in larger banks. A_medium therefore includes A_small's K+
    values (1, 3, 5) plus its own extensions (10, 20); A_large includes all
    of A_medium's values plus (50, 100).
    """
    if bank.name == "A_small":
        return (1, 3, 5)
    if bank.name == "A_medium":
        return (1, 3, 5, 10, 20)
    if bank.name == "A_large":
        return (1, 3, 5, 10, 20, 50, 100)
    raise ValueError(f"unknown bank: {bank.name}")


def _bank_k_negative_grid(bank: BankSpec) -> tuple[int, ...]:
    """Return the K- grid for the bank (within bank.max_negative).

    Grids are cumulative across banks to preserve the strict nesting property
    A_small ⊂ A_medium ⊂ A_large: every action valid in a smaller bank must
    also be representable in a larger bank. A_large therefore includes all
    K- values from A_medium (0, 1, 3, 5, 10) plus its own extensions (20, 30).
    """
    if bank.name == "A_small":
        return (0, 1, 3)
    if bank.name == "A_medium":
        return (0, 1, 3, 5, 10)
    if bank.name == "A_large":
        return (0, 1, 3, 5, 10, 20, 30)
    raise ValueError(f"unknown bank: {bank.name}")


# ---------------------------------------------------------------------------
# SAM inference + fusion (reuses Stage S4 logic)
# ---------------------------------------------------------------------------


def _run_action(
    predictor: Any,
    action: ActionSpec,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    fusion_config: FusionConfig,
    episode_id: str,
) -> dict[str, Any]:
    """Run one (episode, action) combo. Returns a metrics record."""
    groups = split_groups(action.grouping, action.k_positive, positive_coords)
    if not groups:
        return _invalid_record(action, episode_id, target_component_count)

    selected_negatives = _select_negatives(
        action.selector, action.k_negative, negative_coords
    )

    candidates: list[MaskCandidate] = []
    t0 = time.perf_counter()
    if len(groups) == 1:
        pos_points = [positive_coords[i] for i in groups[0]]
        try:
            mask, score = _predict_with_negatives(
                predictor, pos_points, selected_negatives
            )
        except ValueError:
            return _invalid_record(action, episode_id, target_component_count)
        candidates.append(
            MaskCandidate(mask=mask, score=score, source_id=f"{action.action_id}:g0")
        )
    else:
        assigned = _assign_negatives_to_groups(
            groups, positive_coords, selected_negatives
        )
        for gi, indices in enumerate(groups):
            pos_points = [positive_coords[i] for i in indices]
            try:
                mask, score = _predict_with_negatives(
                    predictor, pos_points, assigned[gi]
                )
            except ValueError:
                return _invalid_record(action, episode_id, target_component_count)
            candidates.append(
                MaskCandidate(
                    mask=mask, score=score, source_id=f"{action.action_id}:g{gi}"
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
        "action_id": action.action_id,
        "bank": action.bank,
        "k_positive": action.k_positive,
        "k_negative": action.k_negative,
        "grouping": action.grouping,
        "selector": action.selector,
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


def _invalid_record(
    action: ActionSpec,
    episode_id: str,
    target_component_count: int,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "action_id": action.action_id,
        "bank": action.bank,
        "k_positive": action.k_positive,
        "k_negative": action.k_negative,
        "grouping": action.grouping,
        "selector": action.selector,
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
        "status": "SKIPPED_INVALID",
    }


# ---------------------------------------------------------------------------
# Saturation analysis
# ---------------------------------------------------------------------------


def _bank_saturation(
    records: list[dict[str, Any]],
    bank_name: str,
) -> dict[str, Any]:
    """Compute best fixed, oracle, and gap for one bank."""
    bank_records = [
        r for r in records if r["bank"] == bank_name and r["status"] == "OK"
    ]
    if not bank_records:
        return {
            "bank": bank_name,
            "action_count": 0,
            "episode_count": 0,
            "best_fixed_action_id": None,
            "best_fixed_mean_dice": None,
            "oracle_mean_dice": None,
            "oracle_gap": None,
        }

    # Group by action_id to compute per-action mean Dice.
    by_action: dict[str, list[dict[str, Any]]] = {}
    for r in bank_records:
        by_action.setdefault(r["action_id"], []).append(r)
    action_mean_dice = {
        aid: sum(r["dice"] for r in rows) / len(rows)
        for aid, rows in by_action.items()
    }
    best_fixed_action = max(action_mean_dice, key=action_mean_dice.get)
    best_fixed_dice = action_mean_dice[best_fixed_action]

    # Per-episode oracle: best action per episode.
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for r in bank_records:
        by_episode.setdefault(r["episode_id"], []).append(r)
    oracle_dices: list[float] = []
    oracle_action_counts: dict[str, int] = {}
    for episode_id, rows in by_episode.items():
        best = max(rows, key=lambda r: r["dice"])
        oracle_dices.append(best["dice"])
        oracle_action_counts[best["action_id"]] = (
            oracle_action_counts.get(best["action_id"], 0) + 1
        )
    oracle_mean = sum(oracle_dices) / len(oracle_dices)
    gap = oracle_mean - best_fixed_dice

    # Best fixed action's per-episode Dice for win/tie/loss analysis.
    best_fixed_rows = by_action[best_fixed_action]
    best_fixed_by_episode = {r["episode_id"]: r["dice"] for r in best_fixed_rows}
    win_tie_loss = {"win": 0, "tie": 0, "loss": 0}
    for episode_id, rows in by_episode.items():
        oracle_best = max(r["dice"] for r in rows)
        fixed = best_fixed_by_episode.get(episode_id)
        if fixed is None:
            continue
        if oracle_best - fixed > 0.001:
            win_tie_loss["win"] += 1
        elif fixed - oracle_best > 0.001:
            win_tie_loss["loss"] += 1
        else:
            win_tie_loss["tie"] += 1

    # Cost of best fixed action.
    best_fixed_sam_calls = max(r["sam_calls"] for r in best_fixed_rows)
    best_fixed_runtime = sum(r["runtime_ms"] for r in best_fixed_rows) / len(best_fixed_rows)

    # Oracle's mean cost (mean of per-episode best actions' costs).
    oracle_sam_calls = []
    for episode_id, rows in by_episode.items():
        best = max(rows, key=lambda r: r["dice"])
        oracle_sam_calls.append(best["sam_calls"])
    oracle_mean_calls = sum(oracle_sam_calls) / len(oracle_sam_calls) if oracle_sam_calls else 0.0

    return {
        "bank": bank_name,
        "action_count": len(by_action),
        "episode_count": len(by_episode),
        "best_fixed_action_id": best_fixed_action,
        "best_fixed_mean_dice": best_fixed_dice,
        "best_fixed_sam_calls": best_fixed_sam_calls,
        "best_fixed_runtime_ms": best_fixed_runtime,
        "oracle_mean_dice": oracle_mean,
        "oracle_mean_sam_calls": oracle_mean_calls,
        "oracle_gap": gap,
        "oracle_action_counts": oracle_action_counts,
        "win_tie_loss": win_tie_loss,
    }


def _gate_x4_status(
    saturations: list[dict[str, Any]],
) -> tuple[str, str]:
    """Assess Gate X4 (Action-Space Saturation).

    Per directive section 17:
      - ACTION_SPACE_NOT_SATURATED: oracle gap grows or stays large as bank grows.
      - ACTION_SPACE_SATURATED: oracle gap shrinks to near-zero at A_large.
      - OLD_ACTION_BANK_INADEQUATE: A_small oracle gap is small but A_medium/A_large
        reveal substantially higher absolute Dice (the old bank was too small).
    """
    if not saturations:
        return ("ACTION_SPACE_INSUFFICIENT_DATA", "no saturation data")
    small = next((s for s in saturations if s["bank"] == "A_small"), None)
    large = next((s for s in saturations if s["bank"] == "A_large"), None)
    medium = next((s for s in saturations if s["bank"] == "A_medium"), None)
    if small is None or large is None:
        return ("ACTION_SPACE_INSUFFICIENT_DATA", "missing A_small or A_large")
    if small["oracle_gap"] is None or large["oracle_gap"] is None:
        return ("ACTION_SPACE_INSUFFICIENT_DATA", "missing oracle gap data")

    # Check if old bank was inadequate: does A_large's best fixed substantially
    # beat A_small's best fixed?
    if small["best_fixed_mean_dice"] is not None and large["best_fixed_mean_dice"] is not None:
        fixed_gain = large["best_fixed_mean_dice"] - small["best_fixed_mean_dice"]
        if fixed_gain >= 0.05:
            return (
                "OLD_ACTION_BANK_INADEQUATE",
                f"A_large best_fixed={large['best_fixed_mean_dice']:.4f} vs "
                f"A_small best_fixed={small['best_fixed_mean_dice']:.4f}; "
                f"gain={fixed_gain:.4f} >= 0.05",
            )

    # Check if saturated: A_large oracle gap is small (< 0.005).
    if large["oracle_gap"] < 0.005:
        return (
            "ACTION_SPACE_SATURATED",
            f"A_large oracle_gap={large['oracle_gap']:.4f} < 0.005",
        )

    return (
        "ACTION_SPACE_NOT_SATURATED",
        f"A_large oracle_gap={large['oracle_gap']:.4f} >= 0.005",
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


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
                "phase": "phaseX-nested-bank-saturation",
                "command": "scripts/run_phasex_nested_bank_saturation.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Phase X: nested action bank saturation and finite-bank oracle (section 11).",
            }
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
    saturations: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase X: Nested Action Bank Saturation Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: nested_bank_saturation (Gate X4)",
        f"Episodes: {summary['episode_count']}",
        f"Positive limit: {summary['positive_limit']}",
        f"Device: {summary['device']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        f"Elapsed seconds: {summary['elapsed_seconds']}",
        f"GPU-hours: {summary['gpu_hours']}",
        "",
        "## Bank constraints (frozen at preregistration)",
        "",
        "| Bank | Max K+ | Max K- | Max calls |",
        "|---|---|---|---|",
    ]
    for bank in NESTED_BANKS:
        lines.append(
            f"| {bank.name} | {bank.max_positive} | {bank.max_negative} | {bank.max_calls} |"
        )
    lines.extend(
        [
            "",
            "## Gate X4 Status",
            "",
            f"**{summary['gate_x4_status']}**",
            "",
            summary["gate_x4_rationale"],
            "",
            "## Saturation curve",
            "",
            "| Bank | Actions | Episodes | Best fixed action | Best fixed Dice | Oracle Dice | Oracle gap | Best fixed calls | Oracle calls | Win/Tie/Loss |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for s in saturations:
        wtl = s.get("win_tie_loss", {})
        lines.append(
            f"| {s['bank']} | {s['action_count']} | {s['episode_count']} | "
            f"{s['best_fixed_action_id']} | "
            f"{_format_or_na(s['best_fixed_mean_dice'])} | "
            f"{_format_or_na(s['oracle_mean_dice'])} | "
            f"{_format_or_na(s['oracle_gap'])} | "
            f"{s.get('best_fixed_sam_calls', 'N/A')} | "
            f"{_format_or_na(s.get('oracle_mean_sam_calls'), '.1f')} | "
            f"{wtl.get('win', 0)}/{wtl.get('tie', 0)}/{wtl.get('loss', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `OLD_ACTION_BANK_INADEQUATE`: A_large best fixed substantially beats A_small best fixed (>= 0.05 Dice).",
            "- `ACTION_SPACE_SATURATED`: A_large oracle gap < 0.005.",
            "- `ACTION_SPACE_NOT_SATURATED`: A_large oracle gap >= 0.005.",
            "- `ACTION_SPACE_INSUFFICIENT_DATA`: missing bank data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nested_bank_saturation(
    *,
    project_root: Path = PROJECT_ROOT,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    device: str = "cpu",
    positive_limit: int = 100,
    banks: Sequence[BankSpec] | None = None,
) -> dict[str, Any]:
    workspace_root = project_root.parent
    candidates_csv = candidates_csv or (
        project_root / "reports" / "phaseX_candidate_pilot_100" / "candidate_pilot_candidates.csv"
    )
    episodes_csv = episodes_csv or (
        project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv"
    )
    out_dir = out_dir or (project_root / "reports" / "phaseX_nested_bank_saturation")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_CANDIDATE_POOL_MISSING: {candidates_csv}")
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_EPISODES_MISSING: {episodes_csv}")

    episodes = _read_episodes(episodes_csv)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    bank_specs = tuple(banks) if banks is not None else NESTED_BANKS
    action_specs: list[ActionSpec] = []
    for bank in bank_specs:
        action_specs.extend(_action_set_for_bank(bank))

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
        if len(positive_coords) < max(a.k_positive for a in action_specs):
            continue

        predictor.set_image(np.asarray(image))
        episode_count += 1

        for action in action_specs:
            if action.k_positive > len(positive_coords):
                continue
            record = _run_action(
                predictor,
                action,
                positive_coords,
                negative_coords,
                target,
                target_labels,
                target_component_count,
                fusion_config=fusion_config,
                episode_id=episode_id,
            )
            records.append(record)

    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0

    saturations = [_bank_saturation(records, bank.name) for bank in bank_specs]
    gate_status, gate_rationale = _gate_x4_status(saturations)

    _write_csv(out_dir / "nested_bank_query_metrics.csv", records)
    _write_csv(out_dir / "nested_bank_saturation_curve.csv", saturations)

    summary = {
        "phase": "phaseX",
        "stage": "nested_bank_saturation",
        "gate": "X4",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "episode_count": episode_count,
        "positive_limit": positive_limit,
        "banks": [bank.name for bank in bank_specs],
        "action_count": len(action_specs),
        "gate_x4_status": gate_status,
        "gate_x4_rationale": gate_rationale,
        "saturations": saturations,
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
    }
    (out_dir / "nested_bank_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_summary_md(out_dir / "nested_bank_summary.md", summary, saturations)
    _append_ledger(project_root, gpu_hours)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase X: nested action bank saturation (section 11)."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=100)
    args = parser.parse_args()
    summary = run_nested_bank_saturation(
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
