"""Phase X Stage S2: Single-Variable K+ Scaling.

Implements section 8 Stage S1 of ``AdaPromptSet_ActionSpace_Oracle_Retry.md``.
Fixed: selection=Top-K, negative=none, grouping=one group, calls=1.
Only K+ varies across the frozen grid.

This is the first stage that runs SAM inference. For each K+ and each
validation episode, takes the top K+ positives from the candidate pool,
prompts SAM-B once, and records Dice/IoU/AJI/component recall.

Outputs (section 8 Stage S1 layout)::

    reports/phaseX_single_variable_kpos/
        single_variable_kpos_query_metrics.csv
        single_variable_kpos_curve.csv
        single_variable_kpos_summary.md
        single_variable_kpos_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
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
from adapromptset.matching.runtime import checkpoint_record
from adapromptset.scripts.build_action_bank import _candidate_pools_from_csv
from adapromptset.scripts.build_phasex_preregistration import POSITIVE_K_GRID
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518


def _read_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _component_labels(mask: Sequence[Sequence[bool]]) -> dict[tuple[int, int], int]:
    """BFS connected components. Returns {foreground_pixel: label}."""
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


def _component_recall(
    prediction: Sequence[Sequence[bool]],
    target_labels: dict[tuple[int, int], int],
) -> tuple[float, int, int]:
    """Return (recall, hit_count, total_count).

    A GT component is "hit" if at least one of its foreground pixels is
    also predicted as foreground.
    """
    all_components = set(target_labels.values())
    total = len(all_components)
    if total == 0:
        return 0.0, 0, 0
    hit: set[int] = set()
    for (row, col), label in target_labels.items():
        if prediction[row][col]:
            hit.add(label)
    return len(hit) / total, len(hit), total


def _semantic_instance_labels(mask: Sequence[Sequence[bool]]) -> list[list[int]]:
    return [[1 if bool(value) else 0 for value in row] for row in mask]


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
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "phaseX-stage-S2-single-variable-kpos",
                "command": "scripts/run_phasex_single_variable_kpos.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Phase X Stage S2: single-variable K+ scaling with SAM-B on MoNuSAC validation.",
            }
        )


def _gate_x3_status(curve: list[dict[str, Any]]) -> tuple[str, str]:
    """Assess Gate X3 (Dense Prompt Viability) from the K+ scaling curve."""
    k5 = next((c for c in curve if c["k_positive"] == 5), None)
    k100 = next((c for c in curve if c["k_positive"] == 100), None)

    if k5 is None or k100 is None:
        return "DENSE_PROMPT_INSUFFICIENT_DATA", "missing K=5 or K=100 data points"
    if k5["mean_dice"] is None or k100["mean_dice"] is None:
        return "DENSE_PROMPT_INSUFFICIENT_DATA", "missing Dice values at K=5 or K=100"

    dice_gain = k100["mean_dice"] - k5["mean_dice"]
    recall_5 = k5["mean_component_recall"] or 0.0
    recall_100 = k100["mean_component_recall"] or 0.0
    recall_gain = recall_100 - recall_5

    if k100["mean_dice"] < 0.3:
        return (
            "DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW",
            f"K=100 mean Dice = {k100['mean_dice']:.3f} < 0.3; absolute performance too low for dense prompt viability",
        )

    if dice_gain < 0.01 and recall_gain < 0.05:
        return (
            "DENSE_PROMPT_NO_GAIN",
            f"K=5 -> K=100 Dice gain = {dice_gain:.3f} < 0.01, component recall gain = {recall_gain:.3f} < 0.05; no significant improvement from dense prompts",
        )

    if dice_gain < 0.01 and recall_gain >= 0.05:
        return (
            "DENSE_PROMPT_COST_ONLY_GAIN",
            f"K=5 -> K=100 Dice gain = {dice_gain:.3f} < 0.01 but component recall gain = {recall_gain:.3f} >= 0.05; dense prompts improve recall but not Dice",
        )

    return (
        "DENSE_PROMPT_VIABILITY_ESTABLISHED",
        f"K=5 -> K=100 Dice gain = {dice_gain:.3f}, component recall gain = {recall_gain:.3f}; dense prompts improve both Dice and recall",
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
        "# Phase X Stage S2: Single-Variable K+ Scaling Summary",
        "",
        f"Date: {summary['date']}",
        f"Stage: single_variable_kpos (Gate X3)",
        f"Episodes: {summary['episode_count']}",
        f"Positive limit: {summary['positive_limit']}",
        f"Device: {summary['device']}",
        f"SAM inference run: {summary['sam_inference_run']}",
        f"Elapsed seconds: {summary['elapsed_seconds']}",
        f"GPU-hours: {summary['gpu_hours']}",
        "",
        "## Gate X3 Status",
        "",
        f"**{summary['gate_x3_status']}**",
        "",
        summary["gate_x3_rationale"],
        "",
        "## K+ scaling curve",
        "",
        "| K+ | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean runtime (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in curve:
        lines.append(
            f"| {c['k_positive']} | {c['episode_count']} | "
            f"{_format_or_na(c['mean_dice'])} | "
            f"{_format_or_na(c['mean_iou'])} | "
            f"{_format_or_na(c['mean_aji'])} | "
            f"{_format_or_na(c['mean_component_recall'])} | "
            f"{_format_or_na(c['mean_runtime_ms'], '.1f')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `DENSE_PROMPT_VIABILITY_ESTABLISHED`: dense prompts improve both Dice and component recall.",
            "- `DENSE_PROMPT_NO_GAIN`: K+ scaling does not improve Dice or component recall.",
            "- `DENSE_PROMPT_COST_ONLY_GAIN`: dense prompts improve recall but not Dice (cost-only gain).",
            "- `DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW`: even at K=100, absolute Dice is too low.",
            "- `DENSE_PROMPT_INSUFFICIENT_DATA`: missing K=5 or K=100 data points.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_variable_kpos(
    *,
    project_root: Path = PROJECT_ROOT,
    candidates_csv: Path | None = None,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    device: str = "cpu",
    positive_limit: int = 100,
) -> dict[str, Any]:
    workspace_root = project_root.parent
    candidates_csv = candidates_csv or (
        project_root / "reports" / "phaseX_candidate_pilot_100" / "candidate_pilot_candidates.csv"
    )
    episodes_csv = episodes_csv or (
        project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv"
    )
    out_dir = out_dir or (project_root / "reports" / "phaseX_single_variable_kpos")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_CANDIDATE_POOL_MISSING: {candidates_csv}")
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"PHASEX_BLOCKED_EPISODES_MISSING: {episodes_csv}")

    episodes = _read_episodes(episodes_csv)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

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

        # Get top `positive_limit` positives from candidate pool
        try:
            pools = _candidate_pools_from_csv(
                candidates_csv,
                episode_id=episode_id,
                positive_limit=positive_limit,
            )
        except ValueError:
            continue  # skip episode if no candidates

        positive_coords = [c.coordinate for c in pools.positives]

        # Set image once per episode (expensive on CPU)
        predictor.set_image(np.asarray(image))
        episode_count += 1

        for k in POSITIVE_K_GRID:
            if k > len(positive_coords):
                records.append(
                    {
                        "episode_id": episode_id,
                        "k_positive": k,
                        "actual_positive_count": len(positive_coords),
                        "shortfall": k - len(positive_coords),
                        "dice": None,
                        "iou": None,
                        "aji": None,
                        "component_recall": None,
                        "target_component_count": target_component_count,
                        "hit_components": None,
                        "sam_calls": 0,
                        "runtime_ms": 0.0,
                        "sam_score": None,
                        "status": "SKIPPED_INSUFFICIENT_POSITIVES",
                    }
                )
                continue

            points = positive_coords[:k]
            labels = [1] * k

            t0 = time.perf_counter()
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best].astype(bool).tolist()
            score = float(scores[best])
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            dice = dice_score(mask, target)
            iou = iou_score(mask, target)
            aji = aggregated_jaccard_index(
                _semantic_instance_labels(mask),
                _semantic_instance_labels(target),
            )
            recall, hit_count, total_count = _component_recall(mask, target_labels)

            records.append(
                {
                    "episode_id": episode_id,
                    "k_positive": k,
                    "actual_positive_count": k,
                    "shortfall": 0,
                    "dice": dice,
                    "iou": iou,
                    "aji": aji,
                    "component_recall": recall,
                    "target_component_count": total_count,
                    "hit_components": hit_count,
                    "sam_calls": 1,
                    "runtime_ms": round(runtime_ms, 3),
                    "sam_score": score,
                    "status": "OK",
                }
            )

    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0

    # Aggregate by K+
    curve: list[dict[str, Any]] = []
    for k in POSITIVE_K_GRID:
        k_records = [r for r in records if r["k_positive"] == k and r["status"] == "OK"]
        if not k_records:
            curve.append(
                {
                    "k_positive": k,
                    "episode_count": 0,
                    "mean_dice": None,
                    "mean_iou": None,
                    "mean_aji": None,
                    "mean_component_recall": None,
                    "mean_runtime_ms": None,
                }
            )
            continue
        dices = [r["dice"] for r in k_records]
        ious = [r["iou"] for r in k_records]
        ajis = [r["aji"] for r in k_records]
        recalls = [r["component_recall"] for r in k_records]
        runtimes = [r["runtime_ms"] for r in k_records]
        curve.append(
            {
                "k_positive": k,
                "episode_count": len(k_records),
                "mean_dice": sum(dices) / len(dices),
                "mean_iou": sum(ious) / len(ious),
                "mean_aji": sum(ajis) / len(ajis),
                "mean_component_recall": sum(recalls) / len(recalls),
                "mean_runtime_ms": sum(runtimes) / len(runtimes),
            }
        )

    gate_x3_status, gate_x3_rationale = _gate_x3_status(curve)

    _write_csv(out_dir / "single_variable_kpos_query_metrics.csv", records)
    _write_csv(out_dir / "single_variable_kpos_curve.csv", curve)

    summary = {
        "phase": "phaseX",
        "stage": "single_variable_kpos",
        "gate": "X3",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "episode_count": episode_count,
        "positive_limit": positive_limit,
        "positive_k_grid": list(POSITIVE_K_GRID),
        "gate_x3_status": gate_x3_status,
        "gate_x3_rationale": gate_x3_rationale,
        "dice_curve": {str(c["k_positive"]): c["mean_dice"] for c in curve},
        "component_recall_curve": {
            str(c["k_positive"]): c["mean_component_recall"] for c in curve
        },
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours": round(gpu_hours, 6),
        "device": device,
        "sam_inference_run": True,
        "sam_checkpoint": segmenter_record,
        "candidates_csv": str(candidates_csv),
    }
    (out_dir / "single_variable_kpos_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_summary_md(out_dir / "single_variable_kpos_summary.md", summary, curve)
    _append_ledger(project_root, gpu_hours)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase X Stage S2: single-variable K+ scaling with SAM."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=100)
    args = parser.parse_args()
    summary = run_single_variable_kpos(
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
