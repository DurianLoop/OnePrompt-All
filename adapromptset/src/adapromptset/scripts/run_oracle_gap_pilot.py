from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from adapromptset.actions.bank import candidate_coordinates
from adapromptset.fusion.masks import load_fusion_config
from adapromptset.scripts.build_action_bank import _candidate_pools_from_csv
from adapromptset.scripts.run_action_utility_smoke import (
    _read_episode,
    _write_csv,
    load_actions_from_manifest,
    select_valid_actions,
)
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor
from adapromptset.utility.action_execution import execute_action


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_pilot_paths(project_root: Path, *, dataset_slug: str) -> dict[str, Path]:
    out_suffix = "" if dataset_slug == "monusac" else f"_{dataset_slug}"
    return {
        "episodes_csv": project_root / "manifests" / "episodes" / dataset_slug / "episodes_val.csv",
        "candidates_csv": project_root / "reports" / f"candidate_pilot{out_suffix}" / "candidate_pilot_candidates.csv",
        "out_dir": project_root / "reports" / f"oracle_gap_pilot{out_suffix}",
        "gate_status": project_root / "reports" / f"gate_f_status{out_suffix}.md",
    }


def summarize_oracle_gap(
    records: list[dict[str, Any]],
    *,
    min_episodes: int = 10,
    min_dice_gap: float = 0.010,
    max_top_oracle_fraction: float = 0.70,
) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_episode[str(record["episode_id"])].append(record)
        by_action[str(record["action_id"])].append(record)
    if not by_episode:
        return {"gate_f_status": "FAIL_NO_VALID_RECORDS", "episode_count": 0}

    def metric_available(metric: str) -> bool:
        return all(row.get(metric) not in {None, ""} for row in valid)

    def metric_stats(metric: str, source_by_episode: dict[str, list[dict[str, Any]]], source_by_action: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        if not metric_available(metric):
            return {
                f"{metric}_available": False,
                f"best_fixed_{metric}_action_id": None,
                f"mean_oracle_{metric}": None,
                f"mean_best_fixed_{metric}": None,
                f"{metric}_oracle_gap": None,
            }
        metric_oracles = {
            episode_id: max(items, key=lambda row: float(row[metric]))
            for episode_id, items in source_by_episode.items()
        }
        action_mean = {
            action_id: sum(float(row[metric]) for row in items) / len(items)
            for action_id, items in source_by_action.items()
        }
        best_action = max(action_mean, key=action_mean.get)
        best_by_episode = {
            episode_id: next(row for row in items if row["action_id"] == best_action)
            for episode_id, items in source_by_episode.items()
            if any(row["action_id"] == best_action for row in items)
        }
        mean_oracle = sum(float(row[metric]) for row in metric_oracles.values()) / len(metric_oracles)
        mean_best = sum(float(best_by_episode[episode_id][metric]) for episode_id in metric_oracles if episode_id in best_by_episode) / len(best_by_episode)
        return {
            f"{metric}_available": True,
            f"best_fixed_{metric}_action_id": best_action,
            f"mean_oracle_{metric}": mean_oracle,
            f"mean_best_fixed_{metric}": mean_best,
            f"{metric}_oracle_gap": mean_oracle - mean_best,
        }

    episode_oracles: dict[str, dict[str, Any]] = {
        episode_id: max(items, key=lambda row: float(row["dice"]))
        for episode_id, items in by_episode.items()
    }
    action_mean_dice = {
        action_id: sum(float(row["dice"]) for row in items) / len(items)
        for action_id, items in by_action.items()
    }
    best_fixed_action_id = max(action_mean_dice, key=action_mean_dice.get)
    best_fixed_by_episode = {
        episode_id: next(row for row in items if row["action_id"] == best_fixed_action_id)
        for episode_id, items in by_episode.items()
        if any(row["action_id"] == best_fixed_action_id for row in items)
    }
    oracle_dices = [float(row["dice"]) for row in episode_oracles.values()]
    best_fixed_dices = [float(best_fixed_by_episode[episode_id]["dice"]) for episode_id in episode_oracles if episode_id in best_fixed_by_episode]
    mean_oracle_dice = sum(oracle_dices) / len(oracle_dices)
    mean_best_fixed_dice = sum(best_fixed_dices) / len(best_fixed_dices)
    oracle_counts = Counter(str(row["action_id"]) for row in episode_oracles.values())
    oracle_family_counts = Counter(str(row["action_family"]) for row in episode_oracles.values())
    top_fraction = max(oracle_counts.values()) / len(episode_oracles)
    dice_gap = mean_oracle_dice - mean_best_fixed_dice
    action_calls = {
        action_id: max(int(row["sam_calls"]) for row in items)
        for action_id, items in by_action.items()
    }
    best_fixed_sam_calls = action_calls[best_fixed_action_id]
    budget_gaps: dict[str, dict[str, Any]] = {}
    for budget in sorted(set(action_calls.values())):
        budget_records = [row for row in valid if int(row["sam_calls"]) <= budget]
        budget_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        budget_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in budget_records:
            budget_by_episode[str(row["episode_id"])].append(row)
            budget_by_action[str(row["action_id"])].append(row)
        if len(budget_by_episode) != len(by_episode):
            continue
        budget_oracle = {
            episode_id: max(items, key=lambda row: float(row["dice"]))
            for episode_id, items in budget_by_episode.items()
        }
        budget_action_mean = {
            action_id: sum(float(row["dice"]) for row in items) / len(items)
            for action_id, items in budget_by_action.items()
        }
        budget_best_action = max(budget_action_mean, key=budget_action_mean.get)
        budget_best = {
            episode_id: next(row for row in items if row["action_id"] == budget_best_action)
            for episode_id, items in budget_by_episode.items()
        }
        budget_mean_oracle = sum(float(row["dice"]) for row in budget_oracle.values()) / len(budget_oracle)
        budget_mean_best = sum(float(row["dice"]) for row in budget_best.values()) / len(budget_best)
        budget_gaps[str(budget)] = {
            "best_fixed_action_id": budget_best_action,
            "mean_oracle_dice": budget_mean_oracle,
            "mean_best_fixed_dice": budget_mean_best,
            "dice_oracle_gap": budget_mean_oracle - budget_mean_best,
        }
        budget_gaps[str(budget)].update(metric_stats("aji", budget_by_episode, budget_by_action))
    if len(by_episode) < min_episodes:
        status = "PENDING_INSUFFICIENT_EPISODES"
    elif top_fraction > max_top_oracle_fraction:
        status = "FAIL_SINGLE_ACTION_DOMINATES"
    elif dice_gap < min_dice_gap:
        status = "FAIL_ORACLE_GAP_SMALL"
    else:
        status = "PASS_SUBSET_ORACLE_GAP"
    summary = {
        "gate_f_status": status,
        "episode_count": len(by_episode),
        "record_count": len(records),
        "valid_record_count": len(valid),
        "best_fixed_action_id": best_fixed_action_id,
        "mean_oracle_dice": mean_oracle_dice,
        "mean_best_fixed_dice": mean_best_fixed_dice,
        "dice_oracle_gap": dice_gap,
        "oracle_action_counts": dict(oracle_counts),
        "oracle_family_counts": dict(oracle_family_counts),
        "top_oracle_action_fraction": top_fraction,
        "best_fixed_sam_calls": best_fixed_sam_calls,
        "mean_oracle_sam_calls": sum(int(row["sam_calls"]) for row in episode_oracles.values()) / len(episode_oracles),
        "budget_gaps": budget_gaps,
        "gate_f_min_dice_gap": min_dice_gap,
        "gate_f_max_top_oracle_fraction": max_top_oracle_fraction,
        "max_sam_calls": max(int(row["sam_calls"]) for row in valid),
        "decoder_calls": sum(int(row["decoder_calls"]) for row in valid),
        "cache_hits": sum(int(row.get("cache_hits", 0)) for row in valid),
    }
    summary.update(metric_stats("aji", by_episode, by_action))
    return summary


def _read_episode_ids(path: Path, max_episodes: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["episode_id"] for row in list(csv.DictReader(handle))[:max_episodes]]


def _append_ledger(project_root: Path, gpu_hours: float, *, dataset_slug: str) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "6-oracle-gap-pilot",
                "command": "scripts/run_oracle_gap_pilot.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": f"{dataset_slug} validation subset action utility and oracle-gap pilot.",
            }
        )


def _gate_f_policy_decision(summary: dict[str, Any]) -> tuple[str, str]:
    status = str(summary["gate_f_status"])
    action_bank = Path(str(summary.get("action_bank", ""))).name
    if status.startswith("PASS"):
        return "POLICY_TRAINING_ALLOWED", "oracle gap passes the subset gate; policy may start after the remaining lock checks pass"
    if status == "PENDING_INSUFFICIENT_EPISODES":
        return "COLLECT_MORE_EPISODES", "not enough validation episodes for Gate F; keep policy paused and extend the pilot"
    if status in {"FAIL_ORACLE_GAP_SMALL", "FAIL_SINGLE_ACTION_DOMINATES"} and action_bank.startswith("prompt_set_bank_v2"):
        return "PIVOT_F_REQUIRED", "rescue action bank already failed; do not start policy, pivot to fixed-rule/action-oracle analysis or a predeclared new-domain pilot"
    return "RESCUE_F1_REQUIRED", "oracle gap is below threshold; do not start policy, expand the predeclared action bank once on train/validation"


def _write_gate_f_status(project_root: Path, summary: dict[str, Any], *, status_path: Path | None = None) -> None:
    decision, decision_note = _gate_f_policy_decision(summary)
    lines = [
        "# Gate F Pilot Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        f"| action utility pilot | {summary['gate_f_status']} | episodes={summary['episode_count']} records={summary['valid_record_count']} |",
        f"| oracle vs best fixed | {summary['gate_f_status']} | oracle={summary['mean_oracle_dice']:.4f} best_fixed={summary['mean_best_fixed_dice']:.4f} gap={summary['dice_oracle_gap']:.4f} |",
        f"| AJI oracle vs best fixed | {'AVAILABLE' if summary.get('aji_available') else 'UNAVAILABLE'} | oracle={summary.get('mean_oracle_aji')} best_fixed={summary.get('mean_best_fixed_aji')} gap={summary.get('aji_oracle_gap')} |",
        f"| oracle action concentration | {summary['gate_f_status']} | top_fraction={summary['top_oracle_action_fraction']:.3f}; counts={summary['oracle_action_counts']} |",
        f"| policy decision | {decision} | {decision_note} |",
    ]
    (status_path or (project_root / "reports" / "gate_f_status.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oracle_gap_pilot(
    *,
    project_root: Path = PROJECT_ROOT,
    max_episodes: int = 10,
    positive_limit: int = 5,
    max_actions: int | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    action_bank: Path | None = None,
    dataset_slug: str = "monusac",
    episodes_csv: Path | None = None,
    candidates_csv: Path | None = None,
    gate_min_episodes: int = 10,
) -> int:
    pilot_paths = default_pilot_paths(project_root, dataset_slug=dataset_slug)
    out_dir = out_dir or pilot_paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = project_root.parent
    episodes_csv = episodes_csv or pilot_paths["episodes_csv"]
    candidates_csv = candidates_csv or pilot_paths["candidates_csv"]
    episode_ids = _read_episode_ids(episodes_csv, max_episodes)
    actions = select_valid_actions(
        load_actions_from_manifest(action_bank or (project_root / "configs" / "actions" / "prompt_set_bank_v1.json")),
        max_actions=max_actions,
    )
    fusion_config = load_fusion_config(project_root / "configs" / "segmenter" / "fusion_mvp.json")
    predictor = load_sam_predictor(workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth", workspace_root, device)
    start = time.perf_counter()
    all_records: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        episode = _read_episode(episodes_csv, episode_id)
        episode_paths = resolve_episode_paths(episode, workspace_root)
        image = load_rgb(episode_paths["query_image"])
        target = load_mask(episode_paths["query_mask"])
        pools = _candidate_pools_from_csv(
            candidates_csv,
            episode_id=episode_id,
            positive_limit=positive_limit,
        )
        coordinates = candidate_coordinates(pools)
        predictor.set_image(np.asarray(image))

        def segmenter(points: list[tuple[int, int]], labels: list[int]) -> tuple[list[list[bool]], float]:
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            return masks[best].astype(bool).tolist(), float(scores[best])

        cache: dict[str, tuple[list[list[bool]], float]] = {}
        for action in actions:
            all_records.append(
                execute_action(
                    episode_id=episode_id,
                    action=action,
                    coordinates=coordinates,
                    target_mask=target,
                    segmenter=segmenter,
                    fusion_config=fusion_config,
                    cache=cache,
                )
            )
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0
    summary = summarize_oracle_gap(all_records, min_episodes=gate_min_episodes)
    summary.update(
        {
            "status": summary["gate_f_status"],
            "max_episodes": max_episodes,
            "gate_min_episodes": gate_min_episodes,
            "dataset_slug": dataset_slug,
            "positive_limit": positive_limit,
            "max_actions": max_actions,
            "action_bank": str(action_bank or (project_root / "configs" / "actions" / "prompt_set_bank_v1.json")),
            "episodes_csv": str(episodes_csv),
            "candidates_csv": str(candidates_csv),
            "elapsed_seconds": round(elapsed, 3),
            "gpu_hours_estimate": round(gpu_hours, 6),
            "utility_csv": str(out_dir / "action_utility_pilot.csv"),
        }
    )
    _write_csv(out_dir / "action_utility_pilot.csv", all_records)
    (out_dir / "oracle_gap_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_gate_f_status(project_root, summary, status_path=pilot_paths["gate_status"])
    _append_ledger(project_root, gpu_hours, dataset_slug=dataset_slug)
    return 0 if summary["gate_f_status"].startswith("PASS") else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a validation-only oracle-gap pilot over MVP actions.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--positive-limit", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--action-bank", type=Path, default=None)
    parser.add_argument("--dataset-slug", default="monusac")
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--gate-min-episodes", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(
        run_oracle_gap_pilot(
            project_root=args.project_root,
            max_episodes=args.max_episodes,
            positive_limit=args.positive_limit,
            max_actions=args.max_actions,
            device=args.device,
            out_dir=args.out_dir,
            action_bank=args.action_bank,
            dataset_slug=args.dataset_slug,
            episodes_csv=args.episodes_csv,
            candidates_csv=args.candidates_csv,
            gate_min_episodes=args.gate_min_episodes,
        )
    )


if __name__ == "__main__":
    main()
