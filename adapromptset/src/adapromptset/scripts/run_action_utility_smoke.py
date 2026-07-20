from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapromptset.actions.bank import PromptAction, PromptGroup, candidate_coordinates
from adapromptset.candidates.points import Candidate, CandidatePools
from adapromptset.fusion.masks import load_fusion_config
from adapromptset.scripts.build_action_bank import _candidate_pools_from_csv
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths
from adapromptset.scripts.run_sam_smoke import load_mask, load_rgb, load_sam_predictor
from adapromptset.utility.action_execution import execute_action


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_actions_from_manifest(path: Path) -> tuple[PromptAction, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions: list[PromptAction] = []
    for row in payload["actions"]:
        actions.append(
            PromptAction(
                action_id=row["action_id"],
                family=row["family"],
                groups=tuple(
                    PromptGroup(
                        positive_indices=tuple(group["positive_indices"]),
                        negative_indices=tuple(group["negative_indices"]),
                    )
                    for group in row["groups"]
                ),
                config=row["config"],
                valid=bool(row["valid"]),
                invalid_reason=row["invalid_reason"],
            )
        )
    return tuple(actions)


def select_valid_actions(actions: tuple[PromptAction, ...], *, max_actions: int | None = None) -> tuple[PromptAction, ...]:
    selected = tuple(action for action in actions if action.valid)
    if max_actions is not None:
        return selected[:max_actions]
    return selected


def summarize_utility_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    if not valid:
        return {
            "status": "FAIL",
            "valid_action_count": 0,
            "oracle_action_id": None,
            "best_fixed_action_id": None,
            "oracle_dice": None,
            "best_fixed_dice": None,
            "max_sam_calls": 0,
            "decoder_calls": 0,
            "cache_hits": 0,
        }
    oracle = max(valid, key=lambda record: float(record["dice"]))
    return {
        "status": "PASS",
        "valid_action_count": len(valid),
        "oracle_action_id": oracle["action_id"],
        "best_fixed_action_id": oracle["action_id"],
        "oracle_dice": oracle["dice"],
        "best_fixed_dice": oracle["dice"],
        "max_sam_calls": max(int(record["sam_calls"]) for record in valid),
        "decoder_calls": sum(int(record["decoder_calls"]) for record in valid),
        "cache_hits": sum(int(record.get("cache_hits", 0)) for record in valid),
    }


def _read_episode(path: Path, episode_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["episode_id"] == episode_id:
                return row
    raise ValueError(f"episode not found: {episode_id}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_ledger(project_root: Path, gpu_hours: float) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "6-action-utility-smoke",
                "command": "scripts/run_action_utility_smoke.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Validation-only SAM-B action utility smoke on MVP subset actions.",
            }
        )


def _write_gate_f_smoke(project_root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Gate F Pilot Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        f"| action utility smoke | {summary['status']} | valid_actions={summary['valid_action_count']} decoder_calls={summary['decoder_calls']} cache_hits={summary.get('cache_hits', 0)} |",
        "| oracle-gap pilot | PENDING | smoke is one episode/subset only; 10-20 query pilot not yet run |",
    ]
    (project_root / "reports" / "gate_f_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_action_utility_smoke(
    *,
    project_root: Path = PROJECT_ROOT,
    episode_id: str = "monusac_val__train_111__to__train_004",
    positive_limit: int = 5,
    max_actions: int | None = 4,
    device: str = "cuda:0",
    out_dir: Path | None = None,
) -> int:
    out_dir = out_dir or (project_root / "reports" / "action_utility_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = project_root.parent
    episode = _read_episode(project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv", episode_id)
    paths = resolve_episode_paths(episode, workspace_root)
    image = load_rgb(paths["query_image"])
    target = load_mask(paths["query_mask"])
    pools = _candidate_pools_from_csv(
        project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv",
        episode_id=episode_id,
        positive_limit=positive_limit,
    )
    coordinates = candidate_coordinates(pools)
    actions = select_valid_actions(
        load_actions_from_manifest(project_root / "configs" / "actions" / "prompt_set_bank_v1.json"),
        max_actions=max_actions,
    )
    fusion_config = load_fusion_config(project_root / "configs" / "segmenter" / "fusion_mvp.json")
    predictor = load_sam_predictor(workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth", workspace_root, device)
    predictor.set_image(np.asarray(image))

    def segmenter(points: list[tuple[int, int]], labels: list[int]) -> tuple[list[list[bool]], float]:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool).tolist(), float(scores[best])

    start = time.perf_counter()
    cache: dict[str, tuple[list[list[bool]], float]] = {}
    records = [
        execute_action(
            episode_id=episode_id,
            action=action,
            coordinates=coordinates,
            target_mask=target,
            segmenter=segmenter,
            fusion_config=fusion_config,
            cache=cache,
        )
        for action in actions
    ]
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0
    summary = summarize_utility_records(records)
    summary.update(
        {
            "episode_id": episode_id,
            "positive_limit": positive_limit,
            "max_actions": max_actions,
            "elapsed_seconds": round(elapsed, 3),
            "gpu_hours_estimate": round(gpu_hours, 6),
            "utility_csv": str(out_dir / "action_utility_smoke.csv"),
        }
    )
    _write_csv(out_dir / "action_utility_smoke.csv", records)
    (out_dir / "action_utility_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_gate_f_smoke(project_root, summary)
    _append_ledger(project_root, gpu_hours)
    return 0 if summary["status"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small validation-only action utility smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--episode-id", default="monusac_val__train_111__to__train_004")
    parser.add_argument("--positive-limit", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(
        run_action_utility_smoke(
            project_root=args.project_root,
            episode_id=args.episode_id,
            positive_limit=args.positive_limit,
            max_actions=args.max_actions,
            device=args.device,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
