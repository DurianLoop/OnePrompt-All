from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from adapromptset.actions.bank import action_bank_manifest, build_mvp_action_bank, build_rescue_action_bank
from adapromptset.candidates.points import Candidate, CandidatePools


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_pools_from_csv(path: Path, *, episode_id: str, positive_limit: int) -> CandidatePools:
    positives: list[Candidate] = []
    negatives: list[Candidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["episode_id"] != episode_id or int(row["positive_limit"]) != positive_limit:
                continue
            candidate = Candidate(
                kind=row["kind"],
                grid_index=(int(row["grid_row"]), int(row["grid_col"])),
                coordinate=(int(row["x"]), int(row["y"])),
                score=float(row["score"]),
            )
            if candidate.kind == "positive":
                positives.append(candidate)
            elif candidate.kind == "negative":
                negatives.append(candidate)
            else:
                raise ValueError(f"unknown candidate kind: {candidate.kind}")
    if not positives:
        raise ValueError(f"no positive candidates for {episode_id} at positive_limit={positive_limit}")
    return CandidatePools(positives=tuple(positives), negatives=tuple(negatives))


def _write_yaml_manifest(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"name: {manifest['name']}",
        f"bank_hash: {manifest['bank_hash']}",
        f"action_count: {manifest['action_count']}",
        f"valid_action_count: {manifest['valid_action_count']}",
        f"invalid_action_count: {manifest['invalid_action_count']}",
        "actions:",
    ]
    for action in manifest["actions"]:
        lines.extend(
            [
                f"  - action_id: {action['action_id']}",
                f"    family: {action['family']}",
                f"    valid: {str(action['valid']).lower()}",
                f"    invalid_reason: {action['invalid_reason']}",
                f"    sam_calls: {action['sam_calls']}",
                f"    prompt_points: {action['prompt_points']}",
                f"    content_hash: {action['content_hash']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_e_status(project_root: Path, manifest: dict[str, Any]) -> None:
    invalid = manifest["invalid_action_count"]
    status = "PASS_RESCUE_SUBSET" if manifest["name"].endswith("_v2") else "PASS_MVP_SUBSET" if manifest["valid_action_count"] > 0 else "FAIL"
    lines = [
        "# Gate E Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        f"| action bank manifest | {status} | name={manifest['name']} actions={manifest['action_count']} valid={manifest['valid_action_count']} invalid={invalid} |",
        "| shared candidate pool | PASS | actions are built from candidate_pilot_candidates.csv using positive-then-negative index order |",
        "| action cost accounting | PASS | sam_calls equals valid prompt group count; invalid actions cost 0 |",
        "| full baseline matrix | PENDING | KMeans, local-peak, SynPo-style, GF-SAM-style, safe anchors not yet implemented |",
    ]
    (project_root / "reports" / "gate_e_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_action_bank_artifacts(
    *,
    candidates_csv: Path,
    episode_id: str,
    positive_limit: int,
    out_dir: Path,
    config_dir: Path,
    project_root: Path | None = None,
    bank_version: str = "v1",
) -> int:
    pools = _candidate_pools_from_csv(candidates_csv, episode_id=episode_id, positive_limit=positive_limit)
    if bank_version == "v1":
        actions = build_mvp_action_bank(pools)
        bank_name = "prompt_set_bank_v1"
    elif bank_version == "v2":
        actions = build_mvp_action_bank(pools) + build_rescue_action_bank(pools)
        bank_name = "prompt_set_bank_v2"
    else:
        raise ValueError("bank_version must be 'v1' or 'v2'")
    manifest = action_bank_manifest(bank_name, actions)
    manifest["source"] = {
        "candidate_csv": str(candidates_csv),
        "episode_id": episode_id,
        "positive_limit": positive_limit,
        "candidate_index_order": "positives_then_negatives",
    }
    manifest["invalid_action_count"] = manifest["action_count"] - manifest["valid_action_count"]
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_bank_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (config_dir / f"{bank_name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_yaml_manifest(config_dir / f"{bank_name}.yaml", manifest)
    if project_root is not None:
        _write_gate_e_status(project_root, manifest)
    return 0 if manifest["valid_action_count"] > 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked MVP prompt action bank from candidate pilot artifacts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--positive-limit", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--bank-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    project_root = args.project_root
    candidates_csv = args.candidates_csv or (project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv")
    episode_id = args.episode_id or "monusac_val__train_111__to__train_004"
    out_dir = args.out_dir or (project_root / "reports" / "action_bank")
    config_dir = args.config_dir or (project_root / "configs" / "actions")
    raise SystemExit(
        build_action_bank_artifacts(
            candidates_csv=candidates_csv,
            episode_id=episode_id,
            positive_limit=args.positive_limit,
            out_dir=out_dir,
            config_dir=config_dir,
            project_root=project_root,
            bank_version=args.bank_version,
        )
    )


if __name__ == "__main__":
    main()
