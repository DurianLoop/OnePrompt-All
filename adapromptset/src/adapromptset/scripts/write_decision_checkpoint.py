from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _top_action_fraction(summary: dict[str, Any]) -> float:
    counts = summary.get("oracle_action_counts", {})
    total = sum(int(value) for value in counts.values())
    return max((int(value) for value in counts.values()), default=0) / total if total else 0.0


def build_decision_checkpoint(v1_summary: dict[str, Any], v2_summary: dict[str, Any]) -> dict[str, Any]:
    v2_status = str(v2_summary["gate_f_status"])
    top_fraction = _top_action_fraction(v2_summary)
    if v2_status.startswith("PASS"):
        phase5_status = "CONTINUE_POLICY"
        rationale = "Gate F passed after the frozen action-bank evaluation."
        allowed_next_actions = ["utility_label_reliability_gate", "policy_pilot"]
        prohibited_next_actions = ["test_set_action_redesign"]
    elif v2_status == "FAIL_SINGLE_ACTION_DOMINATES" or top_fraction >= 0.70:
        phase5_status = "PIVOT_DETERMINISTIC"
        rationale = "one fixed action dominates oracle selections after the one allowed redesign"
        allowed_next_actions = ["deterministic_method_analysis", "fixed_rule_validation"]
        prohibited_next_actions = ["policy_training_on_monusac", "posthoc_monusac_action_redesign"]
    elif v2_status == "FAIL_ORACLE_GAP_SMALL":
        phase5_status = "PIVOT_ORACLE_ANALYSIS"
        rationale = "no meaningful oracle gap after the one allowed redesign; analyze fixed-rule near-oracle behavior and action-bank limits"
        allowed_next_actions = ["fixed_rule_action_oracle_analysis", "predeclared_new_domain_pilot"]
        prohibited_next_actions = ["policy_training_on_monusac", "posthoc_monusac_action_redesign"]
    elif v1_summary.get("gate_f_status") == "FAIL_ORACLE_GAP_SMALL":
        phase5_status = "REDESIGN_ACTION_BANK_ONCE"
        rationale = "initial oracle gap failed and the predeclared rescue action bank has not been evaluated"
        allowed_next_actions = ["rescue_f1_action_bank"]
        prohibited_next_actions = ["policy_training_on_monusac"]
    else:
        phase5_status = "STOP_DATASET"
        rationale = f"Gate F status {v2_status} does not allow policy training"
        allowed_next_actions = ["write_failure_report"]
        prohibited_next_actions = ["policy_training_on_monusac", "posthoc_monusac_action_redesign"]
    return {
        "phase5_status": phase5_status,
        "rationale": rationale,
        "v1_gate_f_status": v1_summary.get("gate_f_status"),
        "v1_dice_oracle_gap": v1_summary.get("dice_oracle_gap"),
        "v2_gate_f_status": v2_summary.get("gate_f_status"),
        "v2_dice_oracle_gap": v2_summary.get("dice_oracle_gap"),
        "v2_gate_f_min_dice_gap": v2_summary.get("gate_f_min_dice_gap", 0.010),
        "v2_best_fixed_action_id": v2_summary.get("best_fixed_action_id"),
        "v2_oracle_action_counts": v2_summary.get("oracle_action_counts", {}),
        "v2_top_oracle_action_fraction": top_fraction,
        "allowed_next_actions": allowed_next_actions,
        "prohibited_next_actions": prohibited_next_actions,
    }


def oracle_action_heatmap_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("valid", "")).lower() in {"true", "1"} or record.get("valid") is True:
            by_episode[str(record["episode_id"])].append(record)
    oracle_by_episode = {
        episode_id: max(items, key=lambda row: float(row["dice"]))["action_id"]
        for episode_id, items in by_episode.items()
    }
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode):
        for record in sorted(by_episode[episode_id], key=lambda row: str(row["action_id"])):
            action_id = str(record["action_id"])
            rows.append(
                {
                    "episode_id": episode_id,
                    "action_id": action_id,
                    "dice": float(record["dice"]),
                    "is_oracle": 1 if action_id == str(oracle_by_episode[episode_id]) else 0,
                }
            )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_heatmap_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "action_id", "dice", "is_oracle"])
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint_md(path: Path, checkpoint: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Decision Checkpoint",
        "",
        f"Status: `{checkpoint['phase5_status']}`",
        "",
        f"Rationale: {checkpoint['rationale']}.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| v1 Gate F | {checkpoint['v1_gate_f_status']} |",
        f"| v1 Dice oracle gap | {checkpoint['v1_dice_oracle_gap']:.6f} |",
        f"| v2 Gate F | {checkpoint['v2_gate_f_status']} |",
        f"| v2 Dice oracle gap | {checkpoint['v2_dice_oracle_gap']:.6f} |",
        f"| Gate F minimum Dice gap | {checkpoint['v2_gate_f_min_dice_gap']:.3f} |",
        f"| v2 best fixed action | {checkpoint['v2_best_fixed_action_id']} |",
        f"| v2 top oracle action fraction | {checkpoint['v2_top_oracle_action_fraction']:.3f} |",
        "",
        "Allowed next actions:",
        "",
    ]
    lines.extend(f"- `{item}`" for item in checkpoint["allowed_next_actions"])
    lines.extend(["", "Prohibited next actions:", ""])
    lines.extend(f"- `{item}`" for item in checkpoint["prohibited_next_actions"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_checkpoint(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    out_dir = project_root / "reports" / "decision_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_summary = _read_json(project_root / "reports" / "oracle_gap_pilot" / "oracle_gap_pilot_summary.json")
    v2_summary = _read_json(project_root / "reports" / "oracle_gap_pilot_v2" / "oracle_gap_pilot_summary.json")
    checkpoint = build_decision_checkpoint(v1_summary, v2_summary)
    (out_dir / "phase5_decision.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    _write_checkpoint_md(out_dir / "phase5_decision.md", checkpoint)
    rows = oracle_action_heatmap_rows(_read_csv(project_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv"))
    _write_heatmap_csv(out_dir / "oracle_action_heatmap.csv", rows)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Phase 5 decision checkpoint after Gate F.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_decision_checkpoint(args.project_root)


if __name__ == "__main__":
    main()
