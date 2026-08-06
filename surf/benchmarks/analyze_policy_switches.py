from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def episode_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])


def analyze_switches(
    fixed_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixed_by_episode = {episode_id(row): row for row in fixed_rows}
    learned_by_episode = {episode_id(row): row for row in learned_rows}
    if fixed_by_episode.keys() != learned_by_episode.keys():
        raise ValueError("Fixed and learned episode sets differ.")
    rows: list[dict[str, Any]] = []
    for key in sorted(fixed_by_episode):
        fixed = fixed_by_episode[key]
        learned = learned_by_episode[key]
        delta = float(learned["dice"]) - float(fixed["dice"])
        action = str(
            learned.get("selected_action")
            or learned.get("action")
            or learned.get("method")
        )
        switched = int(learned.get("switched", action != str(fixed.get("action") or fixed.get("method"))))
        rows.append({
            "episode_id": key,
            "query_case_id": learned.get("query_case_id", fixed.get("query_case_id", "")),
            "fixed_action": fixed.get("action") or fixed.get("method", ""),
            "selected_action": action,
            "switched": switched,
            "fixed_dice": float(fixed["dice"]),
            "learned_dice": float(learned["dice"]),
            "delta_dice": delta,
        })
    switched_rows = [row for row in rows if row["switched"] == 1]
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in switched_rows:
        by_action[str(row["selected_action"])].append(row)
    action_rows = [
        {
            "action": action,
            "switch_count": len(group),
            "mean_delta_dice": float(np.mean([row["delta_dice"] for row in group])),
            "wins": sum(float(row["delta_dice"]) > 1e-12 for row in group),
            "ties": sum(abs(float(row["delta_dice"])) <= 1e-12 for row in group),
            "losses": sum(float(row["delta_dice"]) < -1e-12 for row in group),
        }
        for action, group in by_action.items()
    ]
    action_rows.sort(key=lambda row: (-int(row["switch_count"]), str(row["action"])))
    switch_deltas = np.asarray([float(row["delta_dice"]) for row in switched_rows], dtype=np.float64)
    summary = {
        "episode_count": len(rows),
        "anchor_keep_count": len(rows) - len(switched_rows),
        "switch_count": len(switched_rows),
        "switch_rate": len(switched_rows) / len(rows) if rows else 0.0,
        "switch_mean_delta_dice": float(switch_deltas.mean()) if len(switch_deltas) else 0.0,
        "switch_wins": int(np.sum(switch_deltas > 1e-12)),
        "switch_ties": int(np.sum(np.abs(switch_deltas) <= 1e-12)),
        "switch_losses": int(np.sum(switch_deltas < -1e-12)),
    }
    return rows, action_rows, summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit when an anchored prompt policy switches actions.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows, action_rows, summary = analyze_switches(
        read_csv(resolve(args.fixed)),
        read_csv(resolve(args.learned)),
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "per_episode_switches.csv", rows)
    write_csv(out_dir / "action_summary.csv", action_rows)
    (out_dir / "switch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
