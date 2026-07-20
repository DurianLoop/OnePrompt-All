from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def raw_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])


def action_name(row: dict[str, Any]) -> str:
    return str(row.get("selected_action") or row.get("action") or row.get("method") or "")


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[raw_id(row)].append(row)
    return grouped


def align_single_rows(
    rows: list[dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    query_to_group: dict[str, str] = {}
    duplicate_queries: set[str] = set()
    for group_id, candidates in candidate_groups.items():
        query_id = str(candidates[0]["query_case_id"])
        if query_id in query_to_group:
            duplicate_queries.add(query_id)
        query_to_group[query_id] = group_id
    aligned: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = raw_id(row)
        if candidate_id in candidate_groups:
            group_id = candidate_id
        else:
            query_id = str(row["query_case_id"])
            if query_id in duplicate_queries or query_id not in query_to_group:
                raise ValueError(f"Cannot uniquely align policy row {candidate_id} / {query_id}.")
            group_id = query_to_group[query_id]
        if group_id in aligned:
            raise ValueError(f"Expected one selected row for {group_id}.")
        aligned[group_id] = row
    if aligned.keys() != candidate_groups.keys():
        raise ValueError("Selected rows do not cover every candidate episode.")
    return aligned


def decompose_policy_regret(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_groups = group_rows(candidates)
    selected_by_id = align_single_rows(selected, candidate_groups)
    fixed_by_id = align_single_rows(fixed, candidate_groups)
    shortlist_groups = group_rows(shortlist)
    if shortlist_groups.keys() != candidate_groups.keys():
        # Legacy shortlists may only carry query ids; align one row at a time, then regroup.
        remapped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        query_to_group = {str(items[0]["query_case_id"]): group_id for group_id, items in candidate_groups.items()}
        for row in shortlist:
            group_id = raw_id(row)
            if group_id not in candidate_groups:
                group_id = query_to_group[str(row["query_case_id"])]
            remapped[group_id].append(row)
        shortlist_groups = remapped

    rows: list[dict[str, Any]] = []
    for episode_id in sorted(candidate_groups):
        episode_candidates = candidate_groups[episode_id]
        full_oracle = max(episode_candidates, key=lambda row: (float(row["dice"]), action_name(row)))
        episode_shortlist = shortlist_groups[episode_id]
        shortlist_oracle = max(episode_shortlist, key=lambda row: (float(row["dice"]), action_name(row)))
        chosen = selected_by_id[episode_id]
        anchor = fixed_by_id[episode_id]
        full_dice = float(full_oracle["dice"])
        shortlist_dice = float(shortlist_oracle["dice"])
        selected_dice = float(chosen["dice"])
        fixed_dice = float(anchor["dice"])
        rows.append({
            "episode_id": episode_id,
            "query_case_id": episode_candidates[0]["query_case_id"],
            "full_oracle_action": action_name(full_oracle),
            "shortlist_oracle_action": action_name(shortlist_oracle),
            "selected_action": action_name(chosen),
            "fixed_action": action_name(anchor),
            "full_oracle_dice": full_dice,
            "shortlist_oracle_dice": shortlist_dice,
            "selected_dice": selected_dice,
            "fixed_dice": fixed_dice,
            "prefilter_regret": full_dice - shortlist_dice,
            "selector_regret": shortlist_dice - selected_dice,
            "policy_regret": full_dice - selected_dice,
            "fixed_regret": full_dice - fixed_dice,
            "actual_switch_gain": selected_dice - fixed_dice,
            "shortlist_contains_full_oracle": int(action_name(full_oracle) in {action_name(row) for row in episode_shortlist}),
            "selected_is_full_oracle": int(action_name(chosen) == action_name(full_oracle)),
            "selected_k_error": abs(float(chosen["k"]) - float(full_oracle["k"])),
            "selected_lambda_error": abs(float(chosen["lambda_bg"]) - float(full_oracle["lambda_bg"])),
        })
    summary = {
        "episode_count": len(rows),
        "mean_full_oracle_dice": float(np.mean([row["full_oracle_dice"] for row in rows])),
        "mean_shortlist_oracle_dice": float(np.mean([row["shortlist_oracle_dice"] for row in rows])),
        "mean_selected_dice": float(np.mean([row["selected_dice"] for row in rows])),
        "mean_fixed_dice": float(np.mean([row["fixed_dice"] for row in rows])),
        "mean_prefilter_regret": float(np.mean([row["prefilter_regret"] for row in rows])),
        "mean_selector_regret": float(np.mean([row["selector_regret"] for row in rows])),
        "mean_policy_regret": float(np.mean([row["policy_regret"] for row in rows])),
        "mean_fixed_regret": float(np.mean([row["fixed_regret"] for row in rows])),
        "mean_actual_switch_gain": float(np.mean([row["actual_switch_gain"] for row in rows])),
        "shortlist_oracle_recall": float(np.mean([row["shortlist_contains_full_oracle"] for row in rows])),
        "selected_oracle_accuracy": float(np.mean([row["selected_is_full_oracle"] for row in rows])),
        "mean_selected_k_error": float(np.mean([row["selected_k_error"] for row in rows])),
        "mean_selected_lambda_error": float(np.mean([row["selected_lambda_error"] for row in rows])),
    }
    return rows, summary


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def filter_candidate_grid(
    rows: list[dict[str, Any]],
    allowed_rounds: set[str],
    allowed_k: set[int],
    allowed_lambda: set[float],
) -> list[dict[str, Any]]:
    normalized_lambda = {round(value, 6) for value in allowed_lambda}
    return [
        row for row in rows
        if (not allowed_rounds or str(row["round_name"]) in allowed_rounds)
        and (not allowed_k or int(float(row["k"])) in allowed_k)
        and (not normalized_lambda or round(float(row["lambda_bg"]), 6) in normalized_lambda)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose adaptive prompt policy regret into shortlist and selector errors.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allowed-rounds", default="")
    parser.add_argument("--allowed-k", default="")
    parser.add_argument("--allowed-lambda", default="")
    args = parser.parse_args()

    candidates = filter_candidate_grid(
        read_csv(resolve(args.candidates)),
        {item.strip() for item in args.allowed_rounds.split(",") if item.strip()},
        {int(item.strip()) for item in args.allowed_k.split(",") if item.strip()},
        {float(item.strip()) for item in args.allowed_lambda.split(",") if item.strip()},
    )
    rows, summary = decompose_policy_regret(
        candidates,
        read_csv(resolve(args.selected)),
        read_csv(resolve(args.fixed)),
        read_csv(resolve(args.shortlist)),
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "per_episode_regret.csv", rows)
    (out_dir / "regret_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
