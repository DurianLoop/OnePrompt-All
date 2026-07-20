from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from all_nuclei_utility import AllNucleiWeights, add_all_nuclei_utility


def episode_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])


def compare_oracles(
    rows: list[dict[str, Any]],
    weights: AllNucleiWeights,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched = add_all_nuclei_utility(rows, weights)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[episode_id(row)].append(row)

    per_episode: list[dict[str, Any]] = []
    for group_id, candidates in sorted(grouped.items()):
        dice_best = max(candidates, key=lambda row: (float(row["dice"]), str(row["method"])))
        utility_best = max(candidates, key=lambda row: (float(row["all_nuclei_utility"]), str(row["method"])))
        per_episode.append({
            "episode_id": group_id,
            "query_case_id": dice_best["query_case_id"],
            "dice_oracle_action": dice_best["method"],
            "utility_oracle_action": utility_best["method"],
            "same_action": int(dice_best["method"] == utility_best["method"]),
            "dice_oracle_dice": float(dice_best["dice"]),
            "utility_oracle_dice": float(utility_best["dice"]),
            "dice_oracle_object_recall": float(dice_best["object_recall"]),
            "utility_oracle_object_recall": float(utility_best["object_recall"]),
            "dice_oracle_merge_error": float(dice_best["merge_error"]),
            "utility_oracle_merge_error": float(utility_best["merge_error"]),
            "dice_oracle_area_error": float(dice_best["area_error"]),
            "utility_oracle_area_error": float(utility_best["area_error"]),
            "dice_oracle_utility": float(dice_best["all_nuclei_utility"]),
            "utility_oracle_utility": float(utility_best["all_nuclei_utility"]),
        })

    if not per_episode:
        raise ValueError("At least one candidate episode is required.")
    summary: dict[str, Any] = {
        "episode_count": len(per_episode),
        "oracle_action_agreement": mean(row["same_action"] for row in per_episode),
        "weights": {
            "object_recall": weights.object_recall,
            "area_error": weights.area_error,
            "merge_error": weights.merge_error,
        },
    }
    for prefix in ("dice_oracle", "utility_oracle"):
        for metric in ("dice", "object_recall", "merge_error", "area_error", "utility"):
            summary[f"mean_{prefix}_{metric}"] = mean(float(row[f"{prefix}_{metric}"]) for row in per_episode)
    return per_episode, summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Dice and object-aware all-nuclei candidate oracles.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--object-recall-weight", type=float, default=0.2)
    parser.add_argument("--area-error-weight", type=float, default=0.1)
    parser.add_argument("--merge-error-weight", type=float, default=0.3)
    args = parser.parse_args()

    weights = AllNucleiWeights(
        object_recall=args.object_recall_weight,
        area_error=args.area_error_weight,
        merge_error=args.merge_error_weight,
    )
    per_episode, summary = compare_oracles(read_csv(Path(args.candidates)), weights)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "per_episode_oracle_comparison.csv", per_episode)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
