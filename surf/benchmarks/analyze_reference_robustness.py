from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_reference_robustness(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_case_id"]].append(row)
    per_query: list[dict[str, Any]] = []
    for query_id, group in sorted(grouped.items()):
        values = [float(row["dice"]) for row in group]
        actions = {row.get("selected_action") or row.get("method") or "" for row in group}
        per_query.append({
            "query_case_id": query_id,
            "reference_count": len(group),
            "mean_dice": mean(values),
            "reference_std_dice": pstdev(values),
            "reference_range_dice": max(values) - min(values),
            "unique_actions": len(actions),
        })
    summary = {
        "query_count": len(per_query),
        "reference_episode_count": len(rows),
        "mean_query_dice": mean(row["mean_dice"] for row in per_query),
        "mean_reference_std": mean(row["reference_std_dice"] for row in per_query),
        "mean_reference_range": mean(row["reference_range_dice"] for row in per_query),
        "mean_unique_actions": mean(row["unique_actions"] for row in per_query),
    }
    return per_query, summary


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify within-query sensitivity to multiple train-only references.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = resolve(args.out_dir)
    summaries: dict[str, Any] = {}
    for name, path_text in (("fixed", args.fixed), ("learned", args.learned)):
        per_query, summary = summarize_reference_robustness(read_csv(resolve(path_text)))
        write_csv(out_dir / f"{name}_per_query.csv", per_query)
        summaries[name] = summary
    summaries["delta_learned_minus_fixed"] = {
        key: summaries["learned"][key] - summaries["fixed"][key]
        for key in ("mean_query_dice", "mean_reference_std", "mean_reference_range", "mean_unique_actions")
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
