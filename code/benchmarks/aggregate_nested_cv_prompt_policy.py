from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_prompt_policy_results import (
    aggregate_paired_by_query,
    pair_policy_rows,
    paired_policy_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_outer_test_coverage(
    rows: list[dict[str, Any]],
    expected_reference_count: int,
) -> dict[str, int]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_case_id"])].append(row)
    for query_id, query_rows in by_query.items():
        folds = {str(row["fold"]) for row in query_rows}
        if len(folds) != 1:
            raise ValueError(f"Query {query_id} appears in multiple outer folds: {sorted(folds)}")
        if len(query_rows) != expected_reference_count:
            raise ValueError(
                f"Query {query_id} has {len(query_rows)} references; expected {expected_reference_count}."
            )
    return {
        "query_count": len(by_query),
        "episode_count": len(rows),
        "fold_count": len({str(row["fold"]) for row in rows}),
    }


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate leakage-free MoNuSeg nested-CV policy predictions.")
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--expected-query-count", type=int, default=37)
    parser.add_argument("--expected-reference-count", type=int, default=5)
    parser.add_argument("--learned-filename", default="test_anchored_predictions.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    fold_root = resolve(args.fold_root)
    fixed_rows: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(args.fold_count):
        policy_dir = fold_root / f"fold_{fold}" / "policy"
        fixed = [{**row, "fold": fold} for row in read_csv(policy_dir / "test_fixed_predictions.csv")]
        learned = [{**row, "fold": fold} for row in read_csv(policy_dir / args.learned_filename)]
        fixed_keys = {str(row.get("episode_id") or row.get("case_name")) for row in fixed}
        learned_keys = {str(row.get("episode_id") or row.get("case_name")) for row in learned}
        if fixed_keys != learned_keys:
            raise ValueError(f"Fold {fold} has unpaired fixed and learned episodes.")
        fold_pairs = pair_policy_rows(fixed, learned)
        fold_cases = aggregate_paired_by_query(fold_pairs)
        fold_summaries.append({
            "fold": fold,
            **paired_policy_summary(fold_cases, bootstrap_samples=args.bootstrap_samples),
            "episode_count": len(fold_pairs),
        })
        fixed_rows.extend(fixed)
        learned_rows.extend(learned)

    coverage = validate_outer_test_coverage(fixed_rows, args.expected_reference_count)
    if coverage["query_count"] != args.expected_query_count:
        raise ValueError(
            f"Nested-CV covers {coverage['query_count']} queries; expected {args.expected_query_count}."
        )
    paired = pair_policy_rows(fixed_rows, learned_rows)
    aggregated = aggregate_paired_by_query(paired)
    summary = paired_policy_summary(aggregated, bootstrap_samples=args.bootstrap_samples)
    summary.update({
        "episode_count": len(paired),
        "fold_count": args.fold_count,
        "reference_count_per_query": args.expected_reference_count,
        "inference_unit": "outer_test_query_after_reference_mean",
        "official_test_used": False,
    })

    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "combined_fixed_predictions.csv", fixed_rows)
    write_csv(out_dir / "combined_learned_predictions.csv", learned_rows)
    write_csv(out_dir / "paired_episode_deltas.csv", paired)
    write_csv(out_dir / "paired_case_deltas.csv", aggregated)
    write_csv(out_dir / "fold_summary.csv", fold_summaries)
    (out_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
