from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])


def pair_policy_rows(
    fixed_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fixed_query_ids = [str(row.get("query_case_id", "")) for row in fixed_rows]
    learned_query_ids = [str(row.get("query_case_id", "")) for row in learned_rows]
    queries_are_unique = (
        all(fixed_query_ids)
        and all(learned_query_ids)
        and len(set(fixed_query_ids)) == len(fixed_query_ids)
        and len(set(learned_query_ids)) == len(learned_query_ids)
        and set(fixed_query_ids) == set(learned_query_ids)
    )
    key = (lambda row: str(row["query_case_id"])) if queries_are_unique else row_id
    fixed_by_id = {key(row): row for row in fixed_rows}
    learned_by_id = {key(row): row for row in learned_rows}
    if fixed_by_id.keys() != learned_by_id.keys():
        missing_fixed = sorted(learned_by_id.keys() - fixed_by_id.keys())
        missing_learned = sorted(fixed_by_id.keys() - learned_by_id.keys())
        raise ValueError(f"Unpaired policy rows: missing_fixed={missing_fixed}, missing_learned={missing_learned}")
    paired: list[dict[str, Any]] = []
    for case_id in sorted(fixed_by_id):
        fixed = fixed_by_id[case_id]
        learned = learned_by_id[case_id]
        fixed_dice = float(fixed["dice"])
        learned_dice = float(learned["dice"])
        paired.append({
            "episode_id": case_id,
            "query_case_id": learned.get("query_case_id", fixed.get("query_case_id", case_id)),
            "fixed_action": fixed.get("action", fixed.get("method", "")),
            "learned_action": learned.get("action", learned.get("method", "")),
            "selection_source": learned.get("selection_source", ""),
            "fixed_dice": fixed_dice,
            "learned_dice": learned_dice,
            "delta_dice": learned_dice - fixed_dice,
        })
    return paired


def paired_policy_summary(
    paired_rows: list[dict[str, Any]],
    bootstrap_samples: int = 10000,
    seed: int = 20260715,
) -> dict[str, Any]:
    if not paired_rows:
        raise ValueError("At least one paired row is required.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    fixed = np.asarray([float(row["fixed_dice"]) for row in paired_rows], dtype=np.float64)
    learned = np.asarray([float(row["learned_dice"]) for row in paired_rows], dtype=np.float64)
    delta = learned - fixed
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot_means = delta[indices].mean(axis=1)
    if len(delta) > 1:
        leave_one_out_means = (float(delta.sum()) - delta) / (len(delta) - 1)
    else:
        leave_one_out_means = delta.copy()
    wins = int(np.sum(delta > 1e-12))
    ties = int(np.sum(np.abs(delta) <= 1e-12))
    losses = int(np.sum(delta < -1e-12))
    sign_test = exact_two_sided_sign_test(wins, losses)
    return {
        "case_count": len(paired_rows),
        "mean_fixed_dice": float(fixed.mean()),
        "mean_learned_dice": float(learned.mean()),
        "mean_delta_dice": float(delta.mean()),
        "median_delta_dice": float(np.median(delta)),
        "leave_one_out_mean_delta_min": float(leave_one_out_means.min()),
        "leave_one_out_mean_delta_max": float(leave_one_out_means.max()),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_non_tied_count": sign_test["non_tied_count"],
        "sign_test_p_two_sided": sign_test["p_value"],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def exact_two_sided_sign_test(wins: int, losses: int) -> dict[str, float | int]:
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative.")
    non_tied = wins + losses
    if non_tied == 0:
        return {"non_tied_count": 0, "p_value": 1.0}
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(non_tied, index) for index in range(tail + 1)) / (2 ** non_tied)
    return {"non_tied_count": non_tied, "p_value": min(1.0, probability)}


def aggregate_paired_by_query(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated reference episodes before case-level inference."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in paired_rows:
        query_id = str(row.get("query_case_id") or row.get("episode_id"))
        buckets.setdefault(query_id, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for query_id in sorted(buckets):
        rows = buckets[query_id]
        fixed = float(np.mean([float(row["fixed_dice"]) for row in rows]))
        learned = float(np.mean([float(row["learned_dice"]) for row in rows]))
        aggregated.append({
            "query_case_id": query_id,
            "reference_count": len(rows),
            "fixed_dice": fixed,
            "learned_dice": learned,
            "delta_dice": learned - fixed,
        })
    return aggregated


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap statistics for fixed versus learned prompt policies.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    paired = pair_policy_rows(read_csv(resolve(args.fixed)), read_csv(resolve(args.learned)))
    aggregated = aggregate_paired_by_query(paired)
    episode_summary = paired_policy_summary(paired, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    summary = paired_policy_summary(aggregated, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    summary["episode_count"] = len(paired)
    summary["inference_unit"] = "query_case_after_reference_mean"
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "paired_episode_deltas.csv", paired)
    write_csv(out_dir / "paired_case_deltas.csv", aggregated)
    (out_dir / "paired_episode_summary.json").write_text(json.dumps(episode_summary, indent=2), encoding="utf-8")
    (out_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
