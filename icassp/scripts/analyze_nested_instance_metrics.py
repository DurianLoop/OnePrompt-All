"""Aggregate frozen nested-CV instance-oriented metrics at the query level.

This is a diagnostic companion to the primary Dice analysis.  It joins the
already selected fixed and learned predictions by episode, averages five
references per outer query, and applies paired bootstrap at 37 queries.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "runs" / "research_direction" / "nested_cv5_negative_budget" / "aggregate"
OUT = ROOT / "runs" / "nested_instance_metrics_v1"
SEED = 20260715


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_summary(learned: np.ndarray, fixed: np.ndarray, lower_is_better: bool, seed: int) -> dict[str, float | int]:
    raw_delta = learned - fixed
    favorable_delta = -raw_delta if lower_is_better else raw_delta
    rng = np.random.default_rng(seed)
    boot = np.asarray([raw_delta[rng.integers(0, len(raw_delta), len(raw_delta))].mean() for _ in range(10_000)])
    return {
        "fixed_mean": float(fixed.mean()), "learned_mean": float(learned.mean()), "delta_learned_minus_fixed": float(raw_delta.mean()),
        "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)),
        "favorable_win": int((favorable_delta > 1e-12).sum()), "tie": int((np.abs(favorable_delta) <= 1e-12).sum()), "favorable_loss": int((favorable_delta < -1e-12).sum()),
    }


def main() -> None:
    fixed_rows = read_csv(SOURCE / "combined_fixed_predictions.csv")
    learned_rows = read_csv(SOURCE / "combined_learned_predictions.csv")
    fixed = {row["episode_id"]: row for row in fixed_rows}
    learned = {row["episode_id"]: row for row in learned_rows}
    if len(fixed) != 185 or set(fixed) != set(learned):
        raise RuntimeError("nested fixed/learned episode join is incomplete")
    by_query: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    for episode_id in sorted(fixed):
        by_query.setdefault(fixed[episode_id]["query_case_id"], []).append((fixed[episode_id], learned[episode_id]))
    if len(by_query) != 37 or any(len(values) != 5 for values in by_query.values()):
        raise RuntimeError("expected 37 five-reference outer queries")
    rows: list[dict[str, float | str]] = []
    for query, pairs in sorted(by_query.items()):
        rows.append({"query_case_id": query, **{f"fixed_{metric}": float(np.mean([float(a[metric]) for a, _ in pairs])) for metric in ("dice", "object_recall", "merge_error")}, **{f"learned_{metric}": float(np.mean([float(b[metric]) for _, b in pairs])) for metric in ("dice", "object_recall", "merge_error")}})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {
        "status": "DIAGNOSTIC_COMPLETE", "protocol": "nested_cv5_negative_budget", "query_count": len(rows), "references_per_query": 5,
        "bootstrap_resamples": 10000, "seed": SEED,
        "metrics": {
            "dice": metric_summary(np.asarray([r["learned_dice"] for r in rows]), np.asarray([r["fixed_dice"] for r in rows]), False, SEED),
            "object_recall": metric_summary(np.asarray([r["learned_object_recall"] for r in rows]), np.asarray([r["fixed_object_recall"] for r in rows]), False, SEED + 1),
            "merge_error": metric_summary(np.asarray([r["learned_merge_error"] for r in rows]), np.asarray([r["fixed_merge_error"] for r in rows]), True, SEED + 2),
        },
        "scope": "diagnostic instance-oriented metrics; Dice remains the predeclared primary endpoint",
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
