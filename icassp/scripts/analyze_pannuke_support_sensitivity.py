"""Descriptive support-stratified sensitivity analysis for the frozen PanNuke cohort.

This consumes only the already fixed majority-gate episode table.  It changes no
model, threshold, support bank, or primary query-level inference; results are
explicitly post-hoc descriptive strata rather than selection evidence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pannuke_zero_shot_v1"
SEED = 20260809


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_summary(candidate: np.ndarray, reference: np.ndarray, seed_offset: int) -> dict[str, object]:
    delta = candidate - reference
    rng = np.random.default_rng(SEED + seed_offset)
    samples = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10_000)])
    return {
        "mean_dice": float(candidate.mean()), "delta_dice": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)),
        "win": int((delta > 1e-12).sum()), "tie": int((np.abs(delta) <= 1e-12).sum()), "loss": int((delta < -1e-12).sum()),
    }


def main() -> None:
    rows = read_csv(RUN / "per_episode_majority_gate.csv")
    if len(rows) != 12700:
        raise RuntimeError(f"expected 12700 episodes, found {len(rows)}")
    by_support: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_support.setdefault(row["support_case_id"], []).append(row)
    result: list[dict[str, object]] = []
    for index, (support, group) in enumerate(sorted(by_support.items())):
        if len(group) != 2540 or len({row["query_case_id"] for row in group}) != 2540:
            raise RuntimeError(f"support stratum incomplete: {support}")
        n0 = np.asarray([float(row["n0_dice"]) for row in group])
        n5 = np.asarray([float(row["n5_dice"]) for row in group])
        gate = np.asarray([float(row["selected_dice"]) for row in group])
        result.append({
            "support_case_id": support, "query_count": len(group), "n0_mean_dice": float(n0.mean()), "n5_mean_dice": float(n5.mean()),
            "gate_mean_dice": float(gate.mean()),
            **{f"gate_vs_n0_{key}": value for key, value in paired_summary(gate, n0, 10 * index).items()},
            **{f"n5_vs_n0_{key}": value for key, value in paired_summary(n5, n0, 10 * index + 1).items()},
        })
    with (RUN / "support_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    report = {
        "status": "DESCRIPTIVE_COMPLETE", "scope": "post-hoc support-stratified sensitivity; no support/model selection or protocol change",
        "query_count_per_stratum": 2540, "support_count": len(result), "bootstrap_resamples": 10000, "seed": SEED,
        "all_gate_vs_n0_ci_lower_positive": all(float(row["gate_vs_n0_ci95_low"]) > 0 for row in result),
        "all_n5_vs_n0_ci_upper_negative": all(float(row["n5_vs_n0_ci95_high"]) < 0 for row in result),
        "supports": result,
    }
    (RUN / "support_sensitivity_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
