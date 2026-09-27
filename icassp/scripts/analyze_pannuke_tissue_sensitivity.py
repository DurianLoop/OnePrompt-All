"""Post-hoc PanNuke tissue-code sensitivity analysis of frozen query-level outputs.

Tissue is public metadata emitted during pre-registered materialization. This
script only joins that immutable metadata to existing per-query scores and never
selects a support, action, model, or threshold.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pannuke_zero_shot_v1"
MANIFEST = ROOT / "data" / "pannuke_hf" / "materialized_zero_shot_v1" / "manifest.csv"
SEED = 20260809


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired(candidate: np.ndarray, reference: np.ndarray, seed: int) -> dict[str, float | int]:
    delta = candidate - reference
    rng = np.random.default_rng(seed)
    boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10_000)])
    return {
        "delta_dice": float(delta.mean()), "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)),
        "win": int((delta > 1e-12).sum()), "tie": int((np.abs(delta) <= 1e-12).sum()), "loss": int((delta < -1e-12).sum()),
    }


def main() -> None:
    tissue = {row["case_id"]: int(row["tissue"]) for row in read_csv(MANIFEST) if row["role"] == "query"}
    queries = read_csv(RUN / "per_query_majority_gate.csv")
    if len(queries) != 2540 or set(tissue) != {row["query_case_id"] for row in queries}:
        raise RuntimeError("PanNuke query/tissue join is incomplete")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in queries:
        grouped.setdefault(tissue[row["query_case_id"]], []).append(row)
    report_rows: list[dict[str, float | int]] = []
    for offset, (code, rows) in enumerate(sorted(grouped.items())):
        n0 = np.asarray([float(row["n0_dice"]) for row in rows])
        n5 = np.asarray([float(row["n5_dice"]) for row in rows])
        gate = np.asarray([float(row["selected_dice"]) for row in rows])
        report_rows.append({
            "tissue_code": code, "query_count": len(rows), "n0_mean_dice": float(n0.mean()), "n5_mean_dice": float(n5.mean()), "gate_mean_dice": float(gate.mean()),
            **{f"gate_vs_n0_{key}": value for key, value in paired(gate, n0, SEED + 2 * offset).items()},
            **{f"n5_vs_n0_{key}": value for key, value in paired(n5, n0, SEED + 2 * offset + 1).items()},
        })
    with (RUN / "tissue_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    report = {
        "status": "DESCRIPTIVE_COMPLETE", "scope": "post-hoc public tissue-code strata; no subgroup selection, model fitting, or protocol change",
        "tissue_code_count": len(report_rows), "query_count": len(queries), "bootstrap_resamples": 10000, "seed": SEED,
        "gate_positive_ci_strata": sum(float(row["gate_vs_n0_ci95_low"]) > 0 for row in report_rows),
        "gate_negative_ci_strata": sum(float(row["gate_vs_n0_ci95_high"]) < 0 for row in report_rows),
        "gate_indeterminate_ci_strata": sum(float(row["gate_vs_n0_ci95_low"]) <= 0 <= float(row["gate_vs_n0_ci95_high"]) for row in report_rows),
        "n5_negative_ci_strata": sum(float(row["n5_vs_n0_ci95_high"]) < 0 for row in report_rows),
        "strata": report_rows,
    }
    (RUN / "tissue_sensitivity_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
