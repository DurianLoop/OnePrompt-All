"""Phase R reference pool audit.

Checks whether the Phase Y reference pool can support Phase R's requirements:
- 6 references per query (A=3, B=3, non-overlapping)
- 5 resampling seeds on train
- 1 fixed A/B on validation
- References must be patient-disjoint from the query

CPU-only. No SAM inference.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = PROJECT_ROOT / "manifests" / "phaseY"
REPORTS = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS / "phaseR_reference_pool_audit"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    print("=" * 70)
    print("Phase R Reference Pool Audit")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    pool = _read_csv(MANIFESTS / "reference_pool.csv")
    train_queries = _read_csv(MANIFESTS / "policy_train_queries.csv")
    val_queries = _read_csv(MANIFESTS / "policy_validation_queries.csv")
    holdout_queries = _read_csv(MANIFESTS / "confirmatory_holdout_queries.csv")
    split_summary = _read_json(REPORTS / "phaseY_data_audit" / "split_summary.json")

    print(f"Reference pool size: {len(pool)} cases")
    print(f"Train queries: {len(train_queries)}")
    print(f"Validation queries: {len(val_queries)}")
    print(f"Holdout queries: {len(holdout_queries)}")
    print()

    # Build patient -> cases mapping from reference pool
    pool_by_patient: dict[str, list[str]] = defaultdict(list)
    pool_case_to_patient: dict[str, str] = {}
    for row in pool:
        case_id = row.get("case_id", "")
        patient = row.get("patient", "")
        pool_by_patient[patient].append(case_id)
        pool_case_to_patient[case_id] = patient

    print(f"Reference pool patients: {len(pool_by_patient)}")
    for patient, cases in sorted(pool_by_patient.items(), key=lambda x: -len(x[1])):
        print(f"  {patient}: {len(cases)} cases")
    print()

    # Build query -> patient mapping
    def build_query_patient_map(queries: list[dict[str, str]]) -> dict[str, str]:
        return {q.get("case_id", ""): q.get("patient", "") for q in queries}

    train_qp = build_query_patient_map(train_queries)
    val_qp = build_query_patient_map(val_queries)
    holdout_qp = build_query_patient_map(holdout_queries)

    # For each query, count available references (patient-disjoint)
    def audit_queries(queries: list[dict[str, str]], label: str) -> list[dict]:
        results = []
        for q in queries:
            qid = q.get("case_id", "")
            qpatient = q.get("patient", "")
            # Available references: all pool cases NOT from query patient
            available = [c for c in pool_case_to_patient if pool_case_to_patient[c] != qpatient]
            available_count = len(available)
            # How many distinct patients in available
            available_patients = set(pool_case_to_patient[c] for c in available)
            results.append({
                "query_case_id": qid,
                "query_patient": qpatient,
                "available_references": available_count,
                "available_patients": len(available_patients),
                "sufficient_for_6": available_count >= 6,
                "sufficient_for_5seeds_x_6": available_count >= 6,  # seeds can reuse pool
            })
        return results

    train_audit = audit_queries(train_queries, "train")
    val_audit = audit_queries(val_queries, "validation")
    holdout_audit = audit_queries(holdout_queries, "holdout")

    # Check if 5 resampling seeds can produce non-identical A/B splits
    # For a query with N available references, the number of ways to choose 6 is C(N, 6)
    # We need at least 5 distinct samples
    def check_resampling_feasibility(audit: list[dict]) -> dict:
        min_available = min(a["available_references"] for a in audit)
        # C(6, 3) = 20 ways to split 6 into A=3/B=3
        # We need 5 seeds, each drawing 6 from the pool
        # With min_available >= 6, we can draw 5 different samples
        feasible = min_available >= 6
        distinct_samples_possible = min_available >= 7  # Need at least 7 to have 2 different samples of 6
        return {
            "min_available": min_available,
            "feasible_for_6": feasible,
            "feasible_for_5_seeds": min_available >= 10,  # Conservative: need enough for 5 non-identical draws
        }

    train_resamp = check_resampling_feasibility(train_audit)
    val_resamp = check_resampling_feasibility(val_audit)

    # Summary
    print("Audit Results:")
    print(f"  Train: min available refs = {train_resamp['min_available']}, feasible for 6 = {train_resamp['feasible_for_6']}")
    print(f"  Train: feasible for 5 seeds = {train_resamp['feasible_for_5_seeds']}")
    print(f"  Validation: min available refs = {val_resamp['min_available']}, feasible for 6 = {val_resamp['feasible_for_6']}")
    print()

    # Check for any insufficient queries
    insufficient_train = [a for a in train_audit if not a["sufficient_for_6"]]
    insufficient_val = [a for a in val_audit if not a["sufficient_for_6"]]

    if insufficient_train:
        print(f"  WARNING: {len(insufficient_train)} train queries have < 6 available references!")
        for a in insufficient_train:
            print(f"    {a['query_case_id']}: only {a['available_references']} available")
    if insufficient_val:
        print(f"  WARNING: {len(insufficient_val)} validation queries have < 6 available references!")
        for a in insufficient_val:
            print(f"    {a['query_case_id']}: only {a['available_references']} available")

    if not insufficient_train and not insufficient_val:
        print("  All queries have sufficient references (>= 6 patient-disjoint available).")

    # Demonstrate 5 resampling seeds on first train query
    if train_audit:
        first_q = train_audit[0]
        qpatient = first_q["query_patient"]
        available = [c for c in pool_case_to_patient if pool_case_to_patient[c] != qpatient]
        print(f"\n  Resampling demo for {first_q['query_case_id']} (patient={qpatient}):")
        print(f"    Available references: {len(available)}")
        rng = random.Random(20260722)
        for seed_idx in range(5):
            seed = 20260722 + seed_idx * 1000
            rng2 = random.Random(seed)
            sample = rng2.sample(available, 6)
            a_set = sample[:3]
            b_set = sample[3:]
            print(f"    Seed {seed}: A={a_set}, B={b_set}")

    # Write audit CSV
    headers = [
        "query_case_id", "query_patient", "split",
        "available_references", "available_patients",
        "sufficient_for_6", "sufficient_for_5seeds",
    ]
    rows = []
    for a in train_audit:
        rows.append([a["query_case_id"], a["query_patient"], "train",
                      str(a["available_references"]), str(a["available_patients"]),
                      str(a["sufficient_for_6"]), str(a["sufficient_for_5seeds_x_6"])])
    for a in val_audit:
        rows.append([a["query_case_id"], a["query_patient"], "validation",
                      str(a["available_references"]), str(a["available_patients"]),
                      str(a["sufficient_for_6"]), str(a["sufficient_for_5seeds_x_6"])])
    for a in holdout_audit:
        rows.append([a["query_case_id"], a["query_patient"], "holdout",
                      str(a["available_references"]), str(a["available_patients"]),
                      str(a["sufficient_for_6"]), str(a["sufficient_for_5seeds_x_6"])])

    out_csv = OUTPUT_DIR / "reference_pool_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)

    # Write summary JSON
    summary = {
        "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_pool_size": len(pool),
        "reference_pool_patients": len(pool_by_patient),
        "train_queries": len(train_queries),
        "validation_queries": len(val_queries),
        "holdout_queries": len(holdout_queries),
        "min_available_train": train_resamp["min_available"],
        "min_available_val": val_resamp["min_available"],
        "all_train_sufficient": len(insufficient_train) == 0,
        "all_val_sufficient": len(insufficient_val) == 0,
        "feasible_for_5_seeds": train_resamp["feasible_for_5_seeds"],
        "status": "PHASER_REFERENCE_POOL_SUFFICIENT" if not insufficient_train and not insufficient_val else "PHASER_REFERENCE_POOL_INSUFFICIENT",
    }

    out_json = OUTPUT_DIR / "reference_pool_audit.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Output: {out_csv}")
    print(f"  Output: {out_json}")
    print(f"\n  Status: {summary['status']}")

    return 0 if not insufficient_train and not insufficient_val else 1


if __name__ == "__main__":
    sys.exit(main())
