"""Phase R manifest builder.

Creates train resampling (5 seeds) and validation A/B manifests per directive section 5.

- Train (80 queries): 5 seeds, each producing 6 references (A=3, B=3)
- Validation (41 queries): 1 fixed seed, 6 references (A=3, B=3)
- References are patient-disjoint from query
- A/B are non-overlapping within each seed
- Holdout (40 queries): NOT created (locked until Gate R3 pass)

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
MANIFESTS_IN = PROJECT_ROOT / "manifests" / "phaseY"
MANIFESTS_OUT = PROJECT_ROOT / "manifests" / "phaseR"
REPORTS = PROJECT_ROOT / "reports"

# Fixed seeds per directive section 5
TRAIN_SEEDS = [20260722, 20261722, 20262722, 20263722, 20264722]
VALIDATION_SEED = 20260722  # Single fixed seed for validation
REFS_PER_QUERY = 6  # A=3, B=3


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def main() -> int:
    print("=" * 70)
    print("Phase R Manifest Builder")
    print("=" * 70)
    print()

    MANIFESTS_OUT.mkdir(parents=True, exist_ok=True)

    # Load reference pool and queries
    pool = _read_csv(MANIFESTS_IN / "reference_pool.csv")
    train_queries = _read_csv(MANIFESTS_IN / "policy_train_queries.csv")
    val_queries = _read_csv(MANIFESTS_IN / "policy_validation_queries.csv")

    print(f"Reference pool: {len(pool)} cases")
    print(f"Train queries: {len(train_queries)}")
    print(f"Validation queries: {len(val_queries)}")
    print()

    # Build patient -> cases mapping
    pool_case_to_patient = {row["case_id"]: row["patient"] for row in pool}
    pool_cases = list(pool_case_to_patient.keys())

    # Manifest row format (matches Phase Y + seed column)
    headers = [
        "query_case_id", "query_patient", "query_fold",
        "reference_case_id", "reference_patient",
        "reference_set", "reference_index", "seed",
    ]

    # === Train resampling: 5 seeds ===
    print("Building train resampling manifests (5 seeds)...")
    all_train_rows: list[list[str]] = []
    per_seed_rows: dict[int, list[list[str]]] = {}

    for seed in TRAIN_SEEDS:
        seed_rows: list[list[str]] = []
        rng = random.Random(seed)

        for q in train_queries:
            qid = q["case_id"]
            qpatient = q["patient"]
            # Available references: patient-disjoint from query
            available = [c for c in pool_cases if pool_case_to_patient[c] != qpatient]
            # Sample 6 references
            sample = rng.sample(available, REFS_PER_QUERY)
            # Split into A=3, B=3
            a_refs = sample[:3]
            b_refs = sample[3:]

            for idx, ref_id in enumerate(a_refs):
                row = [qid, qpatient, "train", ref_id, pool_case_to_patient[ref_id],
                       "A", str(idx), str(seed)]
                seed_rows.append(row)
                all_train_rows.append(row)
            for idx, ref_id in enumerate(b_refs):
                row = [qid, qpatient, "train", ref_id, pool_case_to_patient[ref_id],
                       "B", str(idx), str(seed)]
                seed_rows.append(row)
                all_train_rows.append(row)

        per_seed_rows[seed] = seed_rows
        # Write per-seed file
        seed_csv = MANIFESTS_OUT / f"train_resampling_seed_{TRAIN_SEEDS.index(seed)}.csv"
        _write_csv(seed_csv, headers, seed_rows)
        print(f"  Seed {seed} -> {seed_csv.name}: {len(seed_rows)} rows ({len(train_queries)} queries × 6 refs)")

    # Combined train manifest
    train_combined = MANIFESTS_OUT / "train_resampling_all_seeds.csv"
    _write_csv(train_combined, headers, all_train_rows)
    print(f"  Combined -> {train_combined.name}: {len(all_train_rows)} rows")
    print()

    # === Validation: 1 fixed A/B ===
    print("Building validation A/B manifest (fixed seed)...")
    val_rows: list[list[str]] = []
    rng = random.Random(VALIDATION_SEED)

    for q in val_queries:
        qid = q["case_id"]
        qpatient = q["patient"]
        available = [c for c in pool_cases if pool_case_to_patient[c] != qpatient]
        sample = rng.sample(available, REFS_PER_QUERY)
        a_refs = sample[:3]
        b_refs = sample[3:]

        for idx, ref_id in enumerate(a_refs):
            row = [qid, qpatient, "validation", ref_id, pool_case_to_patient[ref_id],
                   "A", str(idx), str(VALIDATION_SEED)]
            val_rows.append(row)
        for idx, ref_id in enumerate(b_refs):
            row = [qid, qpatient, "validation", ref_id, pool_case_to_patient[ref_id],
                   "B", str(idx), str(VALIDATION_SEED)]
            val_rows.append(row)

    val_csv = MANIFESTS_OUT / "validation_AB_fixed.csv"
    _write_csv(val_csv, headers, val_rows)
    print(f"  {val_csv.name}: {len(val_rows)} rows ({len(val_queries)} queries × 6 refs)")
    print()

    # === Verification ===
    print("Verifying manifests...")

    # Check A/B non-overlap within each seed
    def verify_ab_disjoint(rows: list[list[str]], label: str) -> bool:
        by_query_seed: dict[tuple[str, str], dict[str, set]] = defaultdict(lambda: {"A": set(), "B": set()})
        for row in rows:
            qid = row[0]
            seed = row[7]
            rset = row[5]
            ref_id = row[3]
            by_query_seed[(qid, seed)][rset].add(ref_id)

        overlaps = 0
        for (qid, seed), sets in by_query_seed.items():
            if sets["A"] & sets["B"]:
                overlaps += 1
                print(f"  WARNING: {label} query={qid} seed={seed} has A/B overlap!")

        if overlaps == 0:
            print(f"  {label}: A/B non-overlap verified for all {len(by_query_seed)} query×seed pairs.")
            return True
        else:
            print(f"  {label}: {overlaps} overlaps found!")
            return False

    # Check patient-disjoint from query
    def verify_patient_disjoint(rows: list[list[str]], label: str) -> bool:
        violations = 0
        for row in rows:
            qpatient = row[1]
            ref_patient = row[4]
            if qpatient == ref_patient:
                violations += 1
                print(f"  WARNING: {label} query={row[0]} ref={row[3]} same patient!")
        if violations == 0:
            print(f"  {label}: patient-disjoint verified for all {len(rows)} rows.")
            return True
        else:
            print(f"  {label}: {violations} patient-disjoint violations!")
            return False

    train_ok = verify_ab_disjoint(all_train_rows, "Train") and verify_patient_disjoint(all_train_rows, "Train")
    val_ok = verify_ab_disjoint(val_rows, "Validation") and verify_patient_disjoint(val_rows, "Validation")

    # Check 5 seeds produce different samples
    def verify_seed_diversity(rows: list[list[str]]) -> None:
        by_query_seed: dict[tuple[str, str], set] = defaultdict(set)
        for row in rows:
            qid = row[0]
            seed = row[7]
            ref_id = row[3]
            by_query_seed[(qid, seed)].add(ref_id)

        # For each query, check that 5 seeds produce different reference sets
        queries = set(r[0] for r in rows)
        diverse = 0
        for qid in sorted(queries):
            seed_sets = [by_query_seed[(qid, str(s))] for s in TRAIN_SEEDS if (qid, str(s)) in by_query_seed]
            if len(seed_sets) >= 2:
                # Check at least one pair is different
                all_same = all(s == seed_sets[0] for s in seed_sets[1:])
                if not all_same:
                    diverse += 1

        print(f"  Seed diversity: {diverse}/{len(queries)} queries have different reference sets across 5 seeds.")

    verify_seed_diversity(all_train_rows)

    # Write summary
    summary = {
        "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "train_seeds": TRAIN_SEEDS,
        "validation_seed": VALIDATION_SEED,
        "refs_per_query": REFS_PER_QUERY,
        "train_query_count": len(train_queries),
        "validation_query_count": len(val_queries),
        "train_manifest_rows": len(all_train_rows),
        "validation_manifest_rows": len(val_rows),
        "verification": {
            "train_ab_disjoint": train_ok,
            "train_patient_disjoint": train_ok,
            "validation_ab_disjoint": val_ok,
            "validation_patient_disjoint": val_ok,
        },
        "status": "PHASER_MANIFESTS_COMPLETE" if train_ok and val_ok else "PHASER_MANIFESTS_ERROR",
    }

    summary_path = MANIFESTS_OUT / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary: {summary_path}")
    print(f"  Status: {summary['status']}")

    return 0 if train_ok and val_ok else 1


if __name__ == "__main__":
    sys.exit(main())
