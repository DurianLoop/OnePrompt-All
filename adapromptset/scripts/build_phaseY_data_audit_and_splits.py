"""Phase Y — Fresh-Query Data Audit, Split, and Reference Set Builder.

Implements directive D:\\SAM\\adapromptset\\AdaPromptSet_PhaseY_Policy_Feasibility.md
sections 5.1-5.3.

Outputs (all under D:\\SAM\\adapromptset\\):
  reports/phaseY_data_audit/
    image_mask_pairing.csv
    case_identity_map.csv
    prior_query_exposure.csv
    prior_reference_exposure.csv
    eligible_fresh_queries.csv
    duplicate_hash_audit.csv
    data_audit_summary.md
    split_summary.json
  manifests/phaseY/
    policy_train_queries.csv
    policy_validation_queries.csv
    confirmatory_holdout_queries.csv
    reference_pool.csv
    reference_set_A.csv
    reference_set_B.csv
  reports/phaseY_reference_audit.json

No SAM inference is run. Read-only over MoNuSAC processed manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\SAM\adapromptset")
MANIFEST_SRC = Path(r"D:\SAM\code\datasets\MoNuSAC\processed\monusac_processed_manifest.csv")
AUDIT_DIR = PROJECT_ROOT / "reports" / "phaseY_data_audit"
SPLIT_DIR = PROJECT_ROOT / "manifests" / "phaseY"

# ---------------------------------------------------------------------------
# Phase X exposed case_ids (from reports/phaseX_search_oracle/search_oracle_summary.md)
# 10 query cases + 2 reference cases (train_004, train_005 both TCGA-55-1594)
# ---------------------------------------------------------------------------
PHASEX_QUERY_CASES = [
    "train_111",
    "train_190",
    "train_132",
    "train_204",
    "train_200",
    "train_116",
    "train_123",
    "train_207",
    "train_113",
    "train_017",
]
PHASEX_REFERENCE_CASES = ["train_004", "train_005"]
PHASEX_EXPOSED_CASES = set(PHASEX_QUERY_CASES + PHASEX_REFERENCE_CASES)

# Split targets per directive section 5.2
TARGET_TRAIN = 80
TARGET_VAL = 40
TARGET_HOLDOUT = 40

# Reference design per directive section 5.3
REFS_PER_QUERY_PER_SET = 3  # Set A: 3, Set B: 3
SPLIT_SEED = 20260721  # frozen sampling seed
REFERENCE_SEED = 20260721


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def read_manifest() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(MANIFEST_SRC, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 5.1 Case identity & prior exposure audit
# ---------------------------------------------------------------------------
def build_case_identity_map(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """case_identity_map.csv — case_id, split, patient, tissue, instance_count,
    classes_present, case_role (train/test), phasex_exposure."""
    out = []
    for r in rows:
        case_id = r["case_id"]
        exposure = "none"
        if case_id in PHASEX_QUERY_CASES:
            exposure = "phasex_query"
        elif case_id in PHASEX_REFERENCE_CASES:
            exposure = "phasex_reference"
        out.append(
            {
                "case_id": case_id,
                "split": r["split"],
                "patient": r["patient"],
                "tissue": r["tissue"],
                "instance_count": r["instance_count"],
                "classes_present": r["classes_present"],
                "case_role": r["split"],
                "phasex_exposure": exposure,
            }
        )
    return out


def build_image_mask_pairing(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """image_mask_pairing.csv — verify image/mask file existence and hash.

    The processed MoNuSAC tree stores images at
    processed/<split>/images/<case_id>.png and masks at
    processed/<split>/masks/<case_id>.png. We record existence and size.
    Full SHA-256 of every image is expensive; we hash only the first 8KB to
    detect duplicates cheaply (duplicate_hash_audit).
    """
    base = MANIFEST_SRC.parent  # .../MoNuSAC/processed
    out = []
    for r in rows:
        split = r["split"]
        case_id = r["case_id"]
        img_path = base / split / "images" / f"{case_id}.png"
        mask_path = base / split / "masks" / f"{case_id}.png"
        img_exists = img_path.exists()
        mask_exists = mask_path.exists()
        img_size = img_path.stat().st_size if img_exists else 0
        mask_size = mask_path.stat().st_size if mask_exists else 0
        # cheap hash: first 8KB
        img_hash = ""
        mask_hash = ""
        if img_exists:
            with open(img_path, "rb") as f:
                img_hash = sha256_bytes(f.read(8192))
        if mask_exists:
            with open(mask_path, "rb") as f:
                mask_hash = sha256_bytes(f.read(8192))
        out.append(
            {
                "case_id": case_id,
                "split": split,
                "patient": r["patient"],
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "image_exists": str(img_exists).lower(),
                "mask_exists": str(mask_exists).lower(),
                "image_size_bytes": img_size,
                "mask_size_bytes": mask_size,
                "image_head_sha256": img_hash,
                "mask_head_sha256": mask_hash,
            }
        )
    return out


def build_prior_query_exposure() -> List[Dict[str, str]]:
    """prior_query_exposure.csv — list each Phase X query case and its exposure."""
    rows = []
    # Load manifest lookup for patient info
    manifest = {r["case_id"]: r for r in read_manifest()}
    for case_id in PHASEX_QUERY_CASES:
        r = manifest.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "patient": r.get("patient", ""),
                "exposure_type": "phasex_query",
                "exposure_source": "phaseX_search_oracle_summary.md",
                "allowed_uses": "method_discovery;debug;historical_appendix;sanity_check",
                "prohibited_uses": "policy_validation_primary_metric;confirmatory_bootstrap;formal_claim;model_selection;final_significance_test",
                "label": "EXPLORATORY_PHASEX_EVIDENCE",
            }
        )
    return rows


def build_prior_reference_exposure() -> List[Dict[str, str]]:
    """prior_reference_exposure.csv — list Phase X reference cases."""
    rows = []
    manifest = {r["case_id"]: r for r in read_manifest()}
    for case_id in PHASEX_REFERENCE_CASES:
        r = manifest.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "patient": r.get("patient", ""),
                "exposure_type": "phasex_reference",
                "exposure_source": "phaseX_search_oracle_summary.md",
                "used_for": "phasex_search_oracle_reference_episodes",
                "allowed_uses": "historical_appendix;sanity_check",
                "prohibited_uses": "phaseY_reference_pool;phaseY_query;phaseY_holdout",
                "label": "EXPLORATORY_PHASEX_EVIDENCE",
            }
        )
    return rows


def build_duplicate_hash_audit(pairing: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """duplicate_hash_audit.csv — group cases by image_head_sha256 to detect
    near-duplicates. Only flags cases sharing the same head hash."""
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for p in pairing:
        if p["image_exists"] == "true" and p["image_head_sha256"]:
            by_hash[p["image_head_sha256"]].append(p["case_id"])
    out = []
    for h, case_ids in by_hash.items():
        if len(case_ids) > 1:
            out.append(
                {
                    "image_head_sha256": h,
                    "case_ids": ";".join(case_ids),
                    "duplicate_count": len(case_ids),
                    "flag": "DUPLICATE_HEAD_HASH",
                }
            )
    if not out:
        out.append(
            {
                "image_head_sha256": "",
                "case_ids": "",
                "duplicate_count": 0,
                "flag": "NO_DUPLICATES_FOUND",
            }
        )
    return out


# ---------------------------------------------------------------------------
# 5.2 Fresh query split (patient-level)
# ---------------------------------------------------------------------------
def build_eligible_fresh_queries(
    identity: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    """eligible_fresh_queries.csv — train cases whose PATIENT has zero Phase X
    exposure. Returns (eligible_rows, patient_to_cases)."""
    # Identify Phase X exposed patients
    exposed_patients: set = set()
    for r in identity:
        if r["phasex_exposure"] in ("phasex_query", "phasex_reference"):
            exposed_patients.add(r["patient"])

    eligible: List[Dict[str, str]] = []
    patient_to_cases: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in identity:
        if r["split"] != "train":
            continue  # formal test is LOCKED
        if r["patient"] in exposed_patients:
            continue  # patient-level exclusion
        eligible.append(r)
        patient_to_cases[r["patient"]].append(r)
    return eligible, patient_to_cases


def split_patients_into_folds(
    patient_to_cases: Dict[str, List[Dict[str, str]]],
) -> Dict[str, List[Dict[str, str]]]:
    """Patient-level split targeting 80/40/40 cases.

    Strategy: sort patients by case_count desc, then greedily assign to the
    fold that is most below its target. Uses a fixed seed for determinism on
    tie-breaking.
    """
    rng = random.Random(SPLIT_SEED)
    patients = sorted(
        patient_to_cases.keys(),
        key=lambda p: (-len(patient_to_cases[p]), p),
    )
    # Shuffle tie-break: group patients with the same case count and shuffle
    grouped: Dict[int, List[str]] = defaultdict(list)
    for p in patients:
        grouped[len(patient_to_cases[p])].append(p)
    shuffled_patients: List[str] = []
    for cc in sorted(grouped.keys(), reverse=True):
        grp = grouped[cc]
        rng.shuffle(grp)
        shuffled_patients.extend(grp)

    folds: Dict[str, List[str]] = {"train": [], "val": [], "holdout": []}
    fold_case_counts: Dict[str, int] = {"train": 0, "val": 0, "holdout": 0}
    targets = {"train": TARGET_TRAIN, "val": TARGET_VAL, "holdout": TARGET_HOLDOUT}

    for p in shuffled_patients:
        n = len(patient_to_cases[p])
        # Pick the fold most below its target (normalized deficit)
        deficits = {
            f: (targets[f] - fold_case_counts[f]) / max(targets[f], 1)
            for f in folds
        }
        best_fold = max(deficits, key=deficits.get)
        folds[best_fold].append(p)
        fold_case_counts[best_fold] += n

    # Expand to case rows
    result: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "holdout": []}
    for fold, plist in folds.items():
        for p in plist:
            for case in patient_to_cases[p]:
                row = dict(case)
                row["phaseY_fold"] = fold
                row["patient_case_count"] = len(patient_to_cases[p])
                result[fold].append(row)
    return result


# ---------------------------------------------------------------------------
# 5.3 Reference Set A/B
# ---------------------------------------------------------------------------
def build_reference_sets(
    split: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """For each query in train/val/holdout, sample 3+3 non-overlapping
    references from the train fold (excluding the query's own patient).

    Per directive 5.3:
      - references come from train-only pool
      - reference != query case
      - reference patient != query patient
      - A and B disjoint per query
      - fixed seed
    """
    rng = random.Random(REFERENCE_SEED)
    train_pool = split["train"]
    # Index train pool by patient for fast exclusion
    pool_by_patient: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for c in train_pool:
        pool_by_patient[c["patient"]].append(c)

    ref_pool_rows: List[Dict[str, str]] = []
    refA_rows: List[Dict[str, str]] = []
    refB_rows: List[Dict[str, str]] = []

    # Reference pool = all train cases (they are eligible to be references)
    for c in train_pool:
        ref_pool_rows.append(
            {
                "case_id": c["case_id"],
                "patient": c["patient"],
                "tissue": c["tissue"],
                "instance_count": c["instance_count"],
                "classes_present": c["classes_present"],
                "role": "reference_pool",
            }
        )

    all_queries = (
        split["train"] + split["val"] + split["holdout"]
    )
    for q in all_queries:
        q_patient = q["patient"]
        # Eligible references: train cases NOT in q_patient
        eligible = [c for p, cl in pool_by_patient.items() if p != q_patient for c in cl]
        # Deterministic shuffle for this query
        rng.shuffle(eligible)
        if len(eligible) < 2 * REFS_PER_QUERY_PER_SET:
            # Should not happen given pool size, but handle gracefully
            raise RuntimeError(
                f"Insufficient eligible references for query {q['case_id']}: "
                f"{len(eligible)} available, need {2 * REFS_PER_QUERY_PER_SET}"
            )
        set_a = eligible[:REFS_PER_QUERY_PER_SET]
        set_b = eligible[REFS_PER_QUERY_PER_SET : 2 * REFS_PER_QUERY_PER_SET]
        for ref in set_a:
            refA_rows.append(
                {
                    "query_case_id": q["case_id"],
                    "query_patient": q_patient,
                    "query_fold": q["phaseY_fold"],
                    "reference_case_id": ref["case_id"],
                    "reference_patient": ref["patient"],
                    "reference_set": "A",
                    "reference_index": set_a.index(ref),
                }
            )
        for ref in set_b:
            refB_rows.append(
                {
                    "query_case_id": q["case_id"],
                    "query_patient": q_patient,
                    "query_fold": q["phaseY_fold"],
                    "reference_case_id": ref["case_id"],
                    "reference_patient": ref["patient"],
                    "reference_set": "B",
                    "reference_index": set_b.index(ref),
                }
            )
    return ref_pool_rows, refA_rows, refB_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(
    identity: List[Dict[str, str]],
    pairing: List[Dict[str, str]],
    duplicates: List[Dict[str, str]],
    eligible: List[Dict[str, str]],
    split: Dict[str, List[Dict[str, str]]],
    ref_pool: List[Dict[str, str]],
    refA: List[Dict[str, str]],
    refB: List[Dict[str, str]],
) -> str:
    train_cases = [r for r in identity if r["split"] == "train"]
    test_cases = [r for r in identity if r["split"] == "test"]
    train_patients = sorted({r["patient"] for r in train_cases})
    test_patients = sorted({r["patient"] for r in test_cases})
    phasex_query_patients = sorted(
        {r["patient"] for r in identity if r["phasex_exposure"] == "phasex_query"}
    )
    phasex_ref_patients = sorted(
        {r["patient"] for r in identity if r["phasex_exposure"] == "phasex_reference"}
    )
    exposed_patients = sorted(set(phasex_query_patients + phasex_ref_patients))
    fresh_patients = sorted(set(train_patients) - set(exposed_patients))

    lines = []
    lines.append("# Phase Y — Data Audit Summary")
    lines.append("")
    lines.append("Directive: `D:\\SAM\\adapromptset\\AdaPromptSet_PhaseY_Policy_Feasibility.md` section 5.1")
    lines.append("")
    lines.append("## 1. MoNuSAC Processed Manifest")
    lines.append("")
    lines.append(f"- Source: `{MANIFEST_SRC}`")
    lines.append(f"- Total cases: {len(identity)}")
    lines.append(f"- Train cases: {len(train_cases)} (train_000 … train_208)")
    lines.append(f"- Test cases: {len(test_cases)} (test_000 … test_100) — LOCKED")
    lines.append(f"- Unique train patients: {len(train_patients)}")
    lines.append(f"- Unique test patients: {len(test_patients)}")
    lines.append("")
    lines.append("## 2. Phase X Prior Exposure")
    lines.append("")
    lines.append(f"- Phase X query cases (10): {', '.join(PHASEX_QUERY_CASES)}")
    lines.append(f"- Phase X reference cases (2): {', '.join(PHASEX_REFERENCE_CASES)}")
    lines.append(f"- Phase X query patients ({len(phasex_query_patients)}): {', '.join(phasex_query_patients)}")
    lines.append(f"- Phase X reference patients ({len(phasex_ref_patients)}): {', '.join(phasex_ref_patients)}")
    lines.append("")
    lines.append("**Critical finding**: Phase X reference cases `train_004` and `train_005` are")
    lines.append("both from patient `TCGA-55-1594` — the historical reference pool used a single")
    lines.append("patient for all 10 episodes. Phase Y must use diverse, patient-disjoint references.")
    lines.append("")
    lines.append("**Patient-level leakage risk**: Phase X queries `train_111` and `train_113` are")
    lines.append("both from patient `TCGA-UZ-A9PO`. Phase Y splits at patient level to prevent")
    lines.append("any within-patient leakage between train / validation / holdout.")
    lines.append("")
    lines.append("## 3. Eligible Fresh Query Pool")
    lines.append("")
    lines.append(f"- Exposed patients (excluded): {len(exposed_patients)}")
    lines.append(f"- Fresh patients (eligible): {len(fresh_patients)}")
    lines.append(f"- Fresh cases (eligible): {len(eligible)}")
    lines.append("")
    lines.append("## 4. Image / Mask Pairing")
    lines.append("")
    paired = [r for r in pairing if r["image_exists"] == "true" and r["mask_exists"] == "true"]
    missing = [r for r in pairing if r["image_exists"] != "true" or r["mask_exists"] != "true"]
    lines.append(f"- Paired (image+mask exist): {len(paired)}")
    lines.append(f"- Missing image or mask: {len(missing)}")
    lines.append("")
    lines.append("## 5. Duplicate Hash Audit")
    lines.append("")
    if duplicates and duplicates[0]["flag"] == "NO_DUPLICATES_FOUND":
        lines.append("- No duplicates found by 8KB head hash.")
    else:
        lines.append(f"- Duplicate groups: {len(duplicates)}")
        for d in duplicates:
            lines.append(f"  - {d['duplicate_count']} cases share hash {d['image_head_sha256'][:16]}…: {d['case_ids']}")
    lines.append("")
    lines.append("## 6. Patient-Level Split (target 80/40/40)")
    lines.append("")
    for fold in ("train", "val", "holdout"):
        cases = split[fold]
        patients = sorted({c["patient"] for c in cases})
        lines.append(f"- {fold}: {len(cases)} cases across {len(patients)} patients")
    total = sum(len(split[f]) for f in split)
    lines.append(f"- Total: {total} cases")
    lines.append("")
    lines.append("Split strategy: sort patients by case count desc, shuffle ties with fixed seed")
    lines.append(f"({SPLIT_SEED}), greedily assign to the fold with the largest normalized")
    lines.append("deficit. Patient-level disjoint: no patient appears in more than one fold.")
    lines.append("")
    lines.append("## 7. Reference Sets A / B")
    lines.append("")
    lines.append(f"- Reference pool: {len(ref_pool)} train cases (all train-fold cases)")
    lines.append(f"- Reference Set A rows: {len(refA)} (3 per query)")
    lines.append(f"- Reference Set B rows: {len(refB)} (3 per query)")
    lines.append(f"- Total queries with references: {len(refA) // REFS_PER_QUERY_PER_SET}")
    lines.append("")
    lines.append("Constraints enforced:")
    lines.append("- Reference case comes from train fold only (no val/holdout as reference)")
    lines.append("- Reference patient != query patient (patient-disjoint)")
    lines.append("- Set A and Set B disjoint per query (6 distinct references)")
    lines.append(f"- Fixed seed: {REFERENCE_SEED}")
    lines.append("- No result-based reference selection")
    lines.append("")
    lines.append("## 8. Output Files")
    lines.append("")
    lines.append("### Audit (reports/phaseY_data_audit/)")
    for name in [
        "image_mask_pairing.csv",
        "case_identity_map.csv",
        "prior_query_exposure.csv",
        "prior_reference_exposure.csv",
        "eligible_fresh_queries.csv",
        "duplicate_hash_audit.csv",
        "data_audit_summary.md",
        "split_summary.json",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("### Manifests (manifests/phaseY/)")
    for name in [
        "policy_train_queries.csv",
        "policy_validation_queries.csv",
        "confirmatory_holdout_queries.csv",
        "reference_pool.csv",
        "reference_set_A.csv",
        "reference_set_B.csv",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## 9. Status")
    lines.append("")
    lines.append("```text")
    lines.append("PHASEY_DATA_AUDIT_COMPLETE")
    lines.append("```")
    lines.append("")
    lines.append("No SAM inference was run. All outputs are manifest-level and read-only over")
    lines.append("the MoNuSAC processed data tree.")
    return "\n".join(lines)


def write_split_summary_json(
    identity, eligible, split, ref_pool, refA, refB
) -> Dict:
    summary = {
        "phase": "phaseY",
        "stage": "data_audit_and_split",
        "directive_section": "5.1-5.3",
        "source_manifest": str(MANIFEST_SRC),
        "source_manifest_sha256": sha256_file(MANIFEST_SRC),
        "split_seed": SPLIT_SEED,
        "reference_seed": REFERENCE_SEED,
        "total_cases": len(identity),
        "train_cases": len([r for r in identity if r["split"] == "train"]),
        "test_cases": len([r for r in identity if r["split"] == "test"]),
        "test_locked": True,
        "phasex_query_cases": PHASEX_QUERY_CASES,
        "phasex_reference_cases": PHASEX_REFERENCE_CASES,
        "phasex_exposed_patients": sorted(
            {r["patient"] for r in identity if r["phasex_exposure"] != "none"}
        ),
        "eligible_fresh_cases": len(eligible),
        "eligible_fresh_patients": len(
            {r["patient"] for r in eligible}
        ),
        "split": {
            "train": {
                "case_count": len(split["train"]),
                "patient_count": len({c["patient"] for c in split["train"]}),
                "target": TARGET_TRAIN,
            },
            "validation": {
                "case_count": len(split["val"]),
                "patient_count": len({c["patient"] for c in split["val"]}),
                "target": TARGET_VAL,
            },
            "holdout": {
                "case_count": len(split["holdout"]),
                "patient_count": len({c["patient"] for c in split["holdout"]}),
                "target": TARGET_HOLDOUT,
            },
        },
        "reference_pool_size": len(ref_pool),
        "reference_set_A_rows": len(refA),
        "reference_set_B_rows": len(refB),
        "refs_per_query_per_set": REFS_PER_QUERY_PER_SET,
        "patient_level_disjoint": True,
        "reference_patient_disjoint_from_query": True,
        "set_A_B_disjoint_per_query": True,
        "status": "PHASEY_DATA_AUDIT_COMPLETE",
    }
    return summary


def write_reference_audit_json(
    split, refA, refB, ref_pool
) -> Dict:
    # Verify disjointness and patient-disjointness
    violations = []
    # Group by query
    a_by_q: Dict[str, List[str]] = defaultdict(list)
    b_by_q: Dict[str, List[str]] = defaultdict(list)
    a_pat_by_q: Dict[str, List[str]] = defaultdict(list)
    b_pat_by_q: Dict[str, List[str]] = defaultdict(list)
    for r in refA:
        a_by_q[r["query_case_id"]].append(r["reference_case_id"])
        a_pat_by_q[r["query_case_id"]].append(r["reference_patient"])
    for r in refB:
        b_by_q[r["query_case_id"]].append(r["reference_case_id"])
        b_pat_by_q[r["query_case_id"]].append(r["reference_patient"])

    all_queries = set(a_by_q.keys()) | set(b_by_q.keys())
    for q in all_queries:
        a_cases = set(a_by_q.get(q, []))
        b_cases = set(b_by_q.get(q, []))
        overlap = a_cases & b_cases
        if overlap:
            violations.append(f"{q}: A/B overlap cases {overlap}")
        # Check reference patient != query patient
        q_row = next(
            (c for fold in split.values() for c in fold if c["case_id"] == q),
            None,
        )
        if q_row:
            q_pat = q_row["patient"]
            for rp in a_pat_by_q.get(q, []) + b_pat_by_q.get(q, []):
                if rp == q_pat:
                    violations.append(f"{q}: reference patient {rp} == query patient")

    summary = {
        "phase": "phaseY",
        "stage": "reference_set_audit",
        "directive_section": "5.3",
        "reference_pool_size": len(ref_pool),
        "reference_set_A_rows": len(refA),
        "reference_set_B_rows": len(refB),
        "queries_with_references": len(all_queries),
        "refs_per_query_per_set": REFS_PER_QUERY_PER_SET,
        "constraints": {
            "references_from_train_only": True,
            "reference_patient_not_query_patient": True,
            "set_A_B_disjoint_per_query": True,
            "fixed_seed": REFERENCE_SEED,
            "no_result_based_selection": True,
        },
        "violations": violations,
        "status": "PHASEY_REFERENCE_AUDIT_PASS" if not violations else "PHASEY_REFERENCE_AUDIT_FAIL",
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    rows = read_manifest()

    # 5.1 audits
    identity = build_case_identity_map(rows)
    pairing = build_image_mask_pairing(rows)
    prior_q = build_prior_query_exposure()
    prior_r = build_prior_reference_exposure()
    duplicates = build_duplicate_hash_audit(pairing)
    eligible, patient_to_cases = build_eligible_fresh_queries(identity)

    # 5.2 split
    split = split_patients_into_folds(patient_to_cases)

    # 5.3 reference sets
    ref_pool, refA, refB = build_reference_sets(split)

    # Write audit CSVs
    write_csv(AUDIT_DIR / "image_mask_pairing.csv", pairing)
    write_csv(AUDIT_DIR / "case_identity_map.csv", identity)
    write_csv(AUDIT_DIR / "prior_query_exposure.csv", prior_q)
    write_csv(AUDIT_DIR / "prior_reference_exposure.csv", prior_r)
    write_csv(AUDIT_DIR / "eligible_fresh_queries.csv", eligible)
    write_csv(AUDIT_DIR / "duplicate_hash_audit.csv", duplicates)

    # Write split CSVs
    write_csv(SPLIT_DIR / "policy_train_queries.csv", split["train"])
    write_csv(SPLIT_DIR / "policy_validation_queries.csv", split["val"])
    write_csv(SPLIT_DIR / "confirmatory_holdout_queries.csv", split["holdout"])
    write_csv(SPLIT_DIR / "reference_pool.csv", ref_pool)
    write_csv(SPLIT_DIR / "reference_set_A.csv", refA)
    write_csv(SPLIT_DIR / "reference_set_B.csv", refB)

    # Write summary
    summary_md = write_summary_md(
        identity, pairing, duplicates, eligible, split, ref_pool, refA, refB
    )
    (AUDIT_DIR / "data_audit_summary.md").write_text(summary_md, encoding="utf-8")

    split_summary = write_split_summary_json(identity, eligible, split, ref_pool, refA, refB)
    (AUDIT_DIR / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    ref_audit = write_reference_audit_json(split, refA, refB, ref_pool)
    (PROJECT_ROOT / "reports" / "phaseY_reference_audit.json").write_text(
        json.dumps(ref_audit, indent=2), encoding="utf-8"
    )

    # Console summary
    print("=" * 60)
    print("Phase Y Data Audit & Split — COMPLETE")
    print("=" * 60)
    print(f"Source manifest: {MANIFEST_SRC}")
    print(f"Total cases: {len(identity)}")
    print(f"Eligible fresh cases: {len(eligible)}")
    print(f"Split: train={len(split['train'])} val={len(split['val'])} holdout={len(split['holdout'])}")
    print(f"Reference pool: {len(ref_pool)}")
    print(f"Reference Set A: {len(refA)} rows")
    print(f"Reference Set B: {len(refB)} rows")
    print(f"Reference audit: {ref_audit['status']}")
    print(f"Audit dir: {AUDIT_DIR}")
    print(f"Split dir: {SPLIT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
