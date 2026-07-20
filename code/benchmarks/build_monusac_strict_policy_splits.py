from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def sample_paths(row: dict[str, str]) -> tuple[str, str]:
    base = Path("code/datasets/MoNuSAC/processed") / row["split"]
    case_id = row["case_id"]
    return (
        str(base / "images" / f"{case_id}.png").replace("\\", "/"),
        str(base / "masks" / f"{case_id}.png").replace("\\", "/"),
    )


def build_episodes(
    assignments: list[dict[str, str]],
    query_role: str,
    references_per_query: int,
    seed: int,
) -> list[dict[str, str]]:
    supports = [row for row in assignments if row["role"] == "train"]
    queries = [row for row in assignments if row["role"] == query_role]
    episodes: list[dict[str, str]] = []
    for query in sorted(queries, key=lambda row: row["case_id"]):
        eligible = [row for row in supports if row["patient"] != query["patient"]]
        if len(eligible) < references_per_query:
            raise ValueError(
                f"Query {query['case_id']} has only {len(eligible)} patient-disjoint supports."
            )
        rng = random.Random(f"{seed}:{query_role}:{query['case_id']}")
        selected = rng.sample(sorted(eligible, key=lambda row: row["case_id"]), references_per_query)
        query_image, query_mask = sample_paths(query)
        for support in selected:
            support_image, support_mask = sample_paths(support)
            episodes.append({
                "pilot_id": (
                    f"monusac_strict_{query_role}__{support['case_id']}__to__{query['case_id']}"
                ),
                "dataset": "MoNuSAC",
                "domain": "pathology_nuclei_all",
                "role": query_role,
                "support_role": "train",
                "support_case_id": support["case_id"],
                "support_patient": support["patient"],
                "support_split": support["split"],
                "support_image_path": support_image,
                "support_mask_path": support_mask,
                "query_case_id": query["case_id"],
                "query_patient": query["patient"],
                "query_split": query_role,
                "query_source_split": query["split"],
                "query_image_path": query_image,
                "query_mask_path": query_mask,
            })
    return episodes


def build_strict_protocol(
    rows: list[dict[str, str]],
    val_patient_count: int,
    references_per_query: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    train_patients = {row["patient"] for row in rows if row["split"] == "train"}
    test_patients = {row["patient"] for row in rows if row["split"] == "test"}
    overlap_patients = train_patients & test_patients
    eligible_patients = sorted(train_patients - overlap_patients)
    if not 0 < val_patient_count < len(eligible_patients):
        raise ValueError("val_patient_count must leave at least one train patient.")
    rng = random.Random(seed)
    shuffled = eligible_patients.copy()
    rng.shuffle(shuffled)
    val_patients = set(shuffled[:val_patient_count])
    policy_train_patients = set(shuffled[val_patient_count:])

    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: (item["split"], item["case_id"])):
        if row["split"] == "test":
            role = "test"
        elif row["patient"] in overlap_patients:
            role = "excluded_overlap"
        elif row["patient"] in val_patients:
            role = "val"
        elif row["patient"] in policy_train_patients:
            role = "train"
        else:
            raise AssertionError(f"Unassigned sample: {row}")
        assignments.append({**row, "role": role})

    protocol = {"case_assignments": assignments}
    for role in ("train", "val", "test"):
        protocol[f"episodes_{role}"] = build_episodes(
            assignments,
            role,
            references_per_query=references_per_query,
            seed=seed,
        )
    role_counts = Counter(row["role"] for row in assignments)
    metadata: dict[str, Any] = {
        "protocol": "monusac_patient_disjoint_policy_split_official_test_preserved",
        "seed": seed,
        "references_per_query": references_per_query,
        "source_patch_count": len(rows),
        "source_train_patch_count": sum(row["split"] == "train" for row in rows),
        "official_test_patch_count": sum(row["split"] == "test" for row in rows),
        "source_train_patient_count": len(train_patients),
        "official_test_patient_count": len(test_patients),
        "overlap_patient_count": len(overlap_patients),
        "overlap_patients": sorted(overlap_patients),
        "policy_train_patient_count": len(policy_train_patients),
        "validation_patient_count": len(val_patients),
        "role_patch_counts": dict(sorted(role_counts.items())),
        "episode_counts": {
            role: len(protocol[f"episodes_{role}"])
            for role in ("train", "val", "test")
        },
        "leakage_rules": [
            "All 101 official test patches are preserved.",
            "Any patient appearing in official test is excluded from the policy-training source split.",
            "Train and validation roles are assigned at patient level.",
            "Every support comes from the policy-train role and a different patient than its query.",
        ],
    }
    return protocol, metadata


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


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patient-disjoint MoNuSAC prompt-policy protocol.")
    parser.add_argument(
        "--manifest",
        default="code/datasets/MoNuSAC/processed/monusac_processed_manifest.csv",
    )
    parser.add_argument("--val-patient-count", type=int, default=9)
    parser.add_argument("--references-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--out-dir",
        default="runs/research_direction/prompt_policy_splits/monusac_strict_patient_disjoint",
    )
    args = parser.parse_args()

    protocol, metadata = build_strict_protocol(
        read_csv(resolve(args.manifest)),
        val_patient_count=args.val_patient_count,
        references_per_query=args.references_per_query,
        seed=args.seed,
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "case_assignments.csv", protocol["case_assignments"])
    for role in ("train", "val", "test"):
        write_csv(out_dir / f"episodes_{role}.csv", protocol[f"episodes_{role}"])
    (out_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
