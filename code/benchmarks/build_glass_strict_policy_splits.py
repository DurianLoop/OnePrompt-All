from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    "external/research_data/extracted/GlaS/Warwick QU Dataset (Released 2016_07_08)"
)
DEFAULT_OUT_DIR = "runs/research_direction/prompt_policy_splits/glas_patient_disjoint"
DEFAULT_SOURCE_URL = (
    "https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest-backup/download/"
    "warwick_qu_dataset_released_2016_07_08.zip"
)
DEFAULT_SOURCE_SHA256 = "7849D349DB105751A157D011E9708D724203E85BBAD08FD047DCDFE5B9A572E2"


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_validation_patients(eligible_patients: set[str], count: int, seed: int) -> set[str]:
    if not 0 < count < len(eligible_patients):
        raise ValueError("val_patient_count must leave at least one policy-training patient.")
    ordered = sorted(eligible_patients)
    random.Random(seed).shuffle(ordered)
    return set(ordered[:count])


def build_glass_protocol(
    rows: list[dict[str, str]],
    val_patient_count: int,
    references_per_query: int,
    seed: int,
    source_url: str = DEFAULT_SOURCE_URL,
    source_sha256: str = DEFAULT_SOURCE_SHA256,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    if references_per_query < 1:
        raise ValueError("references_per_query must be positive.")
    source_train = [row for row in rows if row["source_split"] == "train"]
    source_test = [row for row in rows if row["source_split"] in {"testA", "testB"}]
    if not source_train or not source_test:
        raise ValueError("GlaS manifest must contain source train and official test cases.")

    official_test_patients = {row["patient"] for row in source_test}
    eligible_patients = {row["patient"] for row in source_train} - official_test_patients
    validation_patients = choose_validation_patients(eligible_patients, val_patient_count, seed)

    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        if row["source_split"] in {"testA", "testB"}:
            role = "test"
        elif row["patient"] in official_test_patients:
            role = "excluded_overlap"
        elif row["patient"] in validation_patients:
            role = "val"
        else:
            role = "train"
        assignments.append({**row, "role": role})

    train_supports = [row for row in assignments if row["role"] == "train"]
    episodes: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for role in episodes:
        for query in (row for row in assignments if row["role"] == role):
            eligible_supports = [
                support
                for support in train_supports
                if support["case_id"] != query["case_id"] and support["patient"] != query["patient"]
            ]
            if len(eligible_supports) < references_per_query:
                raise ValueError(
                    f"Query {query['case_id']} has only {len(eligible_supports)} patient-disjoint train supports."
                )
            rng = random.Random(f"{seed}:{role}:{query['case_id']}")
            selected = rng.sample(
                sorted(eligible_supports, key=lambda item: item["case_id"]), references_per_query
            )
            for support in selected:
                episodes[role].append({
                    "episode_id": f"glas_{role}__{support['case_id']}__to__{query['case_id']}",
                    "dataset": "GlaS",
                    "domain": "pathology_gland",
                    "role": role,
                    "support_role": "train",
                    "support_case_id": support["case_id"],
                    "support_patient": support["patient"],
                    "support_image_path": support["image_path"],
                    "support_mask_path": support["mask_path"],
                    "query_case_id": query["case_id"],
                    "query_patient": query["patient"],
                    "query_source_split": query["source_split"],
                    "query_image_path": query["image_path"],
                    "query_mask_path": query["mask_path"],
                })

    role_counts = Counter(row["role"] for row in assignments)
    metadata: dict[str, Any] = {
        "protocol": "glas_patient_disjoint_official_test_preserved",
        "source_url": source_url,
        "source_sha256": source_sha256,
        "seed": seed,
        "references_per_query": references_per_query,
        "source_case_counts": {
            "train": len(source_train),
            "testA": sum(row["source_split"] == "testA" for row in source_test),
            "testB": sum(row["source_split"] == "testB" for row in source_test),
        },
        "official_test_case_count": len(source_test),
        "excluded_overlap_case_count": role_counts["excluded_overlap"],
        "role_case_counts": dict(sorted(role_counts.items())),
        "policy_train_patients": sorted({row["patient"] for row in assignments if row["role"] == "train"}),
        "validation_patients": sorted(validation_patients),
        "official_test_patients": sorted(official_test_patients),
        "episode_counts": {role: len(items) for role, items in episodes.items()},
        "leakage_rules": [
            "Official test cases are never used for training, validation selection, or reference sampling.",
            "Any source-train patient present in official test is excluded from policy training and validation.",
            "Each support is a policy-train case from a different patient than its query.",
        ],
    }
    return assignments, episodes, metadata


def read_glass_manifest(source_root: Path) -> list[dict[str, str]]:
    grade_path = source_root / "Grade.csv"
    with grade_path.open(newline="", encoding="utf-8-sig") as handle:
        grades = list(csv.DictReader(handle))
    rows: list[dict[str, str]] = []
    for grade in grades:
        case_id = grade["name"]
        image_path = source_root / f"{case_id}.bmp"
        mask_path = source_root / f"{case_id}_anno.bmp"
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError(f"Missing image/mask pair for GlaS case {case_id}.")
        source_split = case_id.rsplit("_", 1)[0]
        rows.append({
            "dataset": "GlaS",
            "case_id": case_id,
            "source_split": source_split,
            "patient": grade["patient ID"].strip(),
            "grade": grade[" grade (GlaS)"].strip(),
            "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
            "mask_path": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe GlaS prompt-policy episodes.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-patient-count", type=int, default=1)
    parser.add_argument("--references-per-query", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--source-sha256", default=DEFAULT_SOURCE_SHA256)
    args = parser.parse_args()

    rows = read_glass_manifest(resolve(args.source_root))
    assignments, episodes, metadata = build_glass_protocol(
        rows,
        args.val_patient_count,
        args.references_per_query,
        args.seed,
        source_url=args.source_url,
        source_sha256=args.source_sha256,
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "case_assignments.csv", assignments)
    for role, items in episodes.items():
        write_csv(out_dir / f"episodes_{role}.csv", items)
    (out_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
