from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "runs/research_direction/dataset_audit/monuseg_official_manifest.csv"
DEFAULT_OUT_DIR = "runs/research_direction/prompt_policy_splits/monuseg_strict_official"

ASSIGNMENT_FIELDS = [
    "dataset",
    "sample_id",
    "source_split",
    "role",
    "image_path",
    "mask_path",
    "mask_area_ratio",
    "nuclei_region_count",
]
EPISODE_FIELDS = [
    "pilot_id",
    "dataset",
    "domain",
    "role",
    "support_case_id",
    "support_split",
    "support_image_path",
    "support_mask_path",
    "query_case_id",
    "query_split",
    "query_source_split",
    "query_image_path",
    "query_mask_path",
]


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Expected numeric '{field}' for case {row.get('sample_id', '<unknown>')}.") from error


def select_validation_ids(train_rows: list[dict[str, str]], val_count: int) -> set[str]:
    if not 0 < val_count < len(train_rows):
        raise ValueError("val_count must be positive and smaller than the number of source training cases.")
    ordered = sorted(
        train_rows,
        key=lambda row: (as_float(row, "mask_area_ratio"), as_float(row, "nuclei_region_count"), row["sample_id"]),
    )
    indices = {
        round(position * (len(ordered) - 1) / (val_count - 1)) if val_count > 1 else len(ordered) // 2
        for position in range(val_count)
    }
    if len(indices) != val_count:
        raise ValueError("val_count is too large for deterministic stratified assignment.")
    return {ordered[index]["sample_id"] for index in indices}


def choose_training_reference(train_rows: list[dict[str, str]]) -> dict[str, str]:
    if not train_rows:
        raise ValueError("At least one training case is required to choose a reference.")
    target_density = median(as_float(row, "mask_area_ratio") for row in train_rows)
    return min(
        train_rows,
        key=lambda row: (abs(as_float(row, "mask_area_ratio") - target_density), row["sample_id"]),
    )


def choose_stratified_references(
    train_rows: list[dict[str, str]],
    count: int,
    exclude_id: str,
) -> list[dict[str, str]]:
    candidates = [row for row in train_rows if row["sample_id"] != exclude_id]
    if not 0 < count <= len(candidates):
        raise ValueError("train_reference_count must be positive and leave enough non-query training cases.")
    ordered = sorted(
        candidates,
        key=lambda row: (as_float(row, "mask_area_ratio"), as_float(row, "nuclei_region_count"), row["sample_id"]),
    )
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(position * (len(ordered) - 1) / (count - 1)) for position in range(count)]
    return [ordered[index] for index in indices]


def build_protocol(
    rows: list[dict[str, str]],
    val_count: int,
    train_reference_count: int = 1,
    evaluation_reference_count: int = 1,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    source_train = [row for row in rows if row.get("split", "").lower() == "train"]
    source_test = [row for row in rows if row.get("split", "").lower() == "test"]
    if not source_train or not source_test:
        raise ValueError("The manifest must contain both source train and source test cases.")

    validation_ids = select_validation_ids(source_train, val_count)
    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: (item.get("split", ""), item["sample_id"])):
        source_split = row["split"].lower()
        role = "test" if source_split == "test" else "val" if row["sample_id"] in validation_ids else "train"
        assignments.append({
            "dataset": row.get("dataset", "MoNuSeg"),
            "sample_id": row["sample_id"],
            "source_split": source_split,
            "role": role,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "mask_area_ratio": row["mask_area_ratio"],
            "nuclei_region_count": row["nuclei_region_count"],
        })

    train_cases = [row for row in assignments if row["role"] == "train"]
    support = choose_training_reference(train_cases)
    episodes: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for role in episodes:
        for query in (row for row in assignments if row["role"] == role):
            if role == "train" and train_reference_count == 1 and query["sample_id"] == support["sample_id"]:
                continue
            if role == "train" and train_reference_count > 1:
                references = choose_stratified_references(train_cases, train_reference_count, query["sample_id"])
            elif role in {"val", "test"} and evaluation_reference_count > 1:
                references = choose_stratified_references(
                    train_cases,
                    evaluation_reference_count,
                    query["sample_id"],
                )
            else:
                references = [support]
            for reference in references:
                episodes[role].append({
                    "pilot_id": f"monuseg_strict_{role}__{reference['sample_id']}__to__{query['sample_id']}",
                    "dataset": reference["dataset"],
                    "domain": "pathology_nuclei_all",
                    "role": role,
                    "support_case_id": reference["sample_id"],
                    "support_split": "train",
                    "support_image_path": reference["image_path"],
                    "support_mask_path": reference["mask_path"],
                    "query_case_id": query["sample_id"],
                    "query_split": role,
                    "query_source_split": query["source_split"],
                    "query_image_path": query["image_path"],
                    "query_mask_path": query["mask_path"],
                })

    metadata: dict[str, Any] = {
        "protocol": "observed_monuseg_37_train_14_test_strict_holdout",
        "source_case_counts": {"train": len(source_train), "test": len(source_test)},
        "case_counts": {role: sum(row["role"] == role for row in assignments) for role in episodes},
        "episode_counts": {role: len(items) for role, items in episodes.items()},
        "reference_case_id": support["sample_id"],
        "reference_source_split": "train",
        "train_reference_count_per_query": train_reference_count,
        "evaluation_reference_count_per_query": evaluation_reference_count,
        "validation_assignment": "area_stratified_even_spacing",
        "val_count": val_count,
        "leakage_rules": [
            "Each query case is assigned to one role only.",
            "The reference is selected from the training role only.",
            "Held-out test cases are not candidates for reference or validation selection.",
        ],
    }
    return assignments, episodes, metadata


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict case-level MoNuSeg train/val/test reference-query episodes.")
    parser.add_argument("--official-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--train-reference-count", type=int, default=1)
    parser.add_argument("--evaluation-reference-count", type=int, default=1)
    args = parser.parse_args()

    assignments, episodes, metadata = build_protocol(
        read_csv(resolve_path(args.official_manifest)),
        args.val_count,
        train_reference_count=args.train_reference_count,
        evaluation_reference_count=args.evaluation_reference_count,
    )
    out_dir = resolve_path(args.out_dir)
    write_csv(out_dir / "case_assignments.csv", assignments, ASSIGNMENT_FIELDS)
    for role, role_episodes in episodes.items():
        write_csv(out_dir / f"episodes_{role}.csv", role_episodes, EPISODE_FIELDS)
    (out_dir / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"Wrote strict MoNuSeg protocol to {out_dir}")


if __name__ == "__main__":
    main()
