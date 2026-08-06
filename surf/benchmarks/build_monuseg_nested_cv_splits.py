from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_monuseg_strict_policy_splits import (
    EPISODE_FIELDS,
    choose_stratified_references,
    choose_training_reference,
    select_validation_ids,
)


ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_FIELDS = [
    "dataset", "sample_id", "source_split", "outer_fold", "role",
    "image_path", "mask_path", "mask_area_ratio", "nuclei_region_count",
]


def _ordered_source_train(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row.get("split", "").lower() == "train"),
        key=lambda row: (float(row["mask_area_ratio"]), float(row["nuclei_region_count"]), row["sample_id"]),
    )


def _episode(
    fold_index: int,
    role: str,
    reference: dict[str, str],
    query: dict[str, str],
) -> dict[str, str]:
    return {
        "pilot_id": f"monuseg_nested_f{fold_index}_{role}__{reference['sample_id']}__to__{query['sample_id']}",
        "dataset": reference.get("dataset", "MoNuSeg"),
        "domain": "pathology_nuclei_all",
        "role": role,
        "support_case_id": reference["sample_id"],
        "support_split": "train",
        "support_image_path": reference["image_path"],
        "support_mask_path": reference["mask_path"],
        "query_case_id": query["sample_id"],
        "query_split": role,
        "query_source_split": "train",
        "query_image_path": query["image_path"],
        "query_mask_path": query["mask_path"],
    }


def build_nested_cv_protocol(
    rows: list[dict[str, str]],
    fold_count: int = 5,
    inner_val_count: int = 6,
    train_reference_count: int = 1,
    evaluation_reference_count: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_train = _ordered_source_train(rows)
    official_test = [row for row in rows if row.get("split", "").lower() == "test"]
    if fold_count < 2 or fold_count > len(source_train):
        raise ValueError("fold_count must be between 2 and the number of source-train cases.")

    fold_ids: list[set[str]] = [set() for _ in range(fold_count)]
    for index, row in enumerate(source_train):
        fold_ids[index % fold_count].add(row["sample_id"])

    folds: list[dict[str, Any]] = []
    for fold_index, test_ids in enumerate(fold_ids):
        development = [row for row in source_train if row["sample_id"] not in test_ids]
        validation_ids = select_validation_ids(development, inner_val_count)
        assignments: list[dict[str, str]] = []
        for row in source_train:
            role = "outer_test" if row["sample_id"] in test_ids else "val" if row["sample_id"] in validation_ids else "train"
            assignments.append({
                "dataset": row.get("dataset", "MoNuSeg"),
                "sample_id": row["sample_id"],
                "source_split": "train",
                "outer_fold": str(fold_index),
                "role": role,
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "mask_area_ratio": row["mask_area_ratio"],
                "nuclei_region_count": row["nuclei_region_count"],
            })

        train_cases = [row for row in assignments if row["role"] == "train"]
        support = choose_training_reference(train_cases)
        episodes: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
        role_map = {"train": "train", "val": "val", "test": "outer_test"}
        for episode_role, assignment_role in role_map.items():
            for query in (row for row in assignments if row["role"] == assignment_role):
                if episode_role == "train" and train_reference_count == 1:
                    references = [] if query["sample_id"] == support["sample_id"] else [support]
                elif episode_role == "train":
                    references = choose_stratified_references(train_cases, train_reference_count, query["sample_id"])
                elif evaluation_reference_count == 1:
                    references = [support]
                else:
                    references = choose_stratified_references(train_cases, evaluation_reference_count, query["sample_id"])
                episodes[episode_role].extend(
                    _episode(fold_index, episode_role, reference, query)
                    for reference in references
                )

        folds.append({
            "fold_index": fold_index,
            "assignments": assignments,
            "episodes": episodes,
            "case_counts": {
                "train": sum(row["role"] == "train" for row in assignments),
                "val": sum(row["role"] == "val" for row in assignments),
                "test": sum(row["role"] == "outer_test" for row in assignments),
            },
            "episode_counts": {role: len(items) for role, items in episodes.items()},
        })

    metadata = {
        "protocol": "monuseg_source_train_nested_cv_official_test_reserved",
        "fold_count": fold_count,
        "source_train_cases": len(source_train),
        "official_test_cases_reserved": len(official_test),
        "inner_val_count": inner_val_count,
        "train_reference_count_per_query": train_reference_count,
        "evaluation_reference_count_per_query": evaluation_reference_count,
        "outer_assignment": "area_density_ordered_round_robin",
        "inner_assignment": "area_stratified_even_spacing",
        "leakage_rules": [
            "The official source-test set is excluded from all nested-CV folds.",
            "Every source-train case is an outer-test case exactly once.",
            "Inner validation and every support reference come from the current outer development split only.",
            "Support references come from the current inner-training role and never equal the query case.",
        ],
        "folds": [
            {"fold_index": fold["fold_index"], "case_counts": fold["case_counts"], "episode_counts": fold["episode_counts"]}
            for fold in folds
        ],
    }
    return folds, metadata


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free nested CV manifests on MoNuSeg source-train cases.")
    parser.add_argument("--official-manifest", default="runs/research_direction/dataset_audit/monuseg_official_manifest.csv")
    parser.add_argument("--out-dir", default="runs/research_direction/prompt_policy_splits/monuseg_nested_cv5")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--inner-val-count", type=int, default=6)
    parser.add_argument("--train-reference-count", type=int, default=5)
    parser.add_argument("--evaluation-reference-count", type=int, default=5)
    args = parser.parse_args()

    manifest = Path(args.official_manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    folds, metadata = build_nested_cv_protocol(
        read_csv(manifest),
        fold_count=args.fold_count,
        inner_val_count=args.inner_val_count,
        train_reference_count=args.train_reference_count,
        evaluation_reference_count=args.evaluation_reference_count,
    )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    for fold in folds:
        fold_dir = out_dir / f"fold_{fold['fold_index']}"
        write_csv(fold_dir / "case_assignments.csv", fold["assignments"], ASSIGNMENT_FIELDS)
        for role, episodes in fold["episodes"].items():
            write_csv(fold_dir / f"episodes_{role}.csv", episodes, EPISODE_FIELDS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
