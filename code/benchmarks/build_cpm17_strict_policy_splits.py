from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = "external/research_data/raw/CPM17"
DEFAULT_OUT_DIR = "runs/research_direction/prompt_policy_splits/cpm17_official_holdout"
DEFAULT_SOURCE_URL = "https://github.com/nauyan/NucleiSegmentation"


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def validate_complete_pairs(
    rows: list[dict[str, str]], image_ids: set[str], label_ids: set[str]
) -> None:
    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    if missing_labels:
        raise ValueError(f"missing label for image {missing_labels[0]}")
    if missing_images:
        raise ValueError(f"missing image for label {missing_images[0]}")
    for row in rows:
        case_id = row["case_id"]
        if case_id not in image_ids:
            raise ValueError(f"missing image for case {case_id}")
        if case_id not in label_ids:
            raise ValueError(f"missing label for image {case_id}")


def build_cpm17_protocol(
    rows: list[dict[str, str]], val_count: int, references_per_query: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    if references_per_query < 1:
        raise ValueError("references_per_query must be positive.")
    source_train = [row for row in rows if row["source_split"] == "train"]
    source_test = [row for row in rows if row["source_split"] == "test"]
    if not source_train or not source_test:
        raise ValueError("CPM-17 manifest must contain source train and official test cases.")
    if not 0 < val_count < len(source_train):
        raise ValueError("val_count must be positive and leave at least one policy-training case.")
    ordered_train = sorted(source_train, key=lambda item: item["case_id"])
    rng = random.Random(seed)
    validation_ids = set(rng.sample([row["case_id"] for row in ordered_train], val_count))

    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: (item["source_split"], item["case_id"])):
        role = "test" if row["source_split"] == "test" else "val" if row["case_id"] in validation_ids else "train"
        assignments.append({**row, "role": role})
    supports = [row for row in assignments if row["role"] == "train"]

    episodes: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for role in episodes:
        for query in (row for row in assignments if row["role"] == role):
            eligible = [support for support in supports if support["case_id"] != query["case_id"]]
            if len(eligible) < references_per_query:
                raise ValueError(
                    f"Query {query['case_id']} has only {len(eligible)} non-self policy-train supports."
                )
            case_rng = random.Random(f"{seed}:{role}:{query['case_id']}")
            selected = case_rng.sample(sorted(eligible, key=lambda item: item["case_id"]), references_per_query)
            for support in selected:
                episodes[role].append({
                    "episode_id": f"cpm17_{role}__{support['case_id']}__to__{query['case_id']}",
                    "dataset": "CPM-17",
                    "domain": "pathology_nuclei",
                    "role": role,
                    "support_role": "train",
                    "support_source_split": "train",
                    "support_case_id": support["case_id"],
                    "support_image_path": support["image_path"],
                    "support_mat_path": support["mat_path"],
                    "query_case_id": query["case_id"],
                    "query_source_split": query["source_split"],
                    "query_image_path": query["image_path"],
                    "query_mat_path": query["mat_path"],
                })

    role_counts = Counter(row["role"] for row in assignments)
    metadata: dict[str, Any] = {
        "protocol": "cpm17_case_disjoint_official_test_preserved",
        "source_url": DEFAULT_SOURCE_URL,
        "seed": seed,
        "references_per_query": references_per_query,
        "source_case_counts": {"train": len(source_train), "test": len(source_test)},
        "official_test_case_count": len(source_test),
        "role_case_counts": dict(sorted(role_counts.items())),
        "patient_metadata_available": False,
        "episode_counts": {role: len(items) for role, items in episodes.items()},
        "leakage_rules": [
            "Official test cases are never used for training, validation selection, or reference sampling.",
            "Validation and policy-train cases are disjoint by case ID.",
            "Each support is a policy-train case different from the query case.",
        ],
    }
    return assignments, episodes, metadata


def read_cpm17_manifest(source_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        image_dir = source_root / split / "Images"
        label_dir = source_root / split / "Labels"
        image_ids = {path.stem for path in image_dir.glob("*.png")}
        label_ids = {path.stem for path in label_dir.glob("*.mat")}
        validate_complete_pairs([], image_ids, label_ids)
        if len(image_ids) != 32:
            raise ValueError(f"Expected 32 CPM-17 {split} image/label pairs, found {len(image_ids)}.")
        for case_id in sorted(image_ids):
            image_path = image_dir / f"{case_id}.png"
            mat_path = label_dir / f"{case_id}.mat"
            rows.append({
                "dataset": "CPM-17",
                "case_id": case_id,
                "source_split": split,
                "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "mat_path": str(mat_path.relative_to(ROOT)).replace("\\", "/"),
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
    parser = argparse.ArgumentParser(description="Build strict CPM-17 prompt-policy episodes.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--references-per-query", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    rows = read_cpm17_manifest(resolve(args.source_root))
    assignments, episodes, metadata = build_cpm17_protocol(
        rows, args.val_count, args.references_per_query, args.seed
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
