from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def audit_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    tissues: Counter[str] = Counter()
    instances = 0
    patients: set[str] = set()
    for row in rows:
        masks = list(row["instances"])
        categories = list(row["categories"])
        if len(masks) != len(categories):
            raise ValueError(f"{split}: instance/category length mismatch for {row['patient']}")
        patients.add(str(row["patient"]))
        instances += len(masks)
        category_counts.update(str(int(value)) for value in categories)
        tissues[str(int(row["tissue"]))] += 1
    return {
        "split": split,
        "patch_count": len(rows),
        "patient_count": len(patients),
        "instance_count": instances,
        "category_counts": dict(sorted(category_counts.items())),
        "tissue_patch_counts": dict(sorted(tissues.items())),
    }


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return (ROOT / path).resolve() if not path.is_absolute() else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local MoNuSAC parquet data for policy-learning expansion.")
    parser.add_argument("--data-dir", default="code/datasets/MoNuSAC/hf_raw/data")
    parser.add_argument("--out-dir", default="runs/research_direction/dataset_audit/monusac")
    args = parser.parse_args()

    data_dir = resolve(args.data_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    for split in ("train", "test"):
        parquet = data_dir / f"{split}-00000-of-00001.parquet"
        frame = pd.read_parquet(parquet)
        rows = frame.to_dict(orient="records")
        split_summaries.append(audit_rows(rows, split))
        for index, row in enumerate(rows):
            split_rows.append(
                {
                    "split": split,
                    "case_id": f"{split}_{index:03d}",
                    "patient": str(row["patient"]),
                    "tissue": int(row["tissue"]),
                    "instance_count": len(row["instances"]),
                    "category_counts": json.dumps(dict(Counter(str(int(v)) for v in row["categories"])), sort_keys=True),
                }
            )
    with (out_dir / "monusac_patch_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    (out_dir / "monusac_audit.json").write_text(json.dumps(split_summaries, indent=2), encoding="utf-8")
    print(json.dumps(split_summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
