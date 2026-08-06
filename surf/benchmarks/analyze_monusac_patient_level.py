from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_prompt_policy_results import (
    aggregate_paired_by_query,
    pair_policy_rows,
    paired_policy_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def aggregate_query_rows_by_patient(
    query_rows: list[dict[str, Any]],
    patient_by_case: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        case_id = str(row["query_case_id"])
        if case_id not in patient_by_case:
            raise KeyError(f"Missing patient mapping for query patch {case_id}")
        grouped[patient_by_case[case_id]].append(row)
    result: list[dict[str, Any]] = []
    for patient in sorted(grouped):
        rows = grouped[patient]
        fixed = float(np.mean([float(row["fixed_dice"]) for row in rows]))
        learned = float(np.mean([float(row["learned_dice"]) for row in rows]))
        result.append({
            "patient": patient,
            "query_case_id": patient,
            "patch_count": len(rows),
            "fixed_dice": fixed,
            "learned_dice": learned,
            "delta_dice": learned - fixed,
        })
    return result


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
    parser = argparse.ArgumentParser(description="Report MoNuSAC prompt-policy effects at patch and patient levels.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    assignments = read_csv(resolve(args.assignments))
    patient_by_case = {
        row["case_id"]: row["patient"]
        for row in assignments
        if row["role"] == "test"
    }
    episode_pairs = pair_policy_rows(
        read_csv(resolve(args.fixed)),
        read_csv(resolve(args.learned)),
    )
    query_pairs = aggregate_paired_by_query(episode_pairs)
    patient_pairs = aggregate_query_rows_by_patient(query_pairs, patient_by_case)
    patch_summary = paired_policy_summary(
        query_pairs,
        bootstrap_samples=args.bootstrap_samples,
    )
    patch_summary.update({
        "episode_count": len(episode_pairs),
        "inference_unit": "query_patch_after_reference_mean",
    })
    patient_summary = paired_policy_summary(
        patient_pairs,
        bootstrap_samples=args.bootstrap_samples,
    )
    patient_summary.update({
        "patch_count": len(query_pairs),
        "episode_count": len(episode_pairs),
        "inference_unit": "patient_after_patch_and_reference_mean",
    })

    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "paired_episode_deltas.csv", episode_pairs)
    write_csv(out_dir / "paired_patch_deltas.csv", query_pairs)
    write_csv(out_dir / "paired_patient_deltas.csv", patient_pairs)
    (out_dir / "patch_summary.json").write_text(json.dumps(patch_summary, indent=2), encoding="utf-8")
    (out_dir / "patient_summary.json").write_text(json.dumps(patient_summary, indent=2), encoding="utf-8")
    print(json.dumps({"patch": patch_summary, "patient": patient_summary}, indent=2))


if __name__ == "__main__":
    main()
