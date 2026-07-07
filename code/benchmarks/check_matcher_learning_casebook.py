from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def write_case(root: Path, name: str, proposals: list[dict[str, float]], positives: list[int], negatives: list[int]) -> dict[str, str]:
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    feature_json = case_dir / "feature_scores.json"
    labels_json = case_dir / "labels.json"
    feature_json.write_text(json.dumps({"metrics": proposals}, indent=2), encoding="utf-8")
    labels_json.write_text(
        json.dumps({"positive_group_ids": positives, "negative_group_ids": negatives}, indent=2),
        encoding="utf-8",
    )
    return {
        "case_name": name,
        "stage3_feature_json": str(feature_json),
        "labels_json": str(labels_json),
    }


def main() -> int:
    tmp = WORKSPACE_ROOT / ".cache" / "matcher_learning_casebook"
    tmp.mkdir(parents=True, exist_ok=True)
    cases = [
        write_case(
            tmp,
            "case_a",
            [
                {"group_id": 1, "instance_similarity": 0.62, "sam_score": 0.70, "cluster_coverage": 0.9, "prompt_coverage": 1.0, "mask_area_pct": 8.0, "bbox_area_pct": 10.0},
                {"group_id": 2, "instance_similarity": 0.58, "sam_score": 0.95, "cluster_coverage": 0.4, "prompt_coverage": 0.5, "mask_area_pct": 70.0, "bbox_area_pct": 82.0},
                {"group_id": 3, "instance_similarity": 0.33, "sam_score": 0.72, "cluster_coverage": 0.2, "prompt_coverage": 0.5, "mask_area_pct": 4.0, "bbox_area_pct": 6.0},
            ],
            positives=[1],
            negatives=[2, 3],
        ),
        write_case(
            tmp,
            "case_b",
            [
                {"group_id": 4, "instance_similarity": 0.64, "sam_score": 0.73, "cluster_coverage": 0.8, "prompt_coverage": 1.0, "mask_area_pct": 11.0, "bbox_area_pct": 15.0},
                {"group_id": 5, "instance_similarity": 0.59, "sam_score": 0.98, "cluster_coverage": 0.5, "prompt_coverage": 0.5, "mask_area_pct": 68.0, "bbox_area_pct": 76.0},
                {"group_id": 6, "instance_similarity": 0.41, "sam_score": 0.69, "cluster_coverage": 0.3, "prompt_coverage": 0.5, "mask_area_pct": 5.0, "bbox_area_pct": 8.0},
            ],
            positives=[4],
            negatives=[5, 6],
        ),
    ]
    cases_json = tmp / "cases.json"
    out_dir = tmp / "out"
    cases_json.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_learning_casebook.py",
        "--cases-json",
        str(cases_json),
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    payload = json.loads((out_dir / "casebook_policy.json").read_text(encoding="utf-8"))
    if payload["labeled_count"] != 6:
        print(json.dumps(payload, indent=2))
        return 1
    if payload["best_policy"]["feature"] != "feedback_learning_score":
        print(json.dumps(payload, indent=2))
        return 1
    if payload["best_policy"]["f1"] != 1.0:
        print(json.dumps(payload, indent=2))
        return 1
    print("Matcher learning casebook checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
