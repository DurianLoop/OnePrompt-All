from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    tmp = WORKSPACE_ROOT / ".cache" / "matcher_feedback_learning_policy_cli"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        feature_json = tmp / "feature_scores.json"
        labels_json = tmp / "labels.json"
        out_dir = tmp / "policy"
        out_dir.mkdir(parents=True, exist_ok=True)
        feature_json.write_text(
            json.dumps(
                {
                    "proposals": [
                        {
                            "group_id": 1,
                            "instance_similarity": 0.88,
                            "sam_score": 0.96,
                            "cluster_coverage": 0.9,
                            "prompt_coverage": 1.0,
                            "mask_area_pct": 11.0,
                            "bbox_area_pct": 14.0,
                        },
                        {
                            "group_id": 2,
                            "instance_similarity": 0.80,
                            "sam_score": 0.91,
                            "cluster_coverage": 0.7,
                            "prompt_coverage": 1.0,
                            "mask_area_pct": 18.0,
                            "bbox_area_pct": 24.0,
                        },
                        {
                            "group_id": 3,
                            "instance_similarity": 0.43,
                            "sam_score": 0.89,
                            "cluster_coverage": 0.4,
                            "prompt_coverage": 0.5,
                            "mask_area_pct": 60.0,
                            "bbox_area_pct": 80.0,
                        },
                        {
                            "group_id": 4,
                            "instance_similarity": 0.31,
                            "sam_score": 0.75,
                            "cluster_coverage": 0.2,
                            "prompt_coverage": 0.5,
                            "mask_area_pct": 4.0,
                            "bbox_area_pct": 6.0,
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        labels_json.write_text(
            json.dumps({"positive_group_ids": [1, 2], "negative_group_ids": [3, 4]}, indent=2),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-B",
            "code/benchmarks/run_matcher_feedback_learning_policy.py",
            "--stage3-feature-json",
            str(feature_json),
            "--labels-json",
            str(labels_json),
            "--out-dir",
            str(out_dir),
        ]
        completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)
            return completed.returncode
        policy = json.loads((out_dir / "matcher_learning_policy.json").read_text(encoding="utf-8"))
        if policy["selected_group_ids"] != [1, 2]:
            print(json.dumps(policy, indent=2))
            return 1
        if policy["instance_similarity_policy"]["f1"] != 1.0:
            print(json.dumps(policy, indent=2))
            return 1
    except Exception as exc:
        print(f"CLI check failed: {exc}")
        return 1
    print("Matcher feedback learning policy CLI checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
