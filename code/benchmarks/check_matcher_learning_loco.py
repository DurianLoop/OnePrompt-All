from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    cases_json = WORKSPACE_ROOT / "runs" / "research_direction" / "matcher_learning_casebook" / "teacher_direction_manual_cases.json"
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_learning_loco_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_learning_loco.py",
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
    payload = json.loads((out_dir / "loco_results.json").read_text(encoding="utf-8"))
    expected_models = {"feedback_threshold", "instance_threshold", "logistic_regression", "decision_stump"}
    if set(payload["macro_f1_by_model"]) != expected_models:
        print(json.dumps(payload["macro_f1_by_model"], indent=2))
        return 1
    if payload["fold_count"] != 3:
        print(json.dumps(payload, indent=2))
        return 1
    if payload["row_count"] < 50:
        print(json.dumps(payload, indent=2))
        return 1
    print("Matcher learning LOCO checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
