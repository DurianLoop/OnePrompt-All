from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    cases_json = WORKSPACE_ROOT / "runs" / "research_direction" / "matcher_learning_casebook" / "teacher_direction_manual_cases.json"
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_feedback_loop_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_feedback_loop.py",
        "--cases-json",
        str(cases_json),
        "--max-feedback-rounds",
        "3",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    payload = json.loads((out_dir / "feedback_loop_results.json").read_text(encoding="utf-8"))
    if payload["fold_count"] != 3:
        print(json.dumps(payload, indent=2))
        return 1
    if len(payload["curves"]) < 12:
        print(json.dumps(payload, indent=2))
        return 1
    if not {"round", "mean_f1", "mean_labeled_feedback"}.issubset(payload["round_summary"][0]):
        print(json.dumps(payload["round_summary"], indent=2))
        return 1
    print("Matcher feedback loop checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
