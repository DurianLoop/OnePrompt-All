from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_casebook_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/build_matcher_medical_casebook.py",
        "--manifest",
        "runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv",
        "--run-dir",
        "runs/research_direction/matcher_medical_pilot/run",
        "--out-dir",
        str(out_dir),
        "--positive-dice",
        "0.30",
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    payload_path = out_dir / "medical_casebook_policy.json"
    rows_path = out_dir / "medical_casebook_rows.csv"
    summary_path = out_dir / "summary.md"
    if not payload_path.exists() or not rows_path.exists() or not summary_path.exists():
        print("Missing medical casebook outputs.")
        return 1
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {
        "case_count",
        "labeled_count",
        "positive_count",
        "negative_count",
        "best_policy",
        "baseline_stage3",
        "oracle_best_per_case",
        "learned_best_per_case",
    }
    if not required.issubset(payload):
        print(json.dumps(payload, indent=2))
        return 1
    if payload["case_count"] != 5 or payload["labeled_count"] != 60:
        print(f"Unexpected counts: cases={payload.get('case_count')} rows={payload.get('labeled_count')}")
        return 1
    if payload["positive_count"] <= 0 or payload["negative_count"] <= 0:
        print("Expected both positive and negative labels.")
        return 1
    print("Matcher medical casebook checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
