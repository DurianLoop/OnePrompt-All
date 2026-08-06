from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_one_click_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_medical_one_click.py",
        "--out-dir",
        str(out_dir),
        "--idrid-cases",
        "12",
        "--monuseg-cases",
        "8",
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    plan_path = out_dir / "one_click_plan.json"
    if not plan_path.exists():
        print("Missing one_click_plan.json")
        return 1
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("requested_case_count") != 20:
        print(f"Expected 20 cases, got {payload.get('requested_case_count')}")
        return 1
    step_names = [step["name"] for step in payload.get("steps", [])]
    required = {
        "build_manifest",
        "run_stage1",
        "run_stage2_sampling",
        "run_sam_proposals",
        "run_stage3_filtering",
        "run_eval",
        "run_casebook",
        "build_review_gallery",
    }
    if not required.issubset(set(step_names)):
        print(f"Missing steps: {sorted(required - set(step_names))}")
        return 1
    print("Matcher medical one-click dry-run checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
