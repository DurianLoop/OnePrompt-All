from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_pilot_runner_check"
    manifest = out_dir / "manifest" / "medical_matcher_pilot_manifest.csv"
    build_manifest = [
        sys.executable,
        "-B",
        "code/benchmarks/build_matcher_medical_pilot_manifest.py",
        "--out-dir",
        str(manifest.parent),
        "--idrid-cases",
        "3",
        "--monuseg-cases",
        "2",
    ]
    completed = subprocess.run(build_manifest, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_medical_pilot.py",
        "--manifest",
        str(manifest),
        "--out-dir",
        str(out_dir / "runner"),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    plan_path = out_dir / "runner" / "medical_pilot_run_plan.json"
    if not plan_path.exists():
        print("Missing run plan.")
        return 1
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("case_count") != 5:
        print(f"Expected 5 planned cases, got {payload.get('case_count')}")
        return 1
    for item in payload.get("cases", []):
        if not item.get("stage1_command") or not item.get("feedback_adapter_command") or not item.get("stage2_sampler_command"):
            print(json.dumps(item, indent=2))
            return 1
    print("Matcher medical pilot runner checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
