from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_pilot_eval_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/evaluate_matcher_medical_pilot.py",
        "--manifest",
        "runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv",
        "--run-dir",
        "runs/research_direction/matcher_medical_pilot/run",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    payload_path = out_dir / "medical_pilot_eval.json"
    if not payload_path.exists():
        print("Missing eval json.")
        return 1
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("case_count") != 5:
        print(f"Expected 5 cases, got {payload.get('case_count')}")
        return 1
    required = {"proposal_count", "retained_count", "best_dice_mean", "best_iou_mean", "cases"}
    if not required.issubset(payload):
        print(json.dumps(payload, indent=2))
        return 1
    print("Matcher medical pilot eval checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
