from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_transfer_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_medical_transfer_plan.py",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    payload = json.loads((out_dir / "medical_transfer_plan.json").read_text(encoding="utf-8"))
    required = {"medical_gate", "prediction_eval", "matcher_transfer_protocol", "readiness"}
    if not required.issubset(payload):
        print(json.dumps(payload, indent=2))
        return 1
    print("Matcher medical transfer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
