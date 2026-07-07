from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_casebook_expansion_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/build_matcher_casebook_manifest.py",
        "--search-root",
        "runs/research_direction",
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    payload = json.loads((out_dir / "expanded_cases.json").read_text(encoding="utf-8"))
    if len(payload["cases"]) < 7:
        print(json.dumps(payload, indent=2))
        return 1
    print("Matcher casebook expansion checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
