from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _as_box(text: str) -> list[int]:
    value = json.loads(text)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"bad box: {text}")
    return [int(v) for v in value]


def main() -> int:
    out_dir = WORKSPACE_ROOT / ".cache" / "matcher_medical_pilot_manifest_check"
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/build_matcher_medical_pilot_manifest.py",
        "--out-dir",
        str(out_dir),
        "--idrid-cases",
        "3",
        "--monuseg-cases",
        "2",
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    csv_path = out_dir / "medical_matcher_pilot_manifest.csv"
    json_path = out_dir / "medical_matcher_pilot_manifest.json"
    summary_path = out_dir / "summary.md"
    if not csv_path.exists() or not json_path.exists() or not summary_path.exists():
        print("Missing one or more output files.")
        return 1

    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
    if len(rows) < 5:
        print(f"Expected at least 5 rows, got {len(rows)}")
        return 1
    datasets = {row["dataset"] for row in rows}
    if not {"IDRID", "MonuSeg"}.issubset(datasets):
        print(f"Expected IDRID and MonuSeg rows, got {sorted(datasets)}")
        return 1

    required = {
        "pilot_id",
        "dataset",
        "support_case_id",
        "support_image_path",
        "support_mask_path",
        "support_box_xyxy",
        "query_case_id",
        "query_image_path",
        "query_mask_path",
        "target_hint",
    }
    for row in rows:
        missing = [key for key in required if not row.get(key)]
        if missing:
            print(f"Missing fields {missing} in row {row}")
            return 1
        for key in ("support_image_path", "support_mask_path", "query_image_path", "query_mask_path"):
            if not (WORKSPACE_ROOT / row[key]).exists():
                print(f"Path does not exist for {key}: {row[key]}")
                return 1
        box = _as_box(row["support_box_xyxy"])
        if not (box[0] < box[2] and box[1] < box[3]):
            print(f"Invalid support box: {box}")
            return 1

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("case_count") != len(rows):
        print("JSON case_count does not match CSV row count.")
        return 1
    print("Matcher medical pilot manifest checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
