from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fspl.qualitative import build_panel_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-selection qualitative panel instructions for stage 17.")
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--policy-selections", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    case_ids = payload.get("case_ids")
    if not isinstance(case_ids, list) or not all(isinstance(case_id, str) for case_id in case_ids):
        raise ValueError("case manifest must contain a string case_ids list")
    with args.policy_selections.open(newline="", encoding="utf-8") as handle:
        selections = list(csv.DictReader(handle))
    reference_case_ids = sorted(
        {
            str(row["case_id"]).split("::", maxsplit=1)[0]
            for row in selections
            if "::" in str(row.get("case_id", "")) and str(row["case_id"]).split("::", maxsplit=1)[1] in case_ids
        }
    )
    panel_manifest = build_panel_manifest(case_ids, reference_case_ids, selections)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(panel_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_path)


if __name__ == "__main__":
    main()
