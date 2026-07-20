from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fspl.reporting import summarize_rule_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate balanced action rows and emit a rule-wise Dice summary.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_path = args.artifact_dir / "case_action_utilities.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing action utility CSV: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_rule_rows(rows, expected_episodes=args.expected_episodes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
