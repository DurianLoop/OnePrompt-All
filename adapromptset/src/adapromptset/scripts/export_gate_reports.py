from __future__ import annotations

import argparse
from pathlib import Path

from adapromptset.reporting.gate_reports import build_standard_gate_reports


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export standardized AdaPromptSet gate reports.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    build_standard_gate_reports(args.project_root)


if __name__ == "__main__":
    main()
