from __future__ import annotations

import argparse
from pathlib import Path

from adapromptset.reporting.related_work import write_related_work_assets


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AdaPromptSet related-work boundary assets.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_related_work_assets(args.project_root)


if __name__ == "__main__":
    main()
