from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapromptset.reproducibility.metadata import write_run_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_extra(values: list[str]) -> dict[str, str]:
    extra: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"extra values must use key=value syntax: {value}")
        key, raw = value.split("=", 1)
        extra[key] = raw
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Write AdaPromptSet run metadata with config and input hashes.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.out_dir or (args.project_root / "reports" / "run_metadata" / args.run_name)
    out_path = write_run_metadata(
        out_dir=out_dir,
        project_root=args.project_root,
        run_name=args.run_name,
        config_paths=args.config,
        input_paths=args.input,
        extra=_parse_extra(args.extra),
    )
    print(json.dumps({"run_metadata": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
