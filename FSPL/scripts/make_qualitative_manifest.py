from __future__ import annotations

import argparse
import json
from pathlib import Path

from fspl.dataset import discover_idrid_cases
from fspl.qualitative import select_qualitative_case_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preselect reproducible IDRiD qualitative cases.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = discover_idrid_cases(args.dataset_root, "test")
    selected_ids = select_qualitative_case_ids([case.case_id for case in cases], count=args.count, seed=args.seed)
    selected = [case for case in cases if case.case_id in selected_ids]
    manifest = {
        "dataset": "IDRiD test",
        "selection_rule": "sorted unique case IDs shuffled with fixed Python random seed before evaluation results are viewed",
        "seed": args.seed,
        "case_ids": list(selected_ids),
        "cases": [
            {"case_id": case.case_id, "image_path": str(case.image_path), "mask_path": str(case.mask_path)}
            for case in selected
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
