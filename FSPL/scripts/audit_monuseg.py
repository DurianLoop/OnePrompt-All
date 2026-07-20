from __future__ import annotations

import argparse
from pathlib import Path

from fspl.dataset import discover_cases, write_audit_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official MoNuSeg image-mask pairs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_cases = discover_cases(args.root, "train")
    test_cases = discover_cases(args.root, "test")
    write_audit_csv([*train_cases, *test_cases], args.out)
    print(f"wrote {len(train_cases)} train and {len(test_cases)} test cases to {args.out}")


if __name__ == "__main__":
    main()
