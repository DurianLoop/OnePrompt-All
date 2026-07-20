from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fspl.stage1 import Stage1Config, run_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen DINOv2-L plus SAM ViT-B FSPL episode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage1_smoke.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-limit", type=int, default=5)
    parser.add_argument("--negative-limit", type=int, default=3)
    args = parser.parse_args()

    config = replace(
        Stage1Config(),
        output_path=args.output,
        device=args.device,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
    )
    summary = run_stage1(config)
    print(
        f"reference={summary['reference_case_id']} query={summary['query_case_id']} "
        f"dice={summary['dice']:.6f} elapsed={summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
