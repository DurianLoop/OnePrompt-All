from __future__ import annotations

import argparse
from pathlib import Path

from fspl.stage2 import FixedBaselineConfig, run_fixed_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe MoNuSeg fixed prompt baselines.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2_fixed_baselines"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-count", type=int, default=6)
    args = parser.parse_args()
    summary = run_fixed_baselines(
        FixedBaselineConfig(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            validation_count=args.validation_count,
        )
    )
    print(
        "primary=" + summary["primary_rule_selected_on_validation"]
        + " elapsed=" + f"{summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
