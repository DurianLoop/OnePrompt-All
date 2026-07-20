from __future__ import annotations

import argparse
from pathlib import Path

from fspl.stage3 import UtilityPolicyConfig, run_utility_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the offline FSPL utility policy.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage3_utility_policy"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-count", type=int, default=6)
    args = parser.parse_args()
    summary = run_utility_policy(
        UtilityPolicyConfig(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            validation_count=args.validation_count,
        )
    )
    print(
        "test_policy_dice=" + f"{summary['official_test']['policy_selected_mean_dice']:.6f}"
        + " elapsed=" + f"{summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
