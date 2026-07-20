from __future__ import annotations

import argparse
from pathlib import Path

from fspl.stage4 import MultiReferenceFeasibilityConfig, run_multireference_feasibility


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a train-only multi-reference utility-policy feasibility pass.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage4_monusac_multireference_feasibility"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--reference-count", type=int, default=3)
    parser.add_argument("--policy-train-query-count", type=int, default=48)
    parser.add_argument("--validation-query-count", type=int, default=12)
    args = parser.parse_args()
    summary = run_multireference_feasibility(
        MultiReferenceFeasibilityConfig(
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            reference_count=args.reference_count,
            policy_train_query_count=args.policy_train_query_count,
            validation_query_count=args.validation_query_count,
        )
    )
    print(
        "validation_policy_dice=" + f"{summary['validation']['policy_selected_mean_dice']:.6f}"
        + " elapsed=" + f"{summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
