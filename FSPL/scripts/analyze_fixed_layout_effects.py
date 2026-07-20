from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from fspl.statistics import paired_rule_effect


COMPLETED_REPLICATES = (
    ("dinov2_l_sam_b_seed20260716", Path("artifacts/stage6_monusac_heldout/case_action_utilities.csv")),
    ("dinov2_l_sam_b_seed20260717", Path("artifacts/stage8_monusac_seed20260717/case_action_utilities.csv")),
    ("dinov2_b_sam_b_seed20260716", Path("artifacts/stage13_dinob_samb_monusac/case_action_utilities.csv")),
    ("dinov2_l_sam_l_seed20260716", Path("artifacts/stage16_dinol_saml_monusac/case_action_utilities.csv")),
)
STAGE18_REPLICATE = ("dinov2_l_sam_h_seed20260716", Path("artifacts/stage18_dinol_samh_monusac/case_action_utilities.csv"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report paired Top-5 versus fixed spread-radius-3 effects.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--include-stage18", action="store_true")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing completed replicate rows: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"completed replicate has no rows: {path}")
    return rows


def _mean(rows: list[dict[str, str]], rule_name: str) -> float:
    values = [float(row["utility"]) for row in rows if row["rule_name"] == rule_name]
    if not values:
        raise ValueError(f"missing {rule_name!r} rows")
    return sum(values) / len(values)


def build_report(
    replicates: tuple[tuple[str, Path], ...], *, seed: int, resamples: int
) -> dict[str, object]:
    effects: dict[str, dict[str, object]] = {}
    for label, path in replicates:
        rows = _read_rows(path)
        effect = paired_rule_effect(
            rows,
            baseline_rule="top5_n3",
            variant_rule="spread5_n3",
            seed=seed,
            resamples=resamples,
        )
        if effect["episode_count"] != 303:
            raise ValueError(f"{label} has {effect['episode_count']} paired episodes; expected 303")
        effects[label] = {
            "source_csv": str(path),
            "top5_n3_mean_dice": _mean(rows, "top5_n3"),
            "spread5_n3_mean_dice": _mean(rows, "spread5_n3"),
            **effect,
        }
    return {
        "comparison": "spread5_n3 minus top5_n3 on the same reference-query episodes",
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "replicates": effects,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replicates = COMPLETED_REPLICATES + ((STAGE18_REPLICATE,) if args.include_stage18 else ())
    report = build_report(replicates, seed=args.seed, resamples=args.resamples)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_path)


if __name__ == "__main__":
    main()
