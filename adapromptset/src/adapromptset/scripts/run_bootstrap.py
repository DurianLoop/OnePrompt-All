from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from adapromptset.statistics.bootstrap import (
    build_query_level_comparison,
    paired_cluster_bootstrap,
    paired_permutation_test,
    win_tie_loss,
)
from adapromptset.statistics.units import read_csv_rows


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    bootstrap = payload["bootstrap"]
    significance = payload["paired_significance"]
    ci_text = "N/A"
    if bootstrap["ci_low"] is not None and bootstrap["ci_high"] is not None:
        ci_text = f"[{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]"
    p_text = "N/A"
    if significance["p_value_two_sided"] is not None:
        p_text = f"{significance['p_value_two_sided']:.6f}"
    lines = [
        "# Bootstrap Summary",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This report uses patient-cluster bootstrap over query-level paired differences.",
        "Reference episodes are aggregated before resampling.",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Query count | {bootstrap['query_count']} |",
        f"| Patient clusters | {bootstrap['cluster_count']} |",
        f"| Observed mean delta | {bootstrap['observed_mean_delta']:.6f} |",
        f"| 95% CI | {ci_text} |",
        f"| Paired permutation p-value | {p_text} |",
        f"| Wins | {payload['win_tie_loss']['win']} |",
        f"| Ties | {payload['win_tie_loss']['tie']} |",
        f"| Losses | {payload['win_tie_loss']['loss']} |",
        "",
    ]
    if payload["status"] != "PASS":
        lines.extend(
            [
                "Interpretation:",
                "",
                "- Cluster count is too small for a reliable confidence interval.",
                "- Do not use this pilot bootstrap or paired significance as formal evidence.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_bootstrap(
    *,
    episode_regret_csv: Path,
    episode_manifest_csv: Path,
    out_dir: Path,
    iterations: int = 10000,
    seed: int = 0,
    tie_threshold: float = 0.005,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rows = build_query_level_comparison(
        read_csv_rows(episode_regret_csv),
        read_csv_rows(episode_manifest_csv),
        baseline_metric="best_fixed_dice",
        comparator_metric="oracle_dice",
    )
    bootstrap = paired_cluster_bootstrap(query_rows, iterations=iterations, seed=seed)
    paired_significance = paired_permutation_test(query_rows, iterations=iterations, seed=seed)
    payload = {
        "status": bootstrap["status"],
        "input_episode_regret_csv": str(episode_regret_csv),
        "input_episode_manifest_csv": str(episode_manifest_csv),
        "bootstrap": bootstrap,
        "paired_significance": paired_significance,
        "win_tie_loss": win_tie_loss(query_rows, tie_threshold=tie_threshold),
        "tie_threshold": tie_threshold,
        "comparison": "oracle_vs_best_fixed",
        "rule": "Aggregate references to query-level before patient-cluster bootstrap.",
    }
    _write_csv(out_dir / "query_level_comparison.csv", query_rows)
    (out_dir / "bootstrap_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(out_dir / "bootstrap_summary.md", payload)
    return 0 if payload["status"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query-level paired patient-cluster bootstrap.")
    parser.add_argument(
        "--episode-regret-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "fixed_oracle_analysis" / "episode_regret.csv",
    )
    parser.add_argument(
        "--episode-manifest-csv",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "episodes" / "monusac" / "episodes_val.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "bootstrap")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie-threshold", type=float, default=0.005)
    args = parser.parse_args()
    raise SystemExit(
        run_bootstrap(
            episode_regret_csv=args.episode_regret_csv,
            episode_manifest_csv=args.episode_manifest_csv,
            out_dir=args.out_dir,
            iterations=args.iterations,
            seed=args.seed,
            tie_threshold=args.tie_threshold,
        )
    )


if __name__ == "__main__":
    main()
