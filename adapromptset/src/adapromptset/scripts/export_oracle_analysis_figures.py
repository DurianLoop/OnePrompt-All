from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _write_pareto(path: Path, rows: list[dict[str, str]]) -> None:
    budgets = [float(row["budget"]) for row in rows]
    fixed = [float(row["mean_best_fixed_dice"]) for row in rows]
    oracle = [float(row["mean_oracle_dice"]) for row in rows]
    gaps = [float(row["dice_oracle_gap"]) for row in rows]

    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=180)
    ax.plot(budgets, fixed, marker="o", linewidth=1.8, label="Best fixed")
    ax.plot(budgets, oracle, marker="s", linewidth=1.8, label="Oracle upper bound")
    ax.bar(budgets, gaps, width=0.25, alpha=0.25, label="Oracle gap")
    ax.set_xlabel("Maximum SAM calls")
    ax.set_ylabel("Mean Dice")
    ax.set_title("MoNuSAC Gate-F oracle analysis")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_oracle_wins(path: Path, rows: list[dict[str, str]]) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("oracle_count") or 0), reverse=True)
    labels = [row["action_id"].replace("_", "\n") for row in sorted_rows]
    oracle_counts = [float(row["oracle_count"]) for row in sorted_rows]
    mean_dice = [float(row["mean_dice"]) for row in sorted_rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 3.6), dpi=180)
    positions = list(range(len(sorted_rows)))
    ax1.bar(positions, oracle_counts, color="#4C78A8", alpha=0.85, label="Oracle wins")
    ax1.set_ylabel("Oracle-selected episodes")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=0, fontsize=6)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(positions, mean_dice, color="#F58518", marker="o", linewidth=1.4, label="Mean Dice")
    ax2.set_ylabel("Mean Dice")
    ax1.set_title("Action wins and mean Dice after Gate-F pivot")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _figure_item(project_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(project_root, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Oracle Analysis Figures",
        "",
        f"Status: `{summary['status']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        "",
        "These figures visualize existing Gate-F pivot artifacts only. They do not claim adaptive-policy success.",
        "",
        "## Figures",
        "",
    ]
    for name, item in summary["figures"].items():
        lines.append(f"- `{name}`: `{item['path']}` SHA256 `{item['sha256']}`")
    lines.extend(
        [
            "",
            "## Sources",
            "",
        ]
    )
    for name, item in summary["sources"].items():
        lines.append(f"- `{name}`: `{item['path']}` SHA256 `{item['sha256']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oracle_analysis_figures(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    analysis_dir = project_root / "reports" / "fixed_oracle_analysis"
    budget_path = analysis_dir / "budget_pareto.csv"
    action_path = analysis_dir / "action_summary.csv"
    budget_rows = _read_csv(budget_path)
    action_rows = _read_csv(action_path)

    outputs = {
        "pareto_png": project_root / "paper" / "figures" / "oracle_pareto.png",
        "oracle_wins_png": project_root / "paper" / "figures" / "oracle_action_wins.png",
        "json": project_root / "reports" / "oracle_analysis_figures.json",
        "markdown": project_root / "reports" / "oracle_analysis_figures.md",
    }
    _write_pareto(outputs["pareto_png"], budget_rows)
    _write_oracle_wins(outputs["oracle_wins_png"], action_rows)

    summary = {
        "schema_version": "adapromptset.oracle_analysis_figures.v1",
        "status": "PASS_ORACLE_ANALYSIS_FIGURES",
        "claim_boundary": "oracle_analysis_only",
        "source_rows": {
            "budget_pareto": len(budget_rows),
            "action_summary": len(action_rows),
        },
        "sources": {
            "budget_pareto": {
                "path": _relative(project_root, budget_path),
                "sha256": _sha256(budget_path),
            },
            "action_summary": {
                "path": _relative(project_root, action_path),
                "sha256": _sha256(action_path),
            },
        },
        "figures": {
            "pareto_png": _figure_item(project_root, outputs["pareto_png"]),
            "oracle_wins_png": _figure_item(project_root, outputs["oracle_wins_png"]),
        },
        "prohibited_claims": [
            "adaptive_policy_success",
            "test_time_action_search",
            "efficiency_gain",
            "transfer_or_generalization",
        ],
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export claim-safe oracle-analysis figures from real artifacts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_oracle_analysis_figures(args.project_root)


if __name__ == "__main__":
    main()
