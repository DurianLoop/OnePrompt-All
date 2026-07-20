from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def prepare_fold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_float = (
        "mean_fixed_dice",
        "mean_learned_dice",
        "mean_delta_dice",
        "ci95_low",
        "ci95_high",
    )
    numeric_int = ("fold", "case_count", "wins", "ties", "losses")
    prepared: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in rows:
        row = dict(source)
        for key in numeric_float:
            row[key] = float(row[key])
        for key in numeric_int:
            row[key] = int(row[key])
        if row["fold"] in seen:
            raise ValueError(f"Duplicate fold: {row['fold']}")
        seen.add(row["fold"])
        prepared.append(row)
    return sorted(prepared, key=lambda row: row["fold"])


def build_figure(
    fold_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = [f"Fold {row['fold']}" for row in fold_rows] + ["Pooled"]
    fixed = [row["mean_fixed_dice"] for row in fold_rows] + [float(aggregate["mean_fixed_dice"])]
    learned = [row["mean_learned_dice"] for row in fold_rows] + [float(aggregate["mean_learned_dice"])]
    effects = [row["mean_delta_dice"] for row in fold_rows] + [float(aggregate["mean_delta_dice"])]
    ci_low = [row["ci95_low"] for row in fold_rows] + [float(aggregate["ci95_low"])]
    ci_high = [row["ci95_high"] for row in fold_rows] + [float(aggregate["ci95_high"])]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, fixed, width, color="#59788e", label="Fixed N anchor")
    axes[0].bar(x + width / 2, learned, width, color="#cf5c45", label="Learned N gate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean Dice")
    axes[0].set_title("Outer-test performance by fold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    lower = max(0.0, min(fixed + learned) - 0.04)
    upper = min(1.0, max(fixed + learned) + 0.04)
    axes[0].set_ylim(lower, upper)

    y = np.arange(len(labels))
    errors = np.asarray([
        np.asarray(effects) - np.asarray(ci_low),
        np.asarray(ci_high) - np.asarray(effects),
    ])
    colors = ["#cf5c45" if index == len(labels) - 1 else "#4f7988" for index in range(len(labels))]
    for index, color in enumerate(colors):
        axes[1].errorbar(
            effects[index],
            y[index],
            xerr=errors[:, index:index + 1],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7 if index == len(labels) - 1 else 5,
        )
    axes[1].axvline(0.0, color="#252525", linewidth=1, linestyle="--")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Delta Dice (learned - fixed)")
    axes[1].set_title("Query-level bootstrap 95% CI")
    axes[1].grid(axis="x", alpha=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready MoNuSeg nested-CV table and figure.")
    parser.add_argument(
        "--aggregate-dir",
        default="runs/research_direction/nested_cv5_negative_budget/aggregate",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/research_direction/paper_artifacts_20260715",
    )
    args = parser.parse_args()

    aggregate_dir = resolve(args.aggregate_dir)
    out_dir = resolve(args.out_dir)
    fold_rows = prepare_fold_rows(read_csv(aggregate_dir / "fold_summary.csv"))
    aggregate = json.loads((aggregate_dir / "paired_summary.json").read_text(encoding="utf-8"))
    table_rows = [
        {
            "scope": f"fold_{row['fold']}",
            "query_count": row["case_count"],
            "fixed_dice": row["mean_fixed_dice"],
            "learned_dice": row["mean_learned_dice"],
            "delta_dice": row["mean_delta_dice"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "wins": row["wins"],
            "ties": row["ties"],
            "losses": row["losses"],
        }
        for row in fold_rows
    ]
    table_rows.append({
        "scope": "pooled_37_queries",
        "query_count": aggregate["case_count"],
        "fixed_dice": aggregate["mean_fixed_dice"],
        "learned_dice": aggregate["mean_learned_dice"],
        "delta_dice": aggregate["mean_delta_dice"],
        "ci95_low": aggregate["ci95_low"],
        "ci95_high": aggregate["ci95_high"],
        "wins": aggregate["wins"],
        "ties": aggregate["ties"],
        "losses": aggregate["losses"],
    })
    write_csv(out_dir / "nested_cv_result_table.csv", table_rows)
    build_figure(fold_rows, aggregate, out_dir / "nested_cv_forest.png")
    print(json.dumps(table_rows[-1], indent=2))


if __name__ == "__main__":
    main()
