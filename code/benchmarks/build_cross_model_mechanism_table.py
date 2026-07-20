from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def summarize_paired_methods(
    rows: list[dict[str, str]],
    fg_method: str,
    bg_method: str,
    bootstrap_samples: int = 10000,
    seed: int = 20260715,
) -> dict[str, Any]:
    by_method = {
        method: {row["query_case_id"]: row for row in rows if row["method"] == method}
        for method in (fg_method, bg_method)
    }
    case_ids = sorted(set(by_method[fg_method]) & set(by_method[bg_method]))
    if not case_ids:
        raise ValueError(f"No paired cases for {fg_method} and {bg_method}.")
    if len(case_ids) != len(by_method[fg_method]) or len(case_ids) != len(by_method[bg_method]):
        raise ValueError("The foreground and background-suppressed methods must contain identical query cases.")

    metrics = ("dice", "object_recall", "merge_error", "pred_area_pct")
    output: dict[str, Any] = {"case_count": len(case_ids), "fg_method": fg_method, "bg_method": bg_method}
    for metric in metrics:
        fg_values = np.asarray([float(by_method[fg_method][case_id][metric]) for case_id in case_ids])
        bg_values = np.asarray([float(by_method[bg_method][case_id][metric]) for case_id in case_ids])
        output[f"mean_fg_{metric}"] = float(fg_values.mean())
        output[f"mean_bg_{metric}"] = float(bg_values.mean())
        output[f"mean_delta_{metric}"] = float((bg_values - fg_values).mean())

    deltas = np.asarray([
        float(by_method[bg_method][case_id]["dice"]) - float(by_method[fg_method][case_id]["dice"])
        for case_id in case_ids
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    output.update({
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "wins": int((deltas > 1e-9).sum()),
        "ties": int((np.abs(deltas) <= 1e-9).sum()),
        "losses": int((deltas < -1e-9).sum()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    })
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_forest_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["model"]) for row in rows]
    means = np.asarray([float(row["mean_delta_dice"]) for row in rows])
    lows = np.asarray([float(row["ci95_low"]) for row in rows])
    highs = np.asarray([float(row["ci95_high"]) for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, max(3.2, 0.75 * len(rows) + 1.4)))
    ax.axvline(0.0, color="#667085", linewidth=1.2, linestyle="--")
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="o", color="#1261a0", ecolor="#2f80c9", capsize=4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Dice change: fg-bg matcher minus fg-only matcher")
    ax.set_title("Fixed K=40, lambda=0.75 cross-segmenter transfer")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired cross-segmenter background-suppression table.")
    parser.add_argument("--run", action="append", required=True, help="MODEL=path/to/candidates.csv")
    parser.add_argument("--fg-method", default="fg_topk_K40")
    parser.add_argument("--bg-method", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for item in args.run:
        model, path_text = item.split("=", 1)
        summary = summarize_paired_methods(read_csv(Path(path_text)), args.fg_method, args.bg_method)
        summaries.append({"model": model, "feature_backbone": "DINOv2-L/14", **summary, "source": path_text})

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "cross_segmenter_background_suppression.csv", summaries)
    save_forest_plot(summaries, out_dir / "cross_segmenter_background_suppression.png")
    conclusion = "universal_positive" if all(float(row["mean_delta_dice"]) > 0 for row in summaries) else "partial_or_model_dependent"
    (out_dir / "summary.json").write_text(json.dumps({
        "conclusion": conclusion,
        "model_count": len(summaries),
        "fixed_protocol": {"k": 40, "lambda_bg": 0.75},
        "models": summaries,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "models": summaries}, indent=2))


if __name__ == "__main__":
    main()
