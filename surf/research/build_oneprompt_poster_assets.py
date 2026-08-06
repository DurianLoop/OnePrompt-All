"""Build poster-ready tables and plots from the 220-image evaluation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / "runs" / "oneprompt_all" / "benchmark220_final_5click"
OUT = RUN / "poster_assets"

COLORS = {"banana": "#0f62fe", "broccoli": "#24a148", "overall": "#161616"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "trials": float(len(rows)),
        "f1": float(np.mean([float(row["f1"]) for row in rows])),
        "precision": float(np.mean([float(row["precision"]) for row in rows])),
        "recall": float(np.mean([float(row["recall"]) for row in rows])),
        "f1_std": float(np.std([float(row["f1"]) for row in rows])),
    }


def save_table(metrics: dict[str, dict[str, float]]) -> None:
    fields = ["Split", "Images", "Prompts", "F1", "Precision", "Recall", "Prompt F1 SD"]
    image_counts = {"banana": 20, "broccoli": 200, "overall": 220}
    labels = {"banana": "Banana-20", "broccoli": "Broccoli-200", "overall": "Merged-220"}
    data = [[labels[key], str(image_counts[key]), str(int(value["trials"])), f'{value["f1"]:.3f}', f'{value["precision"]:.3f}', f'{value["recall"]:.3f}', f'{value["f1_std"]:.3f}'] for key, value in metrics.items()]
    with (OUT / "poster_metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(data)
    with (OUT / "poster_metrics_table.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in data: handle.write("| " + " | ".join(row) + " |\n")

    fig, ax = plt.subplots(figsize=(11.5, 2.6), dpi=220); ax.axis("off")
    table = ax.table(cellText=data, colLabels=fields, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(13); table.scale(1, 2.15)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c6c6c6")
        if row == 0:
            cell.set_facecolor("#161616"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif row == len(data):
            cell.set_facecolor("#edf5ff"); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f4f4f4")
    fig.tight_layout(pad=.2); fig.savefig(OUT / "poster_metrics_table.png", bbox_inches="tight", transparent=False); plt.close(fig)


def plot_generalization(metrics: dict[str, dict[str, float]]) -> None:
    keys = ["banana", "broccoli", "overall"]; measures = [("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]
    fig, ax = plt.subplots(figsize=(10.6, 5.7), dpi=220); x = np.arange(len(measures)); width = .23
    for index, key in enumerate(keys):
        values = [metrics[key][field] for _, field in measures]
        bars = ax.bar(x + (index - 1) * width, values, width, label={"banana": "Banana-20", "broccoli": "Broccoli-200", "overall": "Merged-220"}[key], color=COLORS[key])
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_xticks(x, [name for name, _ in measures]); ax.set_ylabel("Score"); ax.set_title("OnePrompt-All: Generalization Across Same-Class Scenes", loc="left", weight="bold")
    ax.grid(axis="y", color="#e0e0e0"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncols=3, loc="upper center")
    fig.tight_layout(); fig.savefig(OUT / "poster_generalization.png", bbox_inches="tight"); plt.close(fig)


def plot_prompt_distribution(grouped: dict[str, list[dict[str, str]]]) -> None:
    labels, values = ["Banana-20", "Broccoli-200", "Merged-220"], [[float(row["f1"]) for row in grouped["banana"]], [float(row["f1"]) for row in grouped["broccoli"]], [float(row["f1"]) for row in grouped["banana"] + grouped["broccoli"]]]
    fig, ax = plt.subplots(figsize=(9.8, 5.7), dpi=220)
    plot = ax.violinplot(values, showmedians=True, showextrema=True)
    for body, color in zip(plot["bodies"], [COLORS["banana"], COLORS["broccoli"], COLORS["overall"]]): body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.75)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"): plot[key].set_color("#161616")
    rng = np.random.default_rng(20260731)
    for index, data in enumerate(values, 1): ax.scatter(rng.normal(index, .035, len(data)), data, s=5, color="#161616", alpha=.16, linewidths=0)
    ax.set_xticks([1, 2, 3], labels); ax.set_ylim(-.03, 1.04); ax.set_ylabel("Per-prompt F1"); ax.set_title("Click Sensitivity: Distribution Over 1,100 One-Point Trials", loc="left", weight="bold")
    ax.grid(axis="y", color="#e0e0e0"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "poster_click_robustness.png", bbox_inches="tight"); plt.close(fig)


def plot_image_stability(rows: list[dict[str, str]], prompt_rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in prompt_rows: grouped[row["image"]].append(float(row["f1"]))
    class_by_image = {row["image"]: row["class_name"] for row in prompt_rows}
    fig, ax = plt.subplots(figsize=(9.8, 5.7), dpi=220)
    for key in ("banana", "broccoli"):
        image_ids = [image for image in grouped if class_by_image[image] == key]
        means = [float(np.mean(grouped[image])) for image in image_ids]; stds = [float(np.std(grouped[image])) for image in image_ids]
        ax.scatter(means, stds, s=30, color=COLORS[key], alpha=.72, label=f"{key.title()} ({len(image_ids)} images)")
    ax.axvline(.8, color="#8d8d8d", linestyle="--", linewidth=1); ax.set_xlim(-.02, 1.02); ax.set_ylim(-.01, .44); ax.set_xlabel("Mean F1 across five random clicks"); ax.set_ylabel("F1 standard deviation across five clicks")
    ax.set_title("Image-Level Robustness: Accuracy vs. Click Sensitivity", loc="left", weight="bold")
    ax.grid(color="#e0e0e0"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "poster_image_stability.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = read_csv(RUN / "per_prompt.csv"); grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prompts: grouped[row["class_name"]].append(row)
    metrics = {"banana": summary(grouped["banana"]), "broccoli": summary(grouped["broccoli"]), "overall": summary(prompts)}
    save_table(metrics); plot_generalization(metrics); plot_prompt_distribution(grouped); plot_image_stability(read_csv(RUN / "per_image_prompt_sensitivity.csv"), prompts)
    print(OUT)


if __name__ == "__main__": main()
