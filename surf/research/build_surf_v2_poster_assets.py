"""Create poster-ready tables from the completed 2,220-image SURF v2 run."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = Path(r"D:\SAM\runs\oneprompt_all")
FULL = RUNS / "surf_v2_full2220_5click" / "per_prompt.csv"
OUT = RUNS / "surf_v2_full2220_5click" / "poster_assets"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    f1 = np.asarray([float(row["f1"]) for row in rows])
    precision = np.asarray([float(row["precision"]) for row in rows])
    recall = np.asarray([float(row["recall"]) for row in rows])
    return {"f1": float(f1.mean()), "precision": float(precision.mean()),
            "recall": float(recall.mean()), "std": float(f1.std())}


def image_std(rows: list[dict[str, str]]) -> float:
    per_image: dict[str, list[float]] = {}
    for row in rows:
        per_image.setdefault(row["image"], []).append(float(row["f1"]))
    return float(np.mean([np.std(values) for values in per_image.values()]))


def write_table(groups: list[tuple[str, int, list[dict[str, str]]]]) -> None:
    fields = ["Evaluation split", "Images", "Point prompts", "F1", "Precision", "Recall", "Prompt F1 SD", "Image F1 SD"]
    rendered: list[list[str]] = []
    for label, images, rows in groups:
        stat = metrics(rows)
        rendered.append([label, str(images), str(len(rows)), f'{stat["f1"]:.3f}',
                         f'{stat["precision"]:.3f}', f'{stat["recall"]:.3f}',
                         f'{stat["std"]:.3f}', f'{image_std(rows):.3f}'])
    with (OUT / "poster_metrics_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rendered)
    with (OUT / "poster_metrics_table.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rendered:
            handle.write("| " + " | ".join(row) + " |\n")
        handle.write("\nProtocol: SAM3 + DINOv2-B soft evidence; one random foreground click; five clicks per image; no fixed automatic negatives.\n")
        handle.write("All values are from the completed full benchmark: 2,220 images and 11,100 independent one-point trials.\n")
    fig, ax = plt.subplots(figsize=(15.4, 2.9), dpi=220); ax.axis("off")
    table = ax.table(cellText=rendered, colLabels=fields, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11.2); table.scale(1, 2.1)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c6c6c6")
        if row == 0:
            cell.set_facecolor("#161616"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif row == len(rendered):
            cell.set_facecolor("#edf5ff"); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f4f4f4")
    fig.tight_layout(pad=.2); fig.savefig(OUT / "poster_metrics_table.png", bbox_inches="tight"); plt.close(fig)


def plot_transfer(groups: list[tuple[str, int, list[dict[str, str]]]]) -> None:
    labels = [label.replace(" ", "\n") for label, _, _ in groups]
    values = [metrics(rows) for _, _, rows in groups]
    categories = [("F1", "f1", "#0f62fe"), ("Precision", "precision", "#24a148"), ("Recall", "recall", "#8a3ffc")]
    x = np.arange(len(labels)); width = .23
    fig, ax = plt.subplots(figsize=(10.7, 5.6), dpi=220)
    for index, (name, key, color) in enumerate(categories):
        bars = ax.bar(x + (index - 1) * width, [item[key] for item in values], width, label=name, color=color)
        ax.bar_label(bars, [f'{item[key]:.3f}' for item in values], padding=3, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_xticks(x, labels); ax.set_ylabel("Score")
    ax.set_title("OnePrompt-All: Legacy vs. Full SURF v2 Benchmark", loc="left", weight="bold")
    ax.grid(axis="y", color="#e0e0e0"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    fig.tight_layout(); fig.savefig(OUT / "poster_cross_dataset_metrics.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = read_rows(FULL)
    legacy_trials = 220 * 5
    assert len(full) == 2220 * 5, f"Expected 11,100 trials, got {len(full)}"
    legacy, surf_v2 = full[:legacy_trials], full[legacy_trials:]
    groups = [("Legacy-220", 220, legacy), ("SURF v2 Filtered-2000", 2000, surf_v2),
              ("Unified-2220", 2220, full)]
    write_table(groups); plot_transfer(groups)
    print(OUT)


if __name__ == "__main__":
    main()
