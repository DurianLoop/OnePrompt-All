"""Plot image-level accuracy versus click-location robustness for OnePrompt-All."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN = Path(r"D:\SAM\runs\oneprompt_all\surf_v2_full2220_5click")
INPUT = RUN / "per_image_prompt_sensitivity.csv"
OUT = RUN / "poster_assets" / "poster_robustness_quadrants.png"


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mean_f1 = np.asarray([float(row["mean_f1"]) for row in rows])
    f1_sd = np.asarray([float(row["std_f1"]) for row in rows])
    x_cut, y_cut = float(f1_sd.mean()), float(mean_f1.mean())
    x_max = max(.32, float(np.quantile(f1_sd, .995)) * 1.08)

    fig, ax = plt.subplots(figsize=(8.2, 8.0), dpi=240)
    # Faint backgrounds make the desired direction immediately visible.
    ax.axvspan(0, x_cut, ymin=(y_cut / 1.05), ymax=1, color="#d0e2ff", alpha=.22, zorder=0)
    ax.axvspan(x_cut, x_max, ymin=(y_cut / 1.05), ymax=1, color="#fff1c2", alpha=.22, zorder=0)
    ax.axvspan(0, x_cut, ymin=0, ymax=(y_cut / 1.05), color="#e8e8e8", alpha=.25, zorder=0)
    ax.axvspan(x_cut, x_max, ymin=0, ymax=(y_cut / 1.05), color="#ffd7d9", alpha=.22, zorder=0)
    ax.scatter(f1_sd, mean_f1, s=14, color="#0f62fe", alpha=.42, edgecolors="none", zorder=2)
    ax.axvline(x_cut, color="#525252", linestyle="--", linewidth=1.25, zorder=1)
    ax.axhline(y_cut, color="#525252", linestyle="--", linewidth=1.25, zorder=1)

    quadrants = [
        ((f1_sd <= x_cut) & (mean_f1 >= y_cut), "Robust", "#005d5d", .035, .945, "left", "top"),
        ((f1_sd > x_cut) & (mean_f1 >= y_cut), "Sensitive", "#8a3c00", .965, .945, "right", "top"),
        ((f1_sd <= x_cut) & (mean_f1 < y_cut), "Hard", "#525252", .035, .05, "left", "bottom"),
        ((f1_sd > x_cut) & (mean_f1 < y_cut), "Fragile", "#a2191f", .965, .05, "right", "bottom"),
    ]
    for selected, label, color, x, y, ha, va in quadrants:
        count = int(selected.sum())
        ax.text(x, y, f"{label}\n{count} ({count / len(rows):.1%})", transform=ax.transAxes,
                ha=ha, va=va, fontsize=11, weight="bold", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78, "pad": 3})

    ax.set_xlim(0, x_max); ax.set_ylim(-.02, 1.05)
    ax.set_xlabel("F1 SD across five clicks  ← lower is better", fontsize=11.5)
    ax.set_ylabel("Mean F1 across five clicks  ↑ higher is better", fontsize=11.5)
    ax.text(x_cut + .004, 1.015, f"mean SD = {x_cut:.3f}", fontsize=9.5, color="#525252", va="bottom")
    ax.text(x_max * .99, y_cut + .012, f"mean F1 = {y_cut:.3f}", fontsize=9.5, color="#525252", ha="right", va="bottom")
    ax.grid(color="#d9d9d9", linewidth=.7); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
