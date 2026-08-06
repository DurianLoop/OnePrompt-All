"""Expand the full OnePrompt-All result into poster-sized breakdown tables."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\SAM")
MANIFEST = ROOT / "surf" / "datasets" / "oneprompt_all" / "surf_v2_unified" / "oneprompt_multiinstance_2220" / "manifest.csv"
PROMPTS = ROOT / "runs" / "oneprompt_all" / "surf_v2_full2220_5click" / "per_prompt.csv"
OUT = PROMPTS.parent / "poster_assets"
PROMPTS_PER_IMAGE = 5


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table_rows(manifest: list[dict[str, str]], prompts: list[dict[str, str]], key: str, label: str) -> tuple[list[str], list[list[str]]]:
    buckets: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        meta = manifest[index // PROMPTS_PER_IMAGE]
        buckets[meta[key]].append((meta, prompt))
    if key == "source_group":
        order = ["banana-20", "broccoli-200"] + [f"group_{n:02d}" for n in range(1, 21)]
        labels = {"banana-20": "Legacy / Banana", "broccoli-200": "Legacy / Broccoli"}
        labels.update({f"group_{n:02d}": f"SURF v2 / Partition {n:02d}" for n in range(1, 21)})
    else:
        order = sorted(buckets, key=lambda value: int(value))
        labels = {value: f"{value} instances / image" for value in order}
    fields = [label, "Images", "Classes", "Prompts", "F1", "Precision", "Recall", "Prompt F1 SD"]
    result: list[list[str]] = []
    for bucket in order:
        values = buckets[bucket]
        images = {item[0]["image"] for item in values}
        classes = {item[0]["class_name"] for item in values}
        f1 = np.asarray([float(item[1]["f1"]) for item in values])
        precision = np.asarray([float(item[1]["precision"]) for item in values])
        recall = np.asarray([float(item[1]["recall"]) for item in values])
        result.append([labels[bucket], str(len(images)), str(len(classes)), str(len(values)),
                       f"{f1.mean():.3f}", f"{precision.mean():.3f}", f"{recall.mean():.3f}", f"{f1.std():.3f}"])
    return fields, result


def save(name: str, fields: list[str], rows: list[list[str]], title: str) -> None:
    with (OUT / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
    with (OUT / f"{name}.md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows: handle.write("| " + " | ".join(row) + " |\n")
    fig_height = max(3.0, 0.43 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(15.5, fig_height), dpi=220); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=fields, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10.2 if len(rows) > 10 else 11.5); table.scale(1, 1.58)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c6c6c6")
        if row == 0:
            cell.set_facecolor("#161616"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif row in (1, 2):
            cell.set_facecolor("#f4f4f4")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f4f4f4")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=18)
    fig.tight_layout(pad=.4); fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, prompts = read(MANIFEST), read(PROMPTS)
    assert len(prompts) == len(manifest) * PROMPTS_PER_IMAGE
    fields, rows = table_rows(manifest, prompts, "source_group", "Benchmark source")
    save("poster_group_breakdown", fields, rows, "OnePrompt-All Full Benchmark: Source-Group Breakdown")
    fields, rows = table_rows(manifest, prompts, "target_instance_count", "Scene density")
    save("poster_density_breakdown", fields, rows, "OnePrompt-All Full Benchmark: Scene-Density Breakdown")
    print(OUT)


if __name__ == "__main__":
    main()
