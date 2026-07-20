from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def assemble_examples(
    manifests: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grid: dict[str, dict[int, dict[str, Any]]] = {}
    for row in negative_rows:
        requested_text = row.get("requested_negative_count")
        requested = int(requested_text if requested_text not in (None, "") else row["negative_count"])
        grid.setdefault(str(row["episode_id"]), {})[requested] = row
    decisions = {
        str(row["episode_id"]): str(row.get("selected_action") or row.get("action"))
        for row in learned_rows
    }
    examples: list[dict[str, Any]] = []
    for manifest in manifests:
        episode_id = str(manifest["pilot_id"])
        if episode_id not in grid or 0 not in grid[episode_id] or 5 not in grid[episode_id]:
            continue
        examples.append({
            "episode_id": episode_id,
            "manifest": manifest,
            "n0": grid[episode_id][0],
            "n5": grid[episode_id][5],
            "selected_action": decisions.get(episode_id, "unknown"),
        })
    return examples


def load_rgb(path_text: str) -> Image.Image:
    return Image.open(resolve(path_text)).convert("RGB")


def load_mask(path_text: str, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(resolve(path_text)).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def mask_panel(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8).copy()
    rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.asarray([0, 225, 125])).astype(np.uint8)
    return rgb


def build_figure(examples: list[dict[str, Any]], title: str, out_path: Path) -> None:
    if not examples:
        raise ValueError("No complete N0/N5 qualitative examples were found.")
    fig, axes = plt.subplots(len(examples), 5, figsize=(16, 3.45 * len(examples)), squeeze=False)
    column_titles = ("Reference image", "Reference mask", "Query + GT", "N=0 prediction", "N=5 prediction")
    for row_index, (row_axes, example) in enumerate(zip(axes, examples)):
        manifest = example["manifest"]
        reference = load_rgb(str(manifest["support_image_path"]))
        reference_mask = load_mask(str(manifest["support_mask_path"]), reference.size)
        query = load_rgb(str(manifest["query_image_path"]))
        query_mask = load_mask(str(manifest["query_mask_path"]), query.size)
        panels = (
            np.asarray(reference),
            reference_mask.astype(np.uint8) * 255,
            mask_panel(query, query_mask),
            np.asarray(load_rgb(str(example["n0"]["overlay_path"]))),
            np.asarray(load_rgb(str(example["n5"]["overlay_path"]))),
        )
        for column_index, (axis, panel, column_title) in enumerate(zip(row_axes, panels, column_titles)):
            axis.imshow(panel, cmap="gray" if column_index == 1 else None, vmin=0, vmax=255)
            if row_index == 0:
                axis.set_title(column_title, fontsize=11, fontweight="bold")
            axis.axis("off")
        delta = float(example["n5"]["dice"]) - float(example["n0"]["dice"])
        decision = example["selected_action"].upper()
        outcome = "correction" if delta > 0 else "failure"
        row_axes[0].text(
            0.02,
            0.98,
            f"{outcome}: {manifest['query_case_id']}\nGate {decision}, N5-N0 {delta:+.3f}",
            transform=row_axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
        )
        row_axes[3].text(
            0.02,
            0.98,
            f"Dice {float(example['n0']['dice']):.3f}",
            transform=row_axes[3].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
        )
        row_axes[4].text(
            0.02,
            0.98,
            f"Dice {float(example['n5']['dice']):.3f}",
            transform=row_axes[4].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.9},
        )
    fig.suptitle(
        title + "\nGreen=TP, red=FP, blue=FN, yellow=positive prompt, cyan=negative prompt",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference-aware negative-gate qualitative panels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--negative-rows", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    examples = assemble_examples(
        read_csv(resolve(args.manifest)),
        read_csv(resolve(args.negative_rows)),
        read_csv(resolve(args.learned)),
    )
    build_figure(examples, args.title, resolve(args.out))
    print(resolve(args.out))


if __name__ == "__main__":
    main()
