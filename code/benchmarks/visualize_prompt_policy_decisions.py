from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib_policy_qualitative"))


def segmentation_error_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape != pred.shape:
        raise ValueError("GT and prediction masks must have the same shape.")
    rgb = np.full((*gt.shape, 3), 245, dtype=np.uint8)
    rgb[np.logical_and(gt, pred)] = (35, 190, 90)
    rgb[np.logical_and(~gt, pred)] = (235, 65, 55)
    rgb[np.logical_and(gt, ~pred)] = (55, 125, 235)
    return rgb


def pair_by_query(
    fixed_rows: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fixed = {str(row["query_case_id"]): row for row in fixed_rows}
    learned = {str(row["query_case_id"]): row for row in learned_rows}
    if len(fixed) != len(fixed_rows) or len(learned) != len(learned_rows):
        raise ValueError("pair_by_query requires one row per query; aggregate or use episode IDs for multi-reference inputs.")
    if fixed.keys() != learned.keys():
        raise ValueError("Fixed and learned query sets differ.")
    return [
        {
            "query_case_id": query_id,
            "fixed": fixed[query_id],
            "learned": learned[query_id],
            "delta_dice": round(float(learned[query_id]["dice"]) - float(fixed[query_id]["dice"]), 12),
        }
        for query_id in sorted(fixed)
    ]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.NEAREST)
        array = np.asarray(image)
    return np.any(array > 0, axis=2) if array.ndim == 3 else array > 0


def draw_points(
    image: Image.Image,
    points_path: Path,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for row in reversed(read_csv(points_path)):
        rank = int(float(row.get("rank", 999)))
        x, y = float(row["x"]), float(row["y"])
        radius = 12 if rank <= 10 else 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=outline,
            width=3,
        )
        if rank <= 10:
            draw.text((x + radius + 1, y - radius), str(rank), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def score_path_from_points(points_path: Path) -> Path:
    return points_path.parent.parent / "t1_matching" / points_path.name.replace("_points.csv", "_score.npy")


def mask_image(mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], np.asarray([255, 255, 255], dtype=np.uint8), 0).astype(np.uint8)


def compose_case_figure(pair: dict[str, Any], episode: dict[str, str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    query = Image.open(resolve(episode["query_image_path"])).convert("RGB")
    reference = Image.open(resolve(episode["support_image_path"])).convert("RGB")
    gt = load_mask(resolve(episode["query_mask_path"]), query.size)
    reference_mask = load_mask(resolve(episode["support_mask_path"]), reference.size)
    fixed = pair["fixed"]
    learned = pair["learned"]
    fixed_mask = load_mask(resolve(fixed["mask_path"]), query.size)
    learned_mask = load_mask(resolve(learned["mask_path"]), query.size)
    fixed_points = draw_points(query, resolve(fixed["points_path"]), fill=(255, 225, 0), outline=(120, 20, 0))
    learned_points = draw_points(query, resolve(learned["points_path"]), fill=(0, 235, 255), outline=(0, 35, 110))
    score_map = np.load(score_path_from_points(resolve(learned["points_path"])))

    fig, axes = plt.subplots(2, 6, figsize=(22, 8.2), constrained_layout=True)
    panels = [
        (reference, "Reference image", None),
        (mask_image(reference_mask), "Reference mask", None),
        (query, "Query image", None),
        (mask_image(gt), "Query GT", None),
        (score_map, "Matcher score map", "magma"),
    ]
    for axis, (image, title, cmap) in zip(axes[0, :5], panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=12)
        axis.axis("off")

    axes[0, 5].axis("off")
    axes[0, 5].text(
        0.0,
        0.95,
        "Error colors\n"
        "green: TP\nred: FP\nblue: FN\n\n"
        f"Fixed\n{fixed.get('method', fixed.get('action', ''))}\n"
        f"K={float(fixed['k']):g}, lambda={float(fixed['lambda_bg']):g}\n"
        f"Dice={float(fixed['dice']):.4f}\n\n"
        f"Learned\n{learned.get('method', learned.get('action', ''))}\n"
        f"K={float(learned['k']):g}, lambda={float(learned['lambda_bg']):g}\n"
        f"Dice={float(learned['dice']):.4f}\n"
        f"Delta={float(pair['delta_dice']):+.4f}",
        va="top",
        fontsize=11,
        linespacing=1.4,
    )

    comparison = [
        (fixed_points, "Fixed points", None),
        (mask_image(fixed_mask), "Fixed mask", None),
        (segmentation_error_rgb(gt, fixed_mask), "Fixed error", None),
        (learned_points, "Learned points", None),
        (mask_image(learned_mask), "Learned mask", None),
        (segmentation_error_rgb(gt, learned_mask), "Learned error", None),
    ]
    for axis, (image, title, cmap) in zip(axes[1], comparison):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    fig.suptitle(f"{pair['query_case_id']} | policy delta Dice {float(pair['delta_dice']):+.4f}", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_qualitative_cases(pairs: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    changed = [pair for pair in pairs if str(pair["fixed"].get("method", pair["fixed"].get("action", ""))) != str(pair["learned"].get("action", pair["learned"].get("method", "")))]
    pool = changed or pairs
    best = sorted(pool, key=lambda row: float(row["delta_dice"]), reverse=True)
    worst = sorted(pool, key=lambda row: float(row["delta_dice"]))
    selected: list[dict[str, Any]] = []
    for row in best[: max(1, (max_cases + 1) // 2)] + worst[: max_cases // 2]:
        if row["query_case_id"] not in {item["query_case_id"] for item in selected}:
            selected.append(row)
    return selected[:max_cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create separated, paper-ready qualitative policy decision panels.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=6)
    args = parser.parse_args()

    pairs = pair_by_query(read_csv(resolve(args.fixed)), read_csv(resolve(args.learned)))
    episodes = {row["query_case_id"]: row for row in read_csv(resolve(args.episodes))}
    selected = select_qualitative_cases(pairs, args.max_cases)
    out_dir = resolve(args.out_dir)
    index_rows: list[dict[str, Any]] = []
    for pair in selected:
        query_id = pair["query_case_id"]
        out_path = out_dir / f"{query_id}.png"
        compose_case_figure(pair, episodes[query_id], out_path)
        index_rows.append({
            "query_case_id": query_id,
            "fixed_action": pair["fixed"].get("method", pair["fixed"].get("action", "")),
            "learned_action": pair["learned"].get("action", pair["learned"].get("method", "")),
            "fixed_dice": pair["fixed"]["dice"],
            "learned_dice": pair["learned"]["dice"],
            "delta_dice": pair["delta_dice"],
            "figure_path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(index_rows)} qualitative figures to {out_dir}")


if __name__ == "__main__":
    main()
