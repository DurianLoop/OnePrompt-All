from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from export_matcher_stage1_points import (
    IMAGE_SIZE,
    extract_features,
    load_dinov2,
    load_rgb,
    make_reference_mask,
    mask_to_patch_weights,
    parse_box,
    resized_rgb,
    resolve_path,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SlidingPoint:
    rank: int
    tile_id: int
    tile_x0: int
    tile_y0: int
    tile_x1: int
    tile_y1: int
    patch_index: int
    x_original: float
    y_original: float
    similarity: float
    selected_by_threshold: bool


def clean_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    tiles = []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            tiles.append((x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size)))
    return tiles


def restore_tile_coordinates(x_518: int, y_518: int, tile: tuple[int, int, int, int]) -> tuple[float, float]:
    tile_x0, tile_y0, tile_x1, tile_y1 = tile
    tile_w = tile_x1 - tile_x0
    tile_h = tile_y1 - tile_y0
    x = tile_x0 + (float(x_518) / float(IMAGE_SIZE)) * float(tile_w)
    y = tile_y0 + (float(y_518) / float(IMAGE_SIZE)) * float(tile_h)
    return x, y


@torch.inference_mode()
def reference_proto(model: torch.nn.Module, reference_image: Path, reference_box: str, device: torch.device) -> torch.Tensor:
    original, resized = resized_rgb(reference_image)
    mask = make_reference_mask(original, parse_box(reference_box), None)
    features = extract_features(model, resized, device)
    weights = mask_to_patch_weights(mask, int(model.patch_size), device)
    foreground = weights > 0
    proto = (features[foreground] * weights[foreground].unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


@torch.inference_mode()
def rank_tile(
    model: torch.nn.Module,
    proto: torch.Tensor,
    tile_image: Image.Image,
    tile: tuple[int, int, int, int],
    tile_id: int,
    per_tile_top_k: int,
) -> list[SlidingPoint]:
    resized = tile_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    features = extract_features(model, resized, next(model.parameters()).device)
    similarity = (proto @ features.t()).squeeze(0)
    safe_top_k = min(per_tile_top_k, int(similarity.numel()))
    scores, indices = similarity.topk(safe_top_k)
    feat_size = IMAGE_SIZE // int(model.patch_size)
    points: list[SlidingPoint] = []
    for local_rank, (score_t, index_t) in enumerate(zip(scores, indices), start=1):
        patch_index = int(index_t.detach().cpu())
        patch_x = patch_index % feat_size
        patch_y = patch_index // feat_size
        x_518 = int(patch_x * int(model.patch_size) + int(model.patch_size) // 2)
        y_518 = int(patch_y * int(model.patch_size) + int(model.patch_size) // 2)
        x_original, y_original = restore_tile_coordinates(x_518, y_518, tile)
        points.append(SlidingPoint(
            rank=local_rank,
            tile_id=tile_id,
            tile_x0=tile[0],
            tile_y0=tile[1],
            tile_x1=tile[2],
            tile_y1=tile[3],
            patch_index=patch_index,
            x_original=round(x_original, 3),
            y_original=round(y_original, 3),
            similarity=round(float(score_t.detach().cpu()), 6),
            selected_by_threshold=False,
        ))
    return points


def draw_overlay(image: Image.Image, points: list[SlidingPoint], out_path: Path, threshold: float) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for point in reversed(points):
        radius = 5 if point.selected_by_threshold else 3
        fill = (255, 0, 0) if point.selected_by_threshold else (255, 230, 0)
        draw.ellipse((point.x_original - radius, point.y_original - radius, point.x_original + radius, point.y_original + radius), fill=fill, outline=(255, 255, 255), width=2)
    draw.rectangle((4, canvas.height - 38, min(canvas.width - 4, 520), canvas.height - 4), fill=(0, 0, 0))
    draw.text((12, canvas.height - 29), f"sliding-window merged points | red >= {threshold}", fill=(255, 255, 255))
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export no-downsample sliding-window Matcher Stage 1 points.")
    parser.add_argument("--reference-image", default="code/gradio_demo/images/elephant1.png")
    parser.add_argument("--target-image", default="code/gradio_demo/images/elephant3.png")
    parser.add_argument("--reference-box", default="0,60,295,425")
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--per-tile-top-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--dinov2-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dinov2-size", default="vit_large")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p15_sliding_window/elephant3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = clean_device(args.device)
    model = load_dinov2(resolve_path(args.dinov2_weights), args.dinov2_size, device)
    proto = reference_proto(model, resolve_path(args.reference_image), args.reference_box, device)
    target = load_rgb(resolve_path(args.target_image))
    all_points: list[SlidingPoint] = []
    tiles = iter_tiles(target.width, target.height, args.tile_size, args.overlap)
    for tile_id, tile in enumerate(tiles, start=1):
        tile_image = target.crop(tile)
        all_points.extend(rank_tile(model, proto, tile_image, tile, tile_id, args.per_tile_top_k))
    all_points.sort(key=lambda point: point.similarity, reverse=True)
    merged = all_points[: args.top_k]
    for rank, point in enumerate(merged, start=1):
        point.rank = rank
        point.selected_by_threshold = point.similarity >= args.threshold
    rows = [asdict(point) for point in merged]
    with (out_dir / "merged_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(SlidingPoint.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "reference_image": args.reference_image,
        "target_image": args.target_image,
        "reference_box": args.reference_box,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "tile_count": len(tiles),
        "top_k": args.top_k,
        "threshold": args.threshold,
        "selected_count": sum(1 for point in merged if point.selected_by_threshold),
        "points": rows,
    }
    (out_dir / "merged_points.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_overlay(target, merged, out_dir / "sliding_window_overlay.png", args.threshold)
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# Sliding-Window Stage 1 Summary",
            "",
            f"- Tile count: `{len(tiles)}`",
            f"- Top-k: `{args.top_k}`",
            f"- Selected count: `{payload['selected_count']}`",
            "",
            "## Artifacts",
            "",
            "- `merged_points.csv`",
            "- `merged_points.json`",
            "- `sliding_window_overlay.png`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Tile count: {len(tiles)}")
    print(f"Selected count: {payload['selected_count']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
