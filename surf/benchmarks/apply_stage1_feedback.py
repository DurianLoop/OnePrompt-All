from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 518
PATCH_SIZE = 14


@dataclass
class FeedbackPoint:
    x: float
    y: float


@dataclass
class RankedPoint:
    rank: int
    patch_index: int
    patch_x: int
    patch_y: int
    x_518: int
    y_518: int
    x_original: float
    y_original: float
    similarity_before: float
    similarity_after: float
    delta: float
    selected_by_threshold: bool


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def parse_points(text: str) -> list[FeedbackPoint]:
    text = text.strip()
    if not text:
        return []
    points: list[FeedbackPoint] = []
    for item in text.split(";"):
        parts = [float(part.strip()) for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Point must be formatted as x,y: {item}")
        points.append(FeedbackPoint(x=parts[0], y=parts[1]))
    return points


def load_stage1(stage1_dir: Path) -> tuple[dict[str, Any], np.ndarray, Image.Image]:
    points_json = stage1_dir / "points.json"
    similarity_npy = stage1_dir / "similarity.npy"
    if not points_json.exists():
        raise FileNotFoundError(points_json)
    if not similarity_npy.exists():
        raise FileNotFoundError(similarity_npy)

    payload = json.loads(points_json.read_text(encoding="utf-8"))
    similarity = np.load(similarity_npy).astype(np.float32)
    target_image = Image.open(resolve_path(payload["target_image"])).convert("RGB")
    return payload, similarity, target_image


def gaussian_field(shape: tuple[int, int], points: list[FeedbackPoint], original_size: tuple[int, int], sigma: float) -> np.ndarray:
    if not points:
        return np.zeros(shape, dtype=np.float32)
    h, w = shape
    orig_w, orig_h = original_size
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros(shape, dtype=np.float32)
    sigma_patch = max(float(sigma) / PATCH_SIZE, 1e-3)
    for point in points:
        px = (point.x / float(orig_w)) * float(IMAGE_SIZE)
        py = (point.y / float(orig_h)) * float(IMAGE_SIZE)
        cx = px / PATCH_SIZE
        cy = py / PATCH_SIZE
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        field += np.exp(-dist2 / (2.0 * sigma_patch * sigma_patch)).astype(np.float32)
    return field


def apply_feedback(
    similarity: np.ndarray,
    positive_points: list[FeedbackPoint],
    negative_points: list[FeedbackPoint],
    original_size: tuple[int, int],
    sigma: float,
    positive_weight: float,
    negative_weight: float,
) -> np.ndarray:
    positive = gaussian_field(similarity.shape, positive_points, original_size, sigma)
    negative = gaussian_field(similarity.shape, negative_points, original_size, sigma)
    reranked = similarity + float(positive_weight) * positive - float(negative_weight) * negative
    return reranked.astype(np.float32)


def rank_points(before: np.ndarray, after: np.ndarray, original_size: tuple[int, int], top_k: int, threshold: float) -> list[RankedPoint]:
    flat_after = after.reshape(-1)
    flat_before = before.reshape(-1)
    safe_top_k = min(int(top_k), int(flat_after.size))
    top_indices = np.argsort(-flat_after)[:safe_top_k]
    feat_size = after.shape[1]
    orig_w, orig_h = original_size
    points: list[RankedPoint] = []
    for rank, patch_index in enumerate(top_indices.tolist(), start=1):
        patch_x = patch_index % feat_size
        patch_y = patch_index // feat_size
        x_518 = int(patch_x * PATCH_SIZE + PATCH_SIZE // 2)
        y_518 = int(patch_y * PATCH_SIZE + PATCH_SIZE // 2)
        x_original = (float(x_518) / IMAGE_SIZE) * float(orig_w)
        y_original = (float(y_518) / IMAGE_SIZE) * float(orig_h)
        before_v = float(flat_before[patch_index])
        after_v = float(flat_after[patch_index])
        points.append(RankedPoint(
            rank=rank,
            patch_index=int(patch_index),
            patch_x=int(patch_x),
            patch_y=int(patch_y),
            x_518=x_518,
            y_518=y_518,
            x_original=round(x_original, 3),
            y_original=round(y_original, 3),
            similarity_before=round(before_v, 6),
            similarity_after=round(after_v, 6),
            delta=round(after_v - before_v, 6),
            selected_by_threshold=after_v >= threshold,
        ))
    return points


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    min_v = float(heatmap.min())
    max_v = float(heatmap.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_v) / (max_v - min_v)


def heatmap_image(target: Image.Image, heatmap: np.ndarray) -> Image.Image:
    gray = Image.fromarray((normalize_heatmap(heatmap) * 255.0).astype(np.uint8)).resize(target.size, Image.BICUBIC)
    color = ImageOps.colorize(gray, black=(0, 0, 40), white=(255, 60, 0), mid=(240, 200, 40))
    return Image.blend(target.convert("RGB"), color.convert("RGB"), alpha=0.55)


def overlay_image(target: Image.Image, points: list[RankedPoint], positive: list[FeedbackPoint], negative: list[FeedbackPoint], threshold: float) -> Image.Image:
    image = target.copy()
    draw = ImageDraw.Draw(image)
    for point in reversed(points):
        x = point.x_original
        y = point.y_original
        radius = 5 if point.selected_by_threshold else 3
        fill = (255, 0, 0) if point.selected_by_threshold else (255, 230, 0)
        outline = (255, 255, 255) if point.selected_by_threshold else (0, 0, 0)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    for point in positive:
        draw.rectangle((point.x - 9, point.y - 9, point.x + 9, point.y + 9), outline=(0, 255, 0), width=4)
    for point in negative:
        draw.line((point.x - 10, point.y - 10, point.x + 10, point.y + 10), fill=(80, 180, 255), width=4)
        draw.line((point.x - 10, point.y + 10, point.x + 10, point.y - 10), fill=(80, 180, 255), width=4)
    legend_y = max(4, image.height - 56)
    draw.rectangle((4, legend_y, min(image.width - 4, 450), image.height - 4), fill=(0, 0, 0))
    draw.text((12, legend_y + 8), f"red >= {threshold} | green positive | blue negative", fill=(255, 255, 255))
    return image


def side_by_side(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    width = left.width + right.width
    height = max(left.height, right.height) + 28
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left, (0, 28))
    canvas.paste(right, (left.width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), left_label, fill=(0, 0, 0))
    draw.text((left.width + 8, 6), right_label, fill=(0, 0, 0))
    return canvas


def point_metrics(
    points: list[RankedPoint],
    positive_points: list[FeedbackPoint],
    negative_points: list[FeedbackPoint],
    radius: float,
) -> dict[str, Any]:
    def nearest_distance(point: RankedPoint, anchors: list[FeedbackPoint]) -> float | None:
        if not anchors:
            return None
        distances = [((point.x_original - anchor.x) ** 2 + (point.y_original - anchor.y) ** 2) ** 0.5 for anchor in anchors]
        return float(min(distances))

    def summarize(subset: list[RankedPoint]) -> dict[str, Any]:
        positive_distances = [nearest_distance(point, positive_points) for point in subset]
        positive_distances = [distance for distance in positive_distances if distance is not None]
        negative_distances = [nearest_distance(point, negative_points) for point in subset]
        negative_distances = [distance for distance in negative_distances if distance is not None]
        return {
            "count": len(subset),
            "mean_distance_to_positive": round(float(np.mean(positive_distances)), 3) if positive_distances else None,
            "median_distance_to_positive": round(float(np.median(positive_distances)), 3) if positive_distances else None,
            "within_positive_radius": sum(1 for distance in positive_distances if distance <= radius),
            "within_negative_radius": sum(1 for distance in negative_distances if distance <= radius),
        }

    selected = [point for point in points if point.selected_by_threshold]
    return {
        "radius": float(radius),
        "top_k": summarize(points),
        "selected": summarize(selected),
    }


def save_feedback_outputs(
    out_dir: Path,
    payload: dict[str, Any],
    before: np.ndarray,
    after: np.ndarray,
    before_points: list[RankedPoint],
    after_points: list[RankedPoint],
    target: Image.Image,
    positive_points: list[FeedbackPoint],
    negative_points: list[FeedbackPoint],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in after_points]
    with (out_dir / "feedback_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(RankedPoint.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    before_selected = sum(1 for point in before_points if point.selected_by_threshold)
    after_selected = sum(1 for point in after_points if point.selected_by_threshold)
    before_metrics = point_metrics(before_points, positive_points, negative_points, args.metric_radius)
    after_metrics = point_metrics(after_points, positive_points, negative_points, args.metric_radius)
    result = {
        "stage1_dir": str(resolve_path(args.stage1_dir).relative_to(WORKSPACE_ROOT)),
        "target_image": payload["target_image"],
        "positive_points": [asdict(point) for point in positive_points],
        "negative_points": [asdict(point) for point in negative_points],
        "sigma": args.sigma,
        "positive_weight": args.positive_weight,
        "negative_weight": args.negative_weight,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "metric_radius": args.metric_radius,
        "before_selected_count": before_selected,
        "after_selected_count": after_selected,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_best": asdict(before_points[0]) if before_points else None,
        "after_best": asdict(after_points[0]) if after_points else None,
        "points": rows,
    }
    (out_dir / "feedback_points.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "similarity_after.npy", after.astype(np.float32))

    before_overlay = overlay_image(target, before_points, positive_points, negative_points, args.threshold)
    after_overlay = overlay_image(target, after_points, positive_points, negative_points, args.threshold)
    side_by_side(before_overlay, after_overlay, "Before feedback", "After feedback").save(out_dir / "feedback_overlay_before_after.png")

    before_heat = heatmap_image(target, before)
    after_heat = heatmap_image(target, after)
    side_by_side(before_heat, after_heat, "Before feedback heatmap", "After feedback heatmap").save(out_dir / "feedback_heatmap_before_after.png")

    summary = [
        "# Matcher Stage 1 Feedback Summary",
        "",
        f"- Stage1 dir: `{result['stage1_dir']}`",
        f"- Positive points: `{result['positive_points']}`",
        f"- Negative points: `{result['negative_points']}`",
        f"- Sigma: `{args.sigma}`",
        f"- Positive weight: `{args.positive_weight}`",
        f"- Negative weight: `{args.negative_weight}`",
        f"- Metric radius: `{args.metric_radius}`",
        f"- Before selected count: `{before_selected}`",
        f"- After selected count: `{after_selected}`",
        f"- Before top-k metrics: `{before_metrics['top_k']}`",
        f"- After top-k metrics: `{after_metrics['top_k']}`",
        f"- Before selected metrics: `{before_metrics['selected']}`",
        f"- After selected metrics: `{after_metrics['selected']}`",
        f"- Before best: `{result['before_best']}`",
        f"- After best: `{result['after_best']}`",
        "",
        "## Artifacts",
        "",
        "- `feedback_points.csv`",
        "- `feedback_points.json`",
        "- `feedback_overlay_before_after.png`",
        "- `feedback_heatmap_before_after.png`",
        "- `similarity_after.npy`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply positive/negative feedback to Matcher Stage 1 similarity.")
    parser.add_argument("--stage1-dir", default="runs/research_direction/stage1_matching/elephant_pair")
    parser.add_argument("--positive-points", default="268,331")
    parser.add_argument("--negative-points", default="30,430;455,420")
    parser.add_argument("--sigma", type=float, default=42.0)
    parser.add_argument("--positive-weight", type=float, default=0.18)
    parser.add_argument("--negative-weight", type=float, default=0.22)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--metric-radius", type=float, default=70.0)
    parser.add_argument("--out-dir", default="runs/research_direction/stage1_feedback/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_dir = resolve_path(args.stage1_dir)
    out_dir = resolve_path(args.out_dir)
    payload, before, target = load_stage1(stage1_dir)
    positive_points = parse_points(args.positive_points)
    negative_points = parse_points(args.negative_points)
    after = apply_feedback(
        similarity=before,
        positive_points=positive_points,
        negative_points=negative_points,
        original_size=target.size,
        sigma=args.sigma,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
    )
    before_points = rank_points(before, before, target.size, args.top_k, args.threshold)
    after_points = rank_points(before, after, target.size, args.top_k, args.threshold)
    save_feedback_outputs(
        out_dir=out_dir,
        payload=payload,
        before=before,
        after=after,
        before_points=before_points,
        after_points=after_points,
        target=target,
        positive_points=positive_points,
        negative_points=negative_points,
        args=args,
    )
    print(f"Before selected count: {sum(1 for point in before_points if point.selected_by_threshold)}")
    print(f"After selected count: {sum(1 for point in after_points if point.selected_by_threshold)}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
