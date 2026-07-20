from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
MATCHER_ROOT = WORKSPACE_ROOT / "Matcher"
SAM2_ROOT = WORKSPACE_ROOT / "sam2"
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_ROOT / ".cache" / "matplotlib_monuseg_dense"))
for root in (CODE_ROOT, MATCHER_ROOT, SAM2_ROOT):
    if not root.exists():
        continue
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform  # noqa: E402
from dinov2.models import vision_transformer as vits  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


IMAGE_SIZE = 518
PATCH_SIZE = 14


@dataclass
class Point:
    rank: int
    patch_index: int
    patch_x: int
    patch_y: int
    x_518: int
    y_518: int
    x: float
    y: float
    score: float
    s_fg: float
    s_bg: float


@dataclass
class CandidateResult:
    case_name: str
    query_case_id: str
    round_name: str
    method: str
    matching: str
    layout: str
    clustering: str
    lambda_bg: float
    k: int
    prompt_count: int
    sam_score: float
    dice: float
    iou: float
    fg_precision: float
    fg_recall: float
    object_recall: float
    object_precision: float
    gt_components: int
    pred_components: int
    component_count_error: float
    merge_error: float
    pred_area_pct: float
    mask_path: str
    overlay_path: str
    points_path: str


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_manifest_rows(
    rows: list[dict[str, str]],
    dataset_filter: str,
    role_filter: str = "",
) -> list[dict[str, str]]:
    wanted_dataset = dataset_filter.strip().lower()
    wanted_role = role_filter.strip().lower()
    return [
        row for row in rows
        if (not wanted_dataset or row.get("dataset", "").lower() == wanted_dataset)
        and (not wanted_role or row.get("role", "").lower() == wanted_role)
    ]


def should_refresh_query_embedding(previous_query_id: str | None, current_query_id: str) -> bool:
    return previous_query_id != current_query_id


def cached_value(cache: dict[str, Any], key: str, factory: Any) -> Any:
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def visual_artifacts_enabled(minimal_artifacts: bool) -> bool:
    return not minimal_artifacts


def requested_rounds(skip_round3: bool) -> tuple[str, ...]:
    return ("round1", "round2") if skip_round3 else ("round1", "round2", "round3")


def runtime_metrics(
    elapsed_seconds: float,
    candidate_count: int,
    case_count: int,
    peak_gpu_mb: float,
) -> dict[str, float | int | None]:
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "runtime_candidate_count": int(candidate_count),
        "runtime_episode_count": int(case_count),
        "seconds_per_candidate": elapsed_seconds / candidate_count if candidate_count else None,
        "seconds_per_episode": elapsed_seconds / case_count if case_count else None,
        "peak_gpu_memory_mb": float(peak_gpu_mb),
    }


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        if size is not None and img.size != size:
            img = img.resize(size, Image.NEAREST)
        arr = np.asarray(img)
    if arr.ndim == 3:
        return np.any(arr > 0, axis=2)
    return arr > 0


def load_mask_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image)


def clean_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_dinov2(weights: Path, size: str, device: torch.device) -> torch.nn.Module:
    kwargs = dict(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    if size not in vits.__dict__:
        raise ValueError(f"Unknown DINOv2 size: {size}")
    model = vits.__dict__[size](**kwargs)
    state_dict = torch.load(weights, map_location="cpu")
    if isinstance(state_dict, dict) and "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    state_dict = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def extract_features(model: torch.nn.Module, image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    features = model.forward_features(tensor)["x_prenorm"][:, 1:]
    features = features.reshape(-1, model.embed_dim)
    return F.normalize(features, dim=1, p=2)


def patch_weights(mask: Image.Image, foreground: bool, device: torch.device) -> torch.Tensor:
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    if not foreground:
        mask_np = 1.0 - mask_np
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    pooled = F.avg_pool2d(mask_t, (PATCH_SIZE, PATCH_SIZE)).reshape(-1)
    keep = pooled > 0
    if float(pooled[keep].sum().detach().cpu()) <= 0.0:
        raise ValueError("Reference mask produced no usable foreground/background patches")
    weights = torch.zeros_like(pooled)
    weights[keep] = pooled[keep] / pooled[keep].sum()
    return weights


def weighted_proto(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    proto = (features * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    std = float(values.std())
    if std < 1e-8:
        return values * 0.0
    return (values - float(values.mean())) / std


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def dense_points_from_score(
    score_map: np.ndarray,
    s_fg: np.ndarray,
    s_bg: np.ndarray,
    original_size: tuple[int, int],
    max_points: int,
) -> list[Point]:
    feat_h, feat_w = score_map.shape
    flat_score = score_map.reshape(-1)
    flat_fg = s_fg.reshape(-1)
    flat_bg = s_bg.reshape(-1)
    order = np.argsort(-flat_score)[:max_points]
    target_w, target_h = original_size
    points: list[Point] = []
    for rank, idx in enumerate(order.tolist(), start=1):
        py = int(idx // feat_w)
        px = int(idx % feat_w)
        x_518 = int(px * PATCH_SIZE + PATCH_SIZE // 2)
        y_518 = int(py * PATCH_SIZE + PATCH_SIZE // 2)
        points.append(
            Point(
                rank=rank,
                patch_index=int(idx),
                patch_x=px,
                patch_y=py,
                x_518=x_518,
                y_518=y_518,
                x=round((x_518 / IMAGE_SIZE) * target_w, 3),
                y=round((y_518 / IMAGE_SIZE) * target_h, 3),
                score=round(float(flat_score[idx]), 6),
                s_fg=round(float(flat_fg[idx]), 6),
                s_bg=round(float(flat_bg[idx]), 6),
            )
        )
    return points


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["case_name"]), str(row["round_name"]), str(row["method"])


def merge_unique_candidate_rows(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {candidate_row_key(row): row for row in existing}
    merged.update({candidate_row_key(row): row for row in new_rows})
    return [merged[key] for key in sorted(merged)]


def completed_case_names(
    rows: list[dict[str, Any]],
    round_name: str,
    expected_candidates: int,
) -> set[str]:
    methods_by_case: dict[str, set[str]] = {}
    for row in rows:
        if str(row.get("round_name")) != round_name:
            continue
        methods_by_case.setdefault(str(row["case_name"]), set()).add(str(row["method"]))
    return {
        case_name
        for case_name, methods in methods_by_case.items()
        if len(methods) == expected_candidates
    }


def append_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def save_points(path: Path, points: list[Point]) -> None:
    save_csv(path, [asdict(point) for point in points], list(Point.__dataclass_fields__.keys()))


def save_heatmap(target: Image.Image, heatmap: np.ndarray, path: Path, palette: str = "hot") -> None:
    norm = normalize01(heatmap)
    gray = Image.fromarray((norm * 255.0).astype(np.uint8)).resize(target.size, Image.BICUBIC)
    if palette == "blue":
        color = ImageOps.colorize(gray, black=(0, 8, 30), white=(40, 180, 255), mid=(45, 90, 150))
    elif palette == "green":
        color = ImageOps.colorize(gray, black=(0, 20, 6), white=(80, 255, 110), mid=(60, 150, 70))
    else:
        color = ImageOps.colorize(gray, black=(0, 0, 40), white=(255, 45, 20), mid=(255, 220, 40))
    Image.blend(target.convert("RGB"), color.convert("RGB"), alpha=0.58).save(path)


def save_point_overlay(target: Image.Image, points: list[Point], path: Path, title: str, color: tuple[int, int, int]) -> None:
    image = target.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for point in reversed(points):
        radius = 5 if point.rank <= 20 else 3
        x = float(point.x)
        y = float(point.y)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)
    draw.rectangle((4, image.height - 40, min(image.width - 4, 760), image.height - 4), fill=(0, 0, 0))
    draw.text((12, image.height - 30), title, fill=(255, 255, 255))
    image.save(path)


def point_array(points: list[Point]) -> np.ndarray:
    return np.asarray([[point.x, point.y] for point in points], dtype=np.float32)


def unique_points(points: Iterable[Point]) -> list[Point]:
    seen: set[tuple[int, int]] = set()
    out: list[Point] = []
    for point in points:
        key = (int(round(point.x)), int(round(point.y)))
        if key in seen:
            continue
        seen.add(key)
        out.append(point)
    return out


def topk(points: list[Point], k: int) -> list[Point]:
    return unique_points(points)[:k]


def diverse_topk(points: list[Point], k: int, min_distance: float = 18.0) -> list[Point]:
    selected: list[Point] = []
    for point in points:
        if all(math.hypot(point.x - kept.x, point.y - kept.y) >= min_distance for kept in selected):
            selected.append(point)
        if len(selected) >= k:
            return selected
    return unique_points(selected + points)[:k]


def local_maxima_points(
    score_map: np.ndarray,
    s_fg: np.ndarray,
    s_bg: np.ndarray,
    original_size: tuple[int, int],
    k: int,
) -> list[Point]:
    padded = np.pad(score_map, 1, mode="edge")
    keep: list[tuple[float, int, int]] = []
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            patch = padded[y : y + 3, x : x + 3]
            value = float(score_map[y, x])
            if value >= float(patch.max()):
                keep.append((value, x, y))
    keep.sort(reverse=True)
    target_w, target_h = original_size
    points: list[Point] = []
    for rank, (_, px, py) in enumerate(keep[: max(k * 3, k)], start=1):
        x_518 = int(px * PATCH_SIZE + PATCH_SIZE // 2)
        y_518 = int(py * PATCH_SIZE + PATCH_SIZE // 2)
        idx = int(py * score_map.shape[1] + px)
        points.append(
            Point(
                rank=rank,
                patch_index=idx,
                patch_x=px,
                patch_y=py,
                x_518=x_518,
                y_518=y_518,
                x=round((x_518 / IMAGE_SIZE) * target_w, 3),
                y=round((y_518 / IMAGE_SIZE) * target_h, 3),
                score=round(float(score_map[py, px]), 6),
                s_fg=round(float(s_fg[py, px]), 6),
                s_bg=round(float(s_bg[py, px]), 6),
            )
        )
    return unique_points(points)[:k]


def boundary_mix(points: list[Point], score_map: np.ndarray, k: int) -> list[Point]:
    gy, gx = np.gradient(normalize01(score_map))
    grad = np.sqrt(gx * gx + gy * gy)
    boundary_rank = sorted(points, key=lambda p: float(grad[p.patch_y, p.patch_x]) * (1.0 + p.score), reverse=True)
    interior_n = max(1, int(round(k * 0.7)))
    boundary_n = max(0, k - interior_n)
    return unique_points(topk(points, interior_n) + boundary_rank[:boundary_n] + points)[:k]


def dbscan_medoid(points: list[Point], k: int, eps: float = 22.0) -> list[Point]:
    pool = topk(points, max(k * 4, 80))
    if not pool:
        return []
    coords = point_array(pool)
    visited = np.zeros(len(pool), dtype=bool)
    clusters: list[list[int]] = []
    for i in range(len(pool)):
        if visited[i]:
            continue
        distances = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        members = np.where(distances <= eps)[0].tolist()
        visited[members] = True
        clusters.append(members)
    representatives: list[Point] = []
    for members in clusters:
        representatives.append(min((pool[i] for i in members), key=lambda p: p.rank))
    representatives.sort(key=lambda p: p.rank)
    return unique_points(representatives + points)[:k]


def heatmap_cc_points(
    score_map: np.ndarray,
    s_fg: np.ndarray,
    s_bg: np.ndarray,
    original_size: tuple[int, int],
    k: int,
) -> list[Point]:
    threshold = float(np.quantile(score_map, 0.88))
    binary = score_map >= threshold
    labels, count = connected_components(binary)
    target_w, target_h = original_size
    reps: list[Point] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        values = score_map[ys, xs]
        best = int(np.argmax(values))
        px = int(xs[best])
        py = int(ys[best])
        x_518 = int(px * PATCH_SIZE + PATCH_SIZE // 2)
        y_518 = int(py * PATCH_SIZE + PATCH_SIZE // 2)
        reps.append(
            Point(
                rank=len(reps) + 1,
                patch_index=int(py * score_map.shape[1] + px),
                patch_x=px,
                patch_y=py,
                x_518=x_518,
                y_518=y_518,
                x=round((x_518 / IMAGE_SIZE) * target_w, 3),
                y=round((y_518 / IMAGE_SIZE) * target_h, 3),
                score=round(float(score_map[py, px]), 6),
                s_fg=round(float(s_fg[py, px]), 6),
                s_bg=round(float(s_bg[py, px]), 6),
            )
        )
    reps.sort(key=lambda p: p.score, reverse=True)
    return unique_points(reps)[:k]


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        import cv2  # type: ignore

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32), int(count - 1)
    except Exception:
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label = 0
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or labels[y, x] != 0:
                    continue
                label += 1
                stack = [(y, x)]
                labels[y, x] = label
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
        return labels, label


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    dice = 2.0 * inter / (pred_sum + target_sum) if pred_sum + target_sum else 1.0
    iou = inter / union if union else 1.0
    return dice, iou


def object_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    gt_labels, gt_count = connected_components(target)
    pred_labels, pred_count = connected_components(pred)
    overlap = np.logical_and(pred, target)
    gt_hit = 0
    for label in range(1, gt_count + 1):
        gt_region = gt_labels == label
        if float(np.logical_and(gt_region, pred).sum()) / max(1.0, float(gt_region.sum())) >= 0.10:
            gt_hit += 1
    pred_hit = 0
    merge = 0
    for label in range(1, pred_count + 1):
        pred_region = pred_labels == label
        pred_area = max(1.0, float(pred_region.sum()))
        if float(np.logical_and(pred_region, target).sum()) / pred_area >= 0.10:
            pred_hit += 1
        touched = np.unique(gt_labels[np.logical_and(pred_region, target)])
        touched = [int(item) for item in touched.tolist() if int(item) > 0]
        if len(touched) >= 2:
            merge += 1
    return {
        "object_recall": gt_hit / gt_count if gt_count else 1.0,
        "object_precision": pred_hit / pred_count if pred_count else 0.0,
        "gt_components": float(gt_count),
        "pred_components": float(pred_count),
        "component_count_error": abs(pred_count - gt_count) / max(1.0, float(gt_count)),
        "merge_error": merge / pred_count if pred_count else 0.0,
        "overlap_px": float(overlap.sum()),
    }


def predict_sam(predictor: SamPredictor, points: list[Point]) -> tuple[np.ndarray, float]:
    coords = point_array(points)
    labels = np.ones((len(coords),), dtype=np.int32)
    masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def predict_sam2(predictor: Any, image: Image.Image, points: list[Point], autocast_context: Any) -> tuple[np.ndarray, float]:
    coords = point_array(points)
    labels = np.ones((len(coords),), dtype=np.int32)
    with torch.inference_mode(), autocast_context:
        predictor.set_image(np.asarray(image.convert("RGB")))
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def predict_sam3(segmenter: Any, image: Image.Image, points: list[Point]) -> tuple[np.ndarray, float]:
    from demos.medical_sam_click_app import PointPrompt  # noqa: E402

    prompts = [PointPrompt(x=int(round(point.x)), y=int(round(point.y)), label=1) for point in points]
    masks, scores = segmenter.predict_interactive(
        image=np.asarray(image.convert("RGB")),
        prompt_type="click",
        points=prompts,
        box=None,
        mode="standard",
    )
    if len(masks) == 0:
        return np.zeros((image.height, image.width), dtype=bool), 0.0
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def save_overlay(image: Image.Image, gt: np.ndarray, pred: np.ndarray, points: list[Point], path: Path, title: str) -> None:
    base = image.convert("RGBA")
    gt_layer = Image.new("RGBA", base.size, (0, 180, 255, 0))
    pred_layer = Image.new("RGBA", base.size, (255, 40, 40, 0))
    gt_layer.putalpha(Image.fromarray((gt.astype(np.uint8) * 70), mode="L"))
    pred_layer.putalpha(Image.fromarray((pred.astype(np.uint8) * 105), mode="L"))
    merged = Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer).convert("RGB")
    draw = ImageDraw.Draw(merged)
    bbox = mask_bbox(pred)
    if bbox != [0, 0, 0, 0]:
        draw.rectangle(tuple(bbox), outline=(255, 230, 0), width=2)
    for point in points:
        x = float(point.x)
        y = float(point.y)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 255, 80), outline=(255, 255, 255), width=1)
    draw.rectangle((4, merged.height - 42, min(merged.width - 4, 980), merged.height - 4), fill=(0, 0, 0))
    draw.text((12, merged.height - 32), title, fill=(255, 255, 255))
    merged.save(path)


def load_sam_predictor(checkpoint: Path, model_type: str, device_text: str) -> tuple[SamPredictor, torch.device]:
    device = clean_device(device_text)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam), device


def load_sam3_segmenter(checkpoint: Path, device_text: str, threshold: float) -> tuple[Any, str]:
    from demos.medical_sam_click_app import Sam3TextSegmenter  # noqa: E402

    device = str(clean_device(device_text))
    segmenter = Sam3TextSegmenter(checkpoint=checkpoint, device=device, threshold=threshold)
    return segmenter, device


def load_sam2_predictor(checkpoint: Path, config: str, device_text: str) -> tuple[Any, torch.device, Any]:
    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

    device = clean_device(device_text)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )
    model = build_sam2(config, str(checkpoint), device=device)
    return SAM2ImagePredictor(model), device, autocast_context


def compute_matching_maps(
    model: torch.nn.Module,
    ref_image: Image.Image,
    ref_mask: Image.Image,
    query_image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ref_features = extract_features(model, ref_image, device)
    query_features = extract_features(model, query_image, device)
    return compute_matching_maps_from_features(ref_features, ref_mask, query_features, device)


def compute_matching_maps_from_features(
    ref_features: torch.Tensor,
    ref_mask: Image.Image,
    query_features: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    fg_weights = patch_weights(ref_mask, foreground=True, device=device)
    bg_weights = patch_weights(ref_mask, foreground=False, device=device)
    fg_proto = weighted_proto(ref_features, fg_weights)
    bg_proto = weighted_proto(ref_features, bg_weights)
    s_fg = (fg_proto @ query_features.t()).squeeze(0).reshape(IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)
    s_bg = (bg_proto @ query_features.t()).squeeze(0).reshape(IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)
    return s_fg.detach().cpu().numpy(), s_bg.detach().cpu().numpy()


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def candidate_specs(
    round_name: str,
    best_k: int,
    best_lambda: float,
    k_list: list[int],
    lambda_list: list[float],
    round2_k_list: list[int],
) -> list[dict[str, Any]]:
    if round_name == "round1":
        return [
            {"round_name": "round1", "method": f"fg_topk_K{k}", "matching": "foreground_only", "layout": "topk", "clustering": "none", "lambda_bg": 0.0, "k": k}
            for k in k_list
        ]
    if round_name == "round2":
        return [
            {
                "round_name": "round2",
                "method": f"contrast_topk_K{k}_lam{lam:g}",
                "matching": "fg_minus_bg",
                "layout": "topk",
                "clustering": "none",
                "lambda_bg": lam,
                "k": k,
            }
            for k in (round2_k_list or [best_k])
            for lam in lambda_list
        ]
    return [
        {"round_name": "round3", "method": "best_topk", "matching": "fg_minus_bg", "layout": "topk", "clustering": "none", "lambda_bg": best_lambda, "k": best_k},
        {"round_name": "round3", "method": "diverse_topk", "matching": "fg_minus_bg", "layout": "diverse_topk", "clustering": "none", "lambda_bg": best_lambda, "k": best_k},
        {"round_name": "round3", "method": "local_maxima", "matching": "fg_minus_bg", "layout": "local_maxima", "clustering": "none", "lambda_bg": best_lambda, "k": best_k},
        {"round_name": "round3", "method": "boundary_mix", "matching": "fg_minus_bg", "layout": "boundary_mix", "clustering": "none", "lambda_bg": best_lambda, "k": best_k},
        {"round_name": "round3", "method": "dbscan_medoid", "matching": "fg_minus_bg", "layout": "cluster_medoid", "clustering": "dbscan", "lambda_bg": best_lambda, "k": best_k},
        {"round_name": "round3", "method": "heatmap_cc", "matching": "fg_minus_bg", "layout": "heatmap_cc", "clustering": "connected_components", "lambda_bg": best_lambda, "k": best_k},
    ]


def choose_points(
    spec: dict[str, Any],
    score_map: np.ndarray,
    s_fg: np.ndarray,
    s_bg: np.ndarray,
    original_size: tuple[int, int],
    pool: list[Point],
) -> list[Point]:
    k = int(spec["k"])
    layout = str(spec["layout"])
    if layout == "topk":
        return topk(pool, k)
    if layout == "diverse_topk":
        return diverse_topk(pool, k)
    if layout == "local_maxima":
        return local_maxima_points(score_map, s_fg, s_bg, original_size, k)
    if layout == "boundary_mix":
        return boundary_mix(pool, score_map, k)
    if layout == "cluster_medoid":
        return dbscan_medoid(pool, k)
    if layout == "heatmap_cc":
        selected = heatmap_cc_points(score_map, s_fg, s_bg, original_size, k)
        return unique_points(selected + pool)[:k]
    raise ValueError(f"Unknown point layout: {layout}")


def normalize_group_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return value


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(normalize_group_value(row[k]) for k in group_keys)
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda kv: kv[0]):
        item = {k: v for k, v in zip(group_keys, key)}
        for metric in ["dice", "iou", "fg_precision", "fg_recall", "object_recall", "object_precision", "component_count_error", "merge_error", "sam_score", "pred_area_pct"]:
            values = [float(row[metric]) for row in items]
            item[f"mean_{metric}"] = round(float(np.mean(values)), 6)
            item[f"std_{metric}"] = round(float(np.std(values)), 6)
        item["case_count"] = len(items)
        output.append(item)
    return output


def best_value(rows: list[dict[str, Any]], key: str, value_key: str) -> Any:
    best = max(rows, key=lambda row: float(row[value_key]))
    return best[key]


def select_round3_config(
    round1_summary: list[dict[str, Any]],
    round2_summary: list[dict[str, Any]],
    fixed_k: int = 0,
    fixed_lambda: float | None = None,
) -> tuple[int, float, str]:
    if fixed_k > 0 or fixed_lambda is not None:
        if fixed_k <= 0 or fixed_lambda is None:
            raise ValueError("--round3-fixed-k and --round3-fixed-lambda must be supplied together.")
        return fixed_k, float(fixed_lambda), "external_fixed"
    best_k = int(best_value(round1_summary, "k", "mean_dice"))
    best_round2 = max(round2_summary, key=lambda row: float(row["mean_dice"]))
    return int(best_round2.get("k", best_k)), float(best_round2["lambda_bg"]), "manifest_oracle"


def table_md(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def generate_report(out_dir: Path, rows: list[dict[str, Any]], round1: list[dict[str, Any]], round2: list[dict[str, Any]], round3: list[dict[str, Any]], examples: list[dict[str, str]]) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    best_k = best_value(round1, "k", "mean_dice")
    best_lambda = best_value(round2, "lambda_bg", "mean_dice")
    best_layout = best_value(round3, "method", "mean_dice")
    lines: list[str] = [
        "# MonuSeg 单域 Dense Point Prompt 闭环实验报告",
        "",
        "## 结论摘要",
        "",
        f"- 本轮只做 MonuSeg 单域，固定 support 为人工 mask，query 为现有 MonuSeg case，共评估 `{len({r['case_name'] for r in rows})}` 个 case。",
        f"- T1 完成：每个 case 输出 foreground 相似度、background 相似度、foreground-background contrastive score、top-k 点图。",
        f"- T2 完成：将不同点预算和点布局输入 SAM，所有候选都保存 mask、overlay、points 和指标。",
        f"- T3 结果：Round 1 最优点数 K 为 `{best_k}`。",
        f"- T4 结果：Round 2 最优背景抑制系数 lambda 为 `{best_lambda}`。",
        f"- T5 结果：Round 3 当前最优布局/聚类策略为 `{best_layout}`。",
        "",
        "这份实验不是大 benchmark，而是最小闭环：从 reference mask 产生 matched points，进入 SAM，多指标评价，再把可解释的变量填回表格。",
        "",
        "## T1：前景-背景对比 Matching",
        "",
        "对每个 query，先用 reference mask 内的 DINO feature 求 foreground prototype，再用 reference mask 外的 feature 求 background prototype。打分公式是：",
        "",
        "```text",
        "score(x) = z(sim(query_x, foreground_proto)) - lambda * z(sim(query_x, background_proto))",
        "```",
        "",
        "直白解释：我们找的不是单纯“像细胞核”的点，而是“像 reference 细胞核、同时不像 reference 背景”的点。",
        "",
        "## T3：点数预算实验",
        "",
        "固定 foreground-only matching、top-k、无聚类，只改变一次性送进 SAM 的正点数量。",
        "",
        *table_md(round1, ["k", "case_count", "mean_dice", "mean_iou", "mean_fg_recall", "mean_object_recall", "mean_merge_error", "mean_pred_area_pct"]),
        "",
        "## T4：前景-背景对比实验",
        "",
        f"固定 K=`{best_k}` 和 top-k 布局，只改变背景抑制强度 lambda。",
        "",
        *table_md(round2, ["lambda_bg", "case_count", "mean_dice", "mean_iou", "mean_fg_recall", "mean_object_recall", "mean_merge_error", "mean_pred_area_pct"]),
        "",
        "## T5：点布局与聚类对照",
        "",
        f"固定 K=`{best_k}`、lambda=`{best_lambda}`，比较 top-k、空间分散、局部峰值、边界混合、DBSCAN medoid 和 heatmap connected components。",
        "",
        *table_md(round3, ["method", "layout", "clustering", "case_count", "mean_dice", "mean_iou", "mean_fg_recall", "mean_object_recall", "mean_merge_error", "mean_pred_area_pct"]),
        "",
        "## 定性图索引",
        "",
        "图中蓝色为 GT mask，红色为 SAM 预测 mask，绿色点为本策略送入 SAM 的 positive prompt，黄色框只是预测 mask 外接框，方便定位预测范围。",
        "",
    ]
    for item in examples:
        lines.extend([
            f"### {item['title']}",
            "",
            f"- case: `{item['case_name']}`",
            f"- method: `{item['method']}`",
            f"- Dice: `{item['dice']}`，IoU: `{item['iou']}`，object recall: `{item['object_recall']}`",
            f"- overlay: `{item['overlay_path']}`",
            "",
            f"![{item['title']}]({item['overlay_rel_to_report']})",
            "",
        ])
    lines.extend([
        "## 下一步",
        "",
        "1. 如果 dense K 明显优于少量点，下一步应围绕“如何预测最合适 K”做学习，而不是先做复杂 proposal scorer。",
        "2. 如果 foreground-background contrastive 有提升，下一步可把 lambda 从手调改成验证集学习，或学习 foreground/background/edge 的线性组合。",
        "3. 如果无聚类优于 DBSCAN，说明 MonuSeg 这种 all-nuclei task 不应强制 instance-level clustering；如果聚类有提升，再继续做更细的聚类算法。",
        "4. 本轮仍然只使用 positive points。若过分扩张或 merge error 高，再加入 negative/background points 做第二阶段反馈。",
        "",
        "## 产物",
        "",
        f"- 全量候选表：`{rel(out_dir / 'candidates.csv')}`",
        f"- Round 1 表：`{rel(out_dir / 'round1_prompt_budget.csv')}`",
        f"- Round 2 表：`{rel(out_dir / 'round2_contrastive_matching.csv')}`",
        f"- Round 3 表：`{rel(out_dir / 'round3_layout_clustering.csv')}`",
        f"- 每个 case 的热力图、点图、mask 和 overlay：`{rel(out_dir / 'cases')}`",
    ])
    (report_dir / "monuseg_dense_prompt_report.md").write_text("\n".join(lines), encoding="utf-8")
    html = markdown_to_html("\n".join(lines), title="MonuSeg Dense Point Prompt Report")
    (report_dir / "index.html").write_text(html, encoding="utf-8")


def markdown_to_html(markdown: str, title: str) -> str:
    body_lines: list[str] = []
    in_code = False
    table: list[str] = []

    def flush_table() -> None:
        nonlocal table
        if not table:
            return
        headers = [cell.strip() for cell in table[0].strip("|").split("|")]
        body_lines.append("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>")
        for row in table[2:]:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            body_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        body_lines.append("</tbody></table>")
        table = []

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("| "):
            table.append(line)
            continue
        flush_table()
        if line.startswith("```"):
            body_lines.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
        elif in_code:
            body_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        elif line.startswith("# "):
            body_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            body_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            body_lines.append(f"<p class='bullet'>{line[2:]}</p>")
        elif line.startswith("!["):
            alt = line.split("](", 1)[0][2:]
            src = line.split("](", 1)[1].rstrip(")")
            body_lines.append(f"<a href='{src}' target='_blank'><img src='{src}' alt='{alt}' loading='lazy'></a>")
        elif line:
            body_lines.append(f"<p>{line}</p>")
        else:
            body_lines.append("")
    flush_table()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #f6f8fb; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 32px; margin: 0 0 20px; }}
    h2 {{ font-size: 22px; margin: 34px 0 12px; padding-top: 12px; border-top: 1px solid #d7dee8; }}
    h3 {{ font-size: 18px; margin: 26px 0 10px; }}
    p {{ line-height: 1.72; margin: 8px 0; }}
    .bullet::before {{ content: "• "; color: #2b6cb0; font-weight: 700; }}
    pre {{ background: #101828; color: #dbeafe; padding: 14px 16px; border-radius: 8px; overflow:auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin: 12px 0 22px; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    th, td {{ border: 1px solid #d9e2ec; padding: 9px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eaf1f8; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; background: white; border-radius: 6px; margin: 8px 0 18px; }}
    a {{ color: #1f5ea8; }}
  </style>
</head>
<body><main>
{chr(10).join(body_lines)}
</main></body></html>
"""


def run(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = filter_manifest_rows(read_manifest(resolve_path(args.manifest)), args.dataset_filter, args.role_filter)
    if args.case_limit > 0:
        manifest = manifest[: args.case_limit]
    if not manifest:
        raise ValueError(
            f"No rows found in manifest for dataset filter '{args.dataset_filter}' and role filter '{args.role_filter}'."
        )

    dino_device = clean_device(args.dino_device)
    sam2_autocast_context = contextlib.nullcontext()
    if args.segmenter == "sam3":
        prompt_predictor, sam_device = load_sam3_segmenter(resolve_path(args.sam3_checkpoint), args.sam_device, args.sam3_threshold)
    elif args.segmenter == "sam2":
        prompt_predictor, sam_device, sam2_autocast_context = load_sam2_predictor(resolve_path(args.sam2_checkpoint), args.sam2_config, args.sam_device)
    else:
        prompt_predictor, sam_device = load_sam_predictor(resolve_path(args.sam_checkpoint), args.sam_model_type, args.sam_device)
    model = load_dinov2(resolve_path(args.dino_weights), args.dino_size, dino_device)
    k_list = parse_int_list(args.k_list)
    lambda_list = parse_float_list(args.lambda_list)
    round2_k_list = parse_int_list(args.round2_k_list) if args.round2_k_list else []
    round_names = requested_rounds(args.skip_round3)
    write_artifacts = visual_artifacts_enabled(args.minimal_artifacts)
    print(f"[init] segmenter={args.segmenter} cases={len(manifest)} dino_device={dino_device} sam_device={sam_device}")

    candidate_fields = list(CandidateResult.__dataclass_fields__.keys())
    progress_path = out_dir / "candidates_progress.csv"
    if args.resume and progress_path.exists():
        all_rows: list[dict[str, Any]] = read_manifest(progress_path)
        print(f"[resume] loaded {len(all_rows)} checkpoint candidates")
    else:
        all_rows = []
        save_csv(progress_path, [], candidate_fields)
    initial_candidate_keys = {
        (str(row["case_name"]), str(row["round_name"]), str(row["method"]))
        for row in all_rows
    }
    case_cache: dict[str, dict[str, Any]] = {}
    dino_feature_cache: dict[str, torch.Tensor] = {}

    for idx, row in enumerate(manifest, start=1):
        case_name = row["pilot_id"]
        case_dir = out_dir / "cases" / case_name
        stage1_dir = case_dir / "t1_matching"
        mask_dir = case_dir / "t2_masks"
        overlay_dir = case_dir / "t2_overlays"
        points_dir = case_dir / "points"
        for path in (stage1_dir, mask_dir, overlay_dir, points_dir):
            path.mkdir(parents=True, exist_ok=True)

        support_image = load_rgb(resolve_path(row["support_image_path"]))
        query_image = load_rgb(resolve_path(row["query_image_path"]))
        support_resized = support_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        support_mask = load_mask_image(resolve_path(row["support_mask_path"]))
        query_gt = load_mask(resolve_path(row["query_mask_path"]), size=query_image.size)

        support_key = str(resolve_path(row["support_image_path"]))
        query_key = str(resolve_path(row["query_image_path"]))
        support_features = cached_value(
            dino_feature_cache,
            support_key,
            lambda image=support_resized: extract_features(model, image, dino_device),
        )
        query_features = cached_value(
            dino_feature_cache,
            query_key,
            lambda image=query_resized: extract_features(model, image, dino_device),
        )
        s_fg, s_bg = compute_matching_maps_from_features(
            support_features,
            support_mask,
            query_features,
            dino_device,
        )
        np.save(stage1_dir / "s_fg.npy", s_fg)
        np.save(stage1_dir / "s_bg.npy", s_bg)
        if write_artifacts:
            save_heatmap(query_image, s_fg, stage1_dir / "s_fg_heatmap.png", "green")
            save_heatmap(query_image, s_bg, stage1_dir / "s_bg_heatmap.png", "blue")

        case_cache[case_name] = {
            "row": row,
            "query_image": query_image,
            "query_gt": query_gt,
            "s_fg": s_fg,
            "s_bg": s_bg,
            "stage1_dir": stage1_dir,
            "mask_dir": mask_dir,
            "overlay_dir": overlay_dir,
            "points_dir": points_dir,
        }
        print(f"[T1] {idx}/{len(manifest)} {case_name} matching maps ready")

    round1_expected = len(candidate_specs("round1", 20, 0.0, k_list, lambda_list, round2_k_list))
    round1_new = run_round(
        case_cache,
        prompt_predictor,
        args.segmenter,
        "round1",
        best_k=20,
        best_lambda=0.0,
        max_points=args.max_points,
        k_list=k_list,
        lambda_list=lambda_list,
        round2_k_list=round2_k_list,
        sam2_autocast_context=sam2_autocast_context,
        write_artifacts=write_artifacts,
        completed_cases=completed_case_names(all_rows, "round1", round1_expected),
        progress_path=progress_path,
    )
    all_rows = merge_unique_candidate_rows(all_rows, round1_new)
    round1_rows = [row for row in all_rows if row["round_name"] == "round1"]
    round1_summary = summarize(round1_rows, ["k"])
    best_k = int(best_value(round1_summary, "k", "mean_dice"))

    if not round2_k_list:
        round2_k_list = [best_k]
    round2_expected = len(candidate_specs("round2", best_k, 0.0, k_list, lambda_list, round2_k_list))
    round2_new = run_round(
        case_cache,
        prompt_predictor,
        args.segmenter,
        "round2",
        best_k=best_k,
        best_lambda=0.0,
        max_points=args.max_points,
        k_list=k_list,
        lambda_list=lambda_list,
        round2_k_list=round2_k_list,
        sam2_autocast_context=sam2_autocast_context,
        write_artifacts=write_artifacts,
        completed_cases=completed_case_names(all_rows, "round2", round2_expected),
        progress_path=progress_path,
    )
    all_rows = merge_unique_candidate_rows(all_rows, round2_new)
    round2_rows = [row for row in all_rows if row["round_name"] == "round2"]
    round2_summary = summarize(round2_rows, ["k", "lambda_bg"])
    best_k, best_lambda, round3_config_source = select_round3_config(
        round1_summary,
        round2_summary,
        fixed_k=args.round3_fixed_k,
        fixed_lambda=args.round3_fixed_lambda,
    )

    if "round3" in round_names:
        round3_expected = len(candidate_specs("round3", best_k, best_lambda, k_list, lambda_list, round2_k_list))
        round3_new = run_round(
            case_cache,
            prompt_predictor,
            args.segmenter,
            "round3",
            best_k=best_k,
            best_lambda=best_lambda,
            max_points=args.max_points,
            k_list=k_list,
            lambda_list=lambda_list,
            round2_k_list=round2_k_list,
            sam2_autocast_context=sam2_autocast_context,
            write_artifacts=write_artifacts,
            completed_cases=completed_case_names(all_rows, "round3", round3_expected),
            progress_path=progress_path,
        )
        all_rows = merge_unique_candidate_rows(all_rows, round3_new)
        round3_rows = [row for row in all_rows if row["round_name"] == "round3"]
        round3_summary = summarize(round3_rows, ["method", "layout", "clustering"])
    else:
        round3_summary = []

    save_csv(out_dir / "candidates.csv", all_rows, candidate_fields)
    save_csv(out_dir / "round1_prompt_budget.csv", round1_summary)
    save_csv(out_dir / "round2_contrastive_matching.csv", round2_summary)
    save_csv(out_dir / "round3_layout_clustering.csv", round3_summary)

    if write_artifacts:
        examples = select_examples(all_rows, out_dir)
        generate_report(out_dir, all_rows, round1_summary, round2_summary, round3_summary, examples)
    metadata = {
        "manifest": rel(resolve_path(args.manifest)),
        "dataset_filter": args.dataset_filter,
        "role_filter": args.role_filter,
        "out_dir": rel(out_dir),
        "case_count": len(manifest),
        "candidate_count": len(all_rows),
        "best_k": best_k,
        "best_lambda": best_lambda,
        "best_round3_method": best_value(round3_summary, "method", "mean_dice") if round3_summary else "",
        "segmenter": args.segmenter,
        "dino_weights": rel(resolve_path(args.dino_weights)),
        "sam_checkpoint": rel(resolve_path(args.sam_checkpoint)),
        "sam3_checkpoint": rel(resolve_path(args.sam3_checkpoint)),
        "k_list": k_list,
        "lambda_list": lambda_list,
        "round2_k_list": round2_k_list,
        "round3_config_source": round3_config_source,
        "rounds": list(round_names),
        "minimal_artifacts": args.minimal_artifacts,
        "resume_enabled": args.resume,
        "progress_checkpoint": rel(progress_path),
    }
    current_candidate_keys = {
        (str(row["case_name"]), str(row["round_name"]), str(row["method"]))
        for row in all_rows
    }
    new_candidate_keys = current_candidate_keys - initial_candidate_keys
    new_episode_count = len({key[0] for key in new_candidate_keys})
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )
    metadata.update(runtime_metrics(
        elapsed_seconds=time.perf_counter() - started_at,
        candidate_count=len(new_candidate_keys),
        case_count=new_episode_count,
        peak_gpu_mb=peak_gpu_mb,
    ))
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] candidates={len(all_rows)} report={rel(out_dir / 'report' / 'monuseg_dense_prompt_report.md')}")


def run_round(
    case_cache: dict[str, dict[str, Any]],
    prompt_predictor: Any,
    segmenter: str,
    round_name: str,
    best_k: int,
    best_lambda: float,
    max_points: int,
    k_list: list[int],
    lambda_list: list[float],
    round2_k_list: list[int],
    sam2_autocast_context: Any,
    write_artifacts: bool,
    completed_cases: set[str] | None = None,
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = candidate_specs(round_name, best_k, best_lambda, k_list, lambda_list, round2_k_list)
    active_sam_query_id: str | None = None
    completed_cases = completed_cases or set()
    for case_name, cache in case_cache.items():
        if case_name in completed_cases:
            print(f"[{round_name}] {case_name} resumed from checkpoint")
            continue
        case_start = len(rows)
        image: Image.Image = cache["query_image"]
        gt: np.ndarray = cache["query_gt"]
        s_fg: np.ndarray = cache["s_fg"]
        s_bg: np.ndarray = cache["s_bg"]
        row: dict[str, str] = cache["row"]
        stage1_dir: Path = cache["stage1_dir"]
        mask_dir: Path = cache["mask_dir"]
        overlay_dir: Path = cache["overlay_dir"]
        points_dir: Path = cache["points_dir"]
        query_case_id = row["query_case_id"]
        if segmenter == "sam" and should_refresh_query_embedding(active_sam_query_id, query_case_id):
            prompt_predictor.set_image(np.asarray(image))
            active_sam_query_id = query_case_id
        for spec in specs:
            lam = float(spec["lambda_bg"])
            score_map = zscore(s_fg) - lam * zscore(s_bg)
            if spec["round_name"] == "round1":
                score_map = s_fg.copy()
            score_name = f"{spec['round_name']}_{spec['method']}"
            np.save(stage1_dir / f"{score_name}_score.npy", score_map)
            if write_artifacts:
                save_heatmap(image, score_map, stage1_dir / f"{score_name}_score_heatmap.png")
            pool = dense_points_from_score(score_map, s_fg, s_bg, image.size, max_points=max_points)
            selected = choose_points(spec, score_map, s_fg, s_bg, image.size, pool)
            if write_artifacts:
                save_point_overlay(image, selected, stage1_dir / f"{score_name}_points_overlay.png", f"{case_name} {score_name} K={len(selected)}", (255, 50, 60))
            points_path = points_dir / f"{score_name}_points.csv"
            save_points(points_path, selected)

            if segmenter == "sam3":
                pred, sam_score = predict_sam3(prompt_predictor, image, selected)
            elif segmenter == "sam2":
                pred, sam_score = predict_sam2(prompt_predictor, image, selected, sam2_autocast_context)
            else:
                pred, sam_score = predict_sam(prompt_predictor, selected)
            dice, iou = dice_iou(pred, gt)
            pred_sum = float(pred.sum())
            gt_sum = float(gt.sum())
            inter = float(np.logical_and(pred, gt).sum())
            fg_precision = inter / pred_sum if pred_sum else 0.0
            fg_recall = inter / gt_sum if gt_sum else 1.0
            obj = object_metrics(pred, gt)
            mask_path = mask_dir / f"{score_name}.png"
            overlay_path = overlay_dir / f"{score_name}.png"
            if write_artifacts:
                Image.fromarray((pred.astype(np.uint8) * 255), mode="L").save(mask_path)
                title = f"{case_name} | {score_name} | Dice {dice:.3f} | IoU {iou:.3f} | K {len(selected)}"
                save_overlay(image, gt, pred, selected, overlay_path, title)
            result = CandidateResult(
                case_name=case_name,
                query_case_id=row["query_case_id"],
                round_name=spec["round_name"],
                method=spec["method"],
                matching=spec["matching"],
                layout=spec["layout"],
                clustering=spec["clustering"],
                lambda_bg=round(lam, 4),
                k=int(spec["k"]),
                prompt_count=len(selected),
                sam_score=round(float(sam_score), 6),
                dice=round(float(dice), 6),
                iou=round(float(iou), 6),
                fg_precision=round(float(fg_precision), 6),
                fg_recall=round(float(fg_recall), 6),
                object_recall=round(float(obj["object_recall"]), 6),
                object_precision=round(float(obj["object_precision"]), 6),
                gt_components=int(obj["gt_components"]),
                pred_components=int(obj["pred_components"]),
                component_count_error=round(float(obj["component_count_error"]), 6),
                merge_error=round(float(obj["merge_error"]), 6),
                pred_area_pct=round(100.0 * pred_sum / float(pred.size), 6),
                mask_path=rel(mask_path),
                overlay_path=rel(overlay_path),
                points_path=rel(points_path),
            )
            rows.append(asdict(result))
        if progress_path is not None:
            append_csv_rows(
                progress_path,
                rows[case_start:],
                list(CandidateResult.__dataclass_fields__.keys()),
            )
        print(f"[{round_name}] {case_name} candidates={len(specs)}")
    return rows


def select_examples(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, str]]:
    examples: list[dict[str, Any]] = []
    for round_name in ["round1", "round2", "round3"]:
        subset = [row for row in rows if row["round_name"] == round_name]
        if not subset:
            continue
        examples.append(max(subset, key=lambda row: float(row["dice"])))
        examples.append(min(subset, key=lambda row: float(row["dice"])))
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    report_dir = out_dir / "report"
    for row in examples:
        key = str(row["overlay_path"])
        if key in seen:
            continue
        seen.add(key)
        overlay_abs = resolve_path(key)
        try:
            overlay_rel = os.path.relpath(overlay_abs, report_dir).replace("\\", "/")
        except ValueError:
            overlay_rel = key
        unique.append(
            {
                "title": f"{row['round_name']} {row['method']}",
                "case_name": str(row["case_name"]),
                "method": str(row["method"]),
                "dice": f"{float(row['dice']):.4f}",
                "iou": f"{float(row['iou']):.4f}",
                "object_recall": f"{float(row['object_recall']):.4f}",
                "overlay_path": key,
                "overlay_rel_to_report": overlay_rel,
            }
        )
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MonuSeg dense prompt and contrastive matcher experiment.")
    parser.add_argument("--manifest", default="runs/research_direction/matcher_medical_expanded_50/medical_matcher_pilot_manifest.csv")
    parser.add_argument("--dataset-filter", default="monuseg", help="Case-insensitive dataset name; pass an empty string to keep every row.")
    parser.add_argument("--role-filter", default="", help="Case-insensitive policy role, such as train, val, or test; pass an empty string to keep every role.")
    parser.add_argument("--out-dir", default="runs/research_direction/monuseg_dense_prompt_t1_t5")
    parser.add_argument("--case-limit", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--segmenter", choices=["sam", "sam2", "sam3"], default="sam")
    parser.add_argument("--k-list", default="1,3,5,10,20,50,100")
    parser.add_argument("--lambda-list", default="0,0.25,0.5,1.0")
    parser.add_argument("--round2-k-list", default="")
    parser.add_argument("--round3-fixed-k", type=int, default=0, help="Optional validation-frozen K for Round 3; requires --round3-fixed-lambda.")
    parser.add_argument("--round3-fixed-lambda", type=float, default=None, help="Optional validation-frozen lambda for Round 3; requires --round3-fixed-k.")
    parser.add_argument("--dino-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dino-size", default="vit_large")
    parser.add_argument("--dino-device", default="auto")
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam3-checkpoint", default="assets/checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--sam3-threshold", type=float, default=0.5)
    parser.add_argument("--sam2-checkpoint", default="assets/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam-device", default="auto")
    parser.add_argument("--skip-round3", action="store_true", help="Run only Round1/2 stable K/lambda actions.")
    parser.add_argument("--minimal-artifacts", action="store_true", help="Skip heatmaps, masks, overlays, and HTML while retaining CSV/NPY policy features.")
    parser.add_argument("--resume", action="store_true", help="Resume fully completed case/round blocks from candidates_progress.csv.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
