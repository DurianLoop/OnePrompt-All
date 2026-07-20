from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def patch_indices_from_mask(
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    foreground_threshold: float = 0.99,
    background_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    grid_height, grid_width = grid_shape
    occupancy = np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.float32)).resize(
            (grid_width, grid_height),
            Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )
    foreground = np.flatnonzero(occupancy.reshape(-1) >= foreground_threshold)
    background = np.flatnonzero(occupancy.reshape(-1) <= background_threshold)
    return foreground.astype(np.int64), background.astype(np.int64)


def _normalize(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def cycle_consistent_matches(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    reference_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = _normalize(reference_features)
    query = _normalize(query_features)
    selected_indices = np.asarray(reference_indices, dtype=np.int64)
    if selected_indices.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    similarities = reference[selected_indices] @ query.T
    forward = similarities.argmax(axis=1)
    backward = (query[forward] @ reference.T).argmax(axis=1)
    keep = np.isin(backward, selected_indices)
    return np.unique(forward[keep]).astype(np.int64), np.unique(forward[~keep]).astype(np.int64)


def indices_to_points(
    indices: np.ndarray,
    grid_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> list[list[int]]:
    grid_height, grid_width = grid_shape
    image_width, image_height = image_size
    points: list[list[int]] = []
    for index in np.asarray(indices, dtype=np.int64).tolist():
        patch_y, patch_x = divmod(index, grid_width)
        points.append([
            int(round((patch_x + 0.5) * image_width / grid_width)),
            int(round((patch_y + 0.5) * image_height / grid_height)),
        ])
    return points


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def save_overlay(
    image: Image.Image,
    gt: np.ndarray,
    pred: np.ndarray,
    positive: list[list[int]],
    negative: list[list[int]],
    path: Path,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    rgb[tp] = (0.55 * rgb[tp] + 0.45 * np.asarray([30, 220, 90])).astype(np.uint8)
    rgb[fp] = (0.45 * rgb[fp] + 0.55 * np.asarray([255, 45, 45])).astype(np.uint8)
    rgb[fn] = (0.45 * rgb[fn] + 0.55 * np.asarray([30, 120, 255])).astype(np.uint8)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for x, y in positive:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 245, 0), outline=(0, 0, 0), width=2)
    for x, y in negative:
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=(0, 255, 255), width=3)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=(0, 255, 255), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    from run_monuseg_dense_prompt_experiment import (
        IMAGE_SIZE,
        PATCH_SIZE,
        clean_device,
        dice_iou,
        extract_features,
        load_dinov2,
        load_mask,
        load_sam_predictor,
        object_metrics,
    )

    gbmseg_root = ROOT / "external" / "GBMSeg"
    sys.path.insert(0, str(gbmseg_root))
    from automatic_prompt_engineering import APE

    parser = argparse.ArgumentParser(description="Run a clearly labeled GBMSeg-style adapted baseline on a reference-query manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dino-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dino-size", default="vit_large")
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--background-threshold", type=float, default=0.01)
    parser.add_argument("--radius-fraction", type=float, default=0.125)
    parser.add_argument("--case-limit", type=int, default=0)
    args = parser.parse_args()

    episodes = read_csv(resolve(args.manifest))
    if args.case_limit > 0:
        episodes = episodes[: args.case_limit]
    device = clean_device(args.device)
    dino = load_dinov2(resolve(args.dino_weights), args.dino_size, device)
    predictor, _ = load_sam_predictor(resolve(args.sam_checkpoint), args.sam_model_type, args.device)
    grid_shape = (IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)
    feature_cache: dict[str, np.ndarray] = {}
    reference_index_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    out_dir = resolve(args.out_dir)

    for episode in episodes:
        support_id = episode["support_case_id"]
        query_id = episode["query_case_id"]
        if support_id not in feature_cache:
            support_image = Image.open(resolve(episode["support_image_path"])).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            feature_cache[support_id] = extract_features(dino, support_image, device).detach().cpu().numpy()
            support_mask = load_mask(resolve(episode["support_mask_path"]), (IMAGE_SIZE, IMAGE_SIZE))
            reference_index_cache[support_id] = patch_indices_from_mask(
                support_mask,
                grid_shape,
                args.foreground_threshold,
                args.background_threshold,
            )
        query_image = Image.open(resolve(episode["query_image_path"])).convert("RGB")
        if query_id not in feature_cache:
            feature_cache[query_id] = extract_features(
                dino,
                query_image.resize((IMAGE_SIZE, IMAGE_SIZE)),
                device,
            ).detach().cpu().numpy()
        reference_features = feature_cache[support_id]
        query_features = feature_cache[query_id]
        foreground_indices, background_indices = reference_index_cache[support_id]
        positive_indices, hard_indices = cycle_consistent_matches(reference_features, query_features, foreground_indices)
        negative_indices, _ = cycle_consistent_matches(reference_features, query_features, background_indices)
        used_positive_fallback = 0
        if positive_indices.size == 0:
            used_positive_fallback = 1
            foreground_proto = _normalize(reference_features[foreground_indices]).mean(axis=0)
            positive_indices = np.asarray([int((_normalize(query_features) @ foreground_proto).argmax())])

        positive = indices_to_points(positive_indices, grid_shape, query_image.size)
        negative = indices_to_points(negative_indices, grid_shape, query_image.size)
        hard = indices_to_points(hard_indices, grid_shape, query_image.size)
        radius = args.radius_fraction * min(query_image.size)
        positive_opt, negative_opt = APE(
            positive,
            negative,
            hard,
            radius,
            radius,
            1.0,
            radius,
            True,
        )
        coords = np.asarray(positive_opt + negative_opt, dtype=np.float32)
        labels = np.asarray([1] * len(positive_opt) + [0] * len(negative_opt), dtype=np.int32)
        predictor.set_image(np.asarray(query_image))
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
        best = int(np.argmax(scores))
        pred = masks[best].astype(bool)
        gt = load_mask(resolve(episode["query_mask_path"]), query_image.size)
        dice, iou = dice_iou(pred, gt)
        obj = object_metrics(pred, gt)
        pred_sum = float(pred.sum())
        gt_sum = float(gt.sum())
        inter = float(np.logical_and(pred, gt).sum())
        stem = episode["pilot_id"]
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / f"{stem}.png")
        overlay_path = out_dir / "overlays" / f"{stem}.png"
        save_overlay(query_image, gt, pred, positive_opt, negative_opt, overlay_path)
        rows.append({
            "episode_id": stem,
            "query_case_id": query_id,
            "positive_count_raw": len(positive),
            "negative_count_raw": len(negative),
            "hard_count_raw": len(hard),
            "positive_count": len(positive_opt),
            "negative_count": len(negative_opt),
            "used_positive_fallback": used_positive_fallback,
            "dice": dice,
            "iou": iou,
            "fg_precision": inter / pred_sum if pred_sum else 0.0,
            "fg_recall": inter / gt_sum if gt_sum else 1.0,
            "object_recall": obj["object_recall"],
            "merge_error": obj["merge_error"],
            "pred_area_pct": 100.0 * pred_sum / pred.size,
            "sam_score": float(scores[best]),
            "overlay_path": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
        })
        print(f"[gbmseg-style] {query_id} pos={len(positive_opt)} neg={len(negative_opt)} dice={dice:.4f}")

    write_csv(out_dir / "per_case.csv", rows)
    metrics = ("dice", "iou", "object_recall", "merge_error", "pred_area_pct", "positive_count", "negative_count")
    summary = {
        "method": "GBMSeg-style adapted baseline",
        "case_count": len(rows),
        **{f"mean_{metric}": mean(float(row[metric]) for row in rows) for metric in metrics},
        "adaptations": [
            "DINOv2 ViT-L replaces the original ViT-G checkpoint",
            "SAM ViT-B is fixed to match the main controlled experiment",
            "reference patches use at least 50% foreground occupancy because the original 100% rule yields zero patches for the verified MoNuSeg support mask",
            "empty cycle-consistent foreground matches use a recorded prototype fallback",
        ],
        "source_repository_commit": "d3e41786ba603bcc85f58a9b267d23de56aa2c87",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
