from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from apply_stage1_feedback import FeedbackPoint, parse_points, rank_points, save_feedback_outputs
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

# Output contract inherited from save_feedback_outputs:
# feedback_points.json, feedback_points.csv, similarity_after.npy,
# feedback_overlay_before_after.png, feedback_heatmap_before_after.png.


def clean_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def load_stage1(stage1_dir: Path) -> tuple[dict[str, Any], np.ndarray, Image.Image]:
    payload_path = stage1_dir / "points.json"
    similarity_path = stage1_dir / "similarity.npy"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    if not similarity_path.exists():
        raise FileNotFoundError(similarity_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    similarity = np.load(similarity_path).astype(np.float32)
    target = Image.open(resolve_path(payload["target_image"])).convert("RGB")
    return payload, similarity, target


@torch.inference_mode()
def compute_reference_prototype(model: torch.nn.Module, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    reference_path = resolve_path(payload["reference_image"])
    reference_original, reference_resized = resized_rgb(reference_path)
    reference_box = tuple(payload["reference_box"]) if payload.get("reference_box") is not None else None
    if reference_box is not None:
        reference_box = parse_box(",".join(str(value) for value in reference_box))
    reference_mask = resolve_path(payload["reference_mask"]) if payload.get("reference_mask") else None
    mask = make_reference_mask(reference_original, reference_box, reference_mask)
    features = extract_features(model, reference_resized, device)
    weights = mask_to_patch_weights(mask, int(model.patch_size), device)
    foreground = weights > 0
    proto = (features[foreground] * weights[foreground].unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


def gaussian_patch_weights(
    points: list[FeedbackPoint],
    original_size: tuple[int, int],
    feature_shape: tuple[int, int],
    patch_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    height, width = feature_shape
    if not points:
        return torch.zeros((height * width,), dtype=torch.float32, device=device)
    orig_w, orig_h = original_size
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    weights = torch.zeros((height, width), dtype=torch.float32, device=device)
    sigma_patch = max(float(sigma) / float(patch_size), 1e-3)
    for point in points:
        px = (float(point.x) / float(orig_w)) * float(IMAGE_SIZE)
        py = (float(point.y) / float(orig_h)) * float(IMAGE_SIZE)
        cx = px / float(patch_size)
        cy = py / float(patch_size)
        weights += torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_patch * sigma_patch))
    flat = weights.reshape(-1)
    total = flat.sum()
    if float(total.detach().cpu()) <= 0.0:
        return flat
    return flat / total


@torch.inference_mode()
def compute_feedback_prototype(
    target_features: torch.Tensor,
    feedback_points: list[FeedbackPoint],
    original_size: tuple[int, int],
    feature_shape: tuple[int, int],
    patch_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor | None:
    weights = gaussian_patch_weights(feedback_points, original_size, feature_shape, patch_size, sigma, device)
    if float(weights.sum().detach().cpu()) <= 0.0:
        return None
    proto = (target_features * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


@torch.inference_mode()
def apply_feature_feedback(
    reference_proto: torch.Tensor,
    target_features: torch.Tensor,
    positive_proto: torch.Tensor | None,
    negative_proto: torch.Tensor | None,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    updated_proto = reference_proto.clone()
    if positive_proto is not None:
        updated_proto = updated_proto + float(positive_weight) * positive_proto
    if negative_proto is not None:
        updated_proto = updated_proto - float(negative_weight) * negative_proto
    updated_proto = F.normalize(updated_proto, dim=1, p=2)
    similarity = (updated_proto @ target_features.t()).squeeze(0)
    feature_size = int(round(float(target_features.shape[0]) ** 0.5))
    return similarity.reshape(feature_size, feature_size).detach().cpu().to(torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply feature/prototype feedback to Matcher Stage 1 points.")
    parser.add_argument("--stage1-dir", default="runs/research_direction/stage1_matching/elephant_pair")
    parser.add_argument("--positive-points", default="268,331")
    parser.add_argument("--negative-points", default="30,430;455,420")
    parser.add_argument("--sigma", type=float, default=42.0)
    parser.add_argument("--positive-weight", type=float, default=0.35)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--metric-radius", type=float, default=70.0)
    parser.add_argument("--dinov2-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dinov2-size", default="vit_large")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p0_feature_feedback/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_dir = resolve_path(args.stage1_dir)
    out_dir = resolve_path(args.out_dir)
    device = clean_device(args.device)
    payload, before, target_original = load_stage1(stage1_dir)
    target_path = resolve_path(payload["target_image"])
    target_image = load_rgb(target_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    positive_points = parse_points(args.positive_points)
    negative_points = parse_points(args.negative_points)

    model = load_dinov2(resolve_path(args.dinov2_weights), args.dinov2_size, device)
    reference_proto = compute_reference_prototype(model, payload, device)
    target_features = extract_features(model, target_image, device)
    patch_size = int(model.patch_size)
    feature_shape = (IMAGE_SIZE // patch_size, IMAGE_SIZE // patch_size)
    positive_proto = compute_feedback_prototype(target_features, positive_points, target_original.size, feature_shape, patch_size, args.sigma, device)
    negative_proto = compute_feedback_prototype(target_features, negative_points, target_original.size, feature_shape, patch_size, args.sigma, device)
    after_t = apply_feature_feedback(
        reference_proto=reference_proto,
        target_features=target_features,
        positive_proto=positive_proto,
        negative_proto=negative_proto,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
    )
    after = after_t.numpy().astype(np.float32)
    before_points = rank_points(before, before, target_original.size, args.top_k, args.threshold)
    after_points = rank_points(before, after, target_original.size, args.top_k, args.threshold)
    save_feedback_outputs(
        out_dir=out_dir,
        payload=payload,
        before=before,
        after=after,
        before_points=before_points,
        after_points=after_points,
        target=target_original,
        positive_points=positive_points,
        negative_points=negative_points,
        args=args,
    )
    mode_path = out_dir / "feature_feedback_method.json"
    mode_payload = {
        "method": "feature_prototype_update",
        "formula": "updated_proto = normalize(reference_proto + positive_weight * positive_proto - negative_weight * negative_proto)",
        "positive_proto_used": positive_proto is not None,
        "negative_proto_used": negative_proto is not None,
    }
    mode_path.write_text(json.dumps(mode_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Before selected count: {sum(1 for point in before_points if point.selected_by_threshold)}")
    print(f"After selected count: {sum(1 for point in after_points if point.selected_by_threshold)}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
