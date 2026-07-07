from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
from filter_stage3_proposals import (
    ClusterPoint,
    bbox_area_pct,
    load_cluster_points,
    load_mask,
    load_proposals,
    mask_iou,
    point_inside,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FeatureMetric:
    group_id: int
    cluster_id: int
    sample_size: int
    sample_index: int
    sam_score: float
    instance_similarity: float
    mask_area_pct: float
    bbox_area_pct: float
    cluster_coverage: float
    cluster_points_inside: int
    cluster_points_total: int
    prompt_coverage: float
    prompt_points_inside: int
    prompt_points_total: int
    feature_filter_score: float
    filter_status: str
    reject_reasons: list[str]
    retained_rank: int
    bbox_xyxy: list[int]
    mask_path: str
    overlay_path: str


def clean_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def load_stage1_metadata(stage1_dir: Path) -> dict[str, Any]:
    points_json = stage1_dir / "points.json"
    if not points_json.exists():
        raise FileNotFoundError(points_json)
    return json.loads(points_json.read_text(encoding="utf-8"))


@torch.inference_mode()
def compute_reference_prototype(
    model: torch.nn.Module,
    stage1_payload: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    reference_image_path = resolve_path(stage1_payload["reference_image"])
    reference_original, reference_resized = resized_rgb(reference_image_path)
    reference_mask_path = resolve_path(stage1_payload["reference_mask"]) if stage1_payload.get("reference_mask") else None
    reference_box = tuple(stage1_payload["reference_box"]) if stage1_payload.get("reference_box") is not None else None
    if reference_box is not None:
        reference_box = parse_box(",".join(str(value) for value in reference_box))
    reference_mask = make_reference_mask(reference_original, reference_box, reference_mask_path)
    ref_features = extract_features(model, reference_resized, device)
    weights = mask_to_patch_weights(reference_mask, int(model.patch_size), device)
    ref_mask = weights > 0
    proto = (ref_features[ref_mask] * weights[ref_mask].unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


@torch.inference_mode()
def compute_target_features(model: torch.nn.Module, target_image_path: Path, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    target_original = load_rgb(target_image_path)
    target_resized = target_original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    target_features = extract_features(model, target_resized, device)
    return target_features, target_original


@torch.inference_mode()
def compute_proposal_prototype(
    target_features: torch.Tensor,
    proposal_mask_path: str,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    mask = Image.open(resolve_path(proposal_mask_path)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    weights = mask_to_patch_weights(mask, patch_size, device)
    fg = weights > 0
    proto = (target_features[fg] * weights[fg].unsqueeze(1)).sum(dim=0, keepdim=True)
    return F.normalize(proto, dim=1, p=2)


def compute_instance_similarity(reference_proto: torch.Tensor, proposal_proto: torch.Tensor) -> float:
    return float((reference_proto @ proposal_proto.t()).squeeze().detach().cpu())


def compute_coverages(proposal: dict[str, Any], cluster_points: list[ClusterPoint]) -> tuple[float, int, int, float, int, int]:
    mask = load_mask(proposal["mask_path"])
    cluster_id = int(proposal["cluster_id"])
    points = [point for point in cluster_points if point.cluster_id == cluster_id]
    cluster_inside = sum(1 for point in points if point_inside(mask, point.x, point.y))
    prompt_points = proposal["prompt_points"]
    prompt_inside = sum(1 for point in prompt_points if point_inside(mask, float(point["x"]), float(point["y"])))
    cluster_coverage = float(cluster_inside) / float(len(points)) if points else 0.0
    prompt_coverage = float(prompt_inside) / float(len(prompt_points)) if prompt_points else 0.0
    return cluster_coverage, cluster_inside, len(points), prompt_coverage, prompt_inside, len(prompt_points)


def feature_metric_for_proposal(
    proposal: dict[str, Any],
    target_features: torch.Tensor,
    reference_proto: torch.Tensor,
    cluster_points: list[ClusterPoint],
    patch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> FeatureMetric:
    proposal_proto = compute_proposal_prototype(target_features, proposal["mask_path"], patch_size, device)
    instance_similarity = compute_instance_similarity(reference_proto, proposal_proto)
    cluster_coverage, cluster_inside, cluster_total, prompt_coverage, prompt_inside, prompt_total = compute_coverages(proposal, cluster_points)
    mask = load_mask(proposal["mask_path"])
    bbox_pct = bbox_area_pct(proposal["bbox_xyxy"], mask.shape)
    area_pct = float(proposal["mask_area_pct"])
    area_penalty = max(0.0, area_pct - args.target_area_pct) / max(args.target_area_pct, 1e-6)
    bbox_penalty = max(0.0, bbox_pct - args.target_bbox_area_pct) / max(args.target_bbox_area_pct, 1e-6)
    feature_filter_score = (
        instance_similarity
        + 0.1 * float(proposal["sam_score"])
        + 0.1 * cluster_coverage
        + 0.05 * prompt_coverage
        - 0.15 * area_penalty
        - 0.15 * bbox_penalty
    )

    reasons: list[str] = []
    if instance_similarity < args.min_instance_similarity:
        reasons.append("low_instance_similarity")
    if float(proposal["sam_score"]) < args.min_score:
        reasons.append("low_score")
    if area_pct < args.min_area_pct:
        reasons.append("too_small_area")
    if area_pct > args.max_area_pct:
        reasons.append("too_large_area")
    if bbox_pct > args.max_bbox_area_pct:
        reasons.append("too_large_bbox")
    if cluster_coverage < args.min_cluster_coverage:
        reasons.append("low_cluster_coverage")
    if prompt_coverage < args.min_prompt_coverage:
        reasons.append("low_prompt_coverage")

    return FeatureMetric(
        group_id=int(proposal["group_id"]),
        cluster_id=int(proposal["cluster_id"]),
        sample_size=int(proposal["sample_size"]),
        sample_index=int(proposal["sample_index"]),
        sam_score=round(float(proposal["sam_score"]), 6),
        instance_similarity=round(instance_similarity, 6),
        mask_area_pct=round(area_pct, 4),
        bbox_area_pct=round(float(bbox_pct), 4),
        cluster_coverage=round(cluster_coverage, 4),
        cluster_points_inside=cluster_inside,
        cluster_points_total=cluster_total,
        prompt_coverage=round(prompt_coverage, 4),
        prompt_points_inside=prompt_inside,
        prompt_points_total=prompt_total,
        feature_filter_score=round(float(feature_filter_score), 6),
        filter_status="candidate" if not reasons else "rejected",
        reject_reasons=reasons,
        retained_rank=0,
        bbox_xyxy=[int(value) for value in proposal["bbox_xyxy"]],
        mask_path=proposal["mask_path"],
        overlay_path=proposal["overlay_path"],
    )


def filter_feature_candidates(metrics: list[FeatureMetric]) -> list[FeatureMetric]:
    return [metric for metric in metrics if metric.filter_status == "candidate"]


def run_feature_nms(candidates: list[FeatureMetric], nms_iou: float, max_retained_per_cluster: int = 0) -> list[FeatureMetric]:
    retained: list[FeatureMetric] = []
    for cluster_id in sorted({metric.cluster_id for metric in candidates}):
        cluster = [metric for metric in candidates if metric.cluster_id == cluster_id]
        cluster.sort(key=lambda metric: (metric.feature_filter_score, metric.instance_similarity, metric.sam_score), reverse=True)
        kept_masks: list[np.ndarray] = []
        kept_in_cluster = 0
        for metric in cluster:
            if max_retained_per_cluster > 0 and kept_in_cluster >= max_retained_per_cluster:
                break
            mask = load_mask(metric.mask_path)
            if all(mask_iou(mask, kept) < nms_iou for kept in kept_masks):
                retained.append(metric)
                kept_masks.append(mask)
                kept_in_cluster += 1
    retained.sort(key=lambda metric: (metric.cluster_id, -metric.feature_filter_score))
    for rank, metric in enumerate(retained, start=1):
        metric.filter_status = "retained"
        metric.retained_rank = rank
    return retained


def draw_retained_overlay(target: Image.Image, retained: list[FeatureMetric]) -> Image.Image:
    canvas = target.convert("RGBA")
    palette = [(255, 40, 40, 110), (50, 190, 255, 110), (70, 220, 90, 110)]
    draw = ImageDraw.Draw(canvas)
    for idx, metric in enumerate(retained):
        mask = load_mask(metric.mask_path)
        color = Image.new("RGBA", canvas.size, palette[idx % len(palette)])
        alpha = Image.fromarray((mask.astype(np.uint8) * 110), mode="L")
        color.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, color)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(tuple(metric.bbox_xyxy), outline=(255, 255, 0), width=3)
        draw.text((max(6, metric.bbox_xyxy[0] + 4), max(6, metric.bbox_xyxy[1] + 4)), f"#{metric.retained_rank} g{metric.group_id}", fill=(255, 255, 255))
    out = canvas.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((4, out.height - 42, min(out.width - 4, 560), out.height - 4), fill=(0, 0, 0))
    draw.text((12, out.height - 31), f"feature-retained masks: {len(retained)}", fill=(255, 255, 255))
    return out


def make_grid(metrics: list[FeatureMetric], columns: int = 3) -> Image.Image:
    overlays = [Image.open(resolve_path(metric.overlay_path)).convert("RGB") for metric in metrics]
    if not overlays:
        return Image.new("RGB", (320, 240), (255, 255, 255))
    thumb_w = 320
    thumb_h = int(round(overlays[0].height * (thumb_w / overlays[0].width)))
    thumbs = [image.resize((thumb_w, thumb_h), Image.BICUBIC) for image in overlays]
    rows = int(np.ceil(len(thumbs) / columns))
    grid = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % columns) * thumb_w, (idx // columns) * thumb_h))
    return grid


def save_feature_outputs(
    out_dir: Path,
    proposals_dir: Path,
    stage1_dir: Path,
    proposal_payload: dict[str, Any],
    target_image: Image.Image,
    metrics: list[FeatureMetric],
    retained: list[FeatureMetric],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row["reject_reasons"] = json.dumps(metric.reject_reasons, ensure_ascii=False)
        row["bbox_xyxy"] = json.dumps(metric.bbox_xyxy)
        rows.append(row)
    with (out_dir / "feature_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(FeatureMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    draw_retained_overlay(target_image, retained).save(out_dir / "feature_retained_overlay.png")
    make_grid(retained).save(out_dir / "feature_retained_grid.png")

    candidate_count = sum(1 for metric in metrics if not metric.reject_reasons)
    reason_counts: dict[str, int] = {}
    for metric in metrics:
        for reason in metric.reject_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    result = {
        "proposals_dir": str(proposals_dir.relative_to(WORKSPACE_ROOT)),
        "stage1_dir": str(stage1_dir.relative_to(WORKSPACE_ROOT)),
        "target_image": proposal_payload["target_image"],
        "thresholds": {
            "min_instance_similarity": args.min_instance_similarity,
            "min_score": args.min_score,
            "min_area_pct": args.min_area_pct,
            "max_area_pct": args.max_area_pct,
            "max_bbox_area_pct": args.max_bbox_area_pct,
            "min_cluster_coverage": args.min_cluster_coverage,
            "min_prompt_coverage": args.min_prompt_coverage,
            "nms_iou": args.nms_iou,
            "max_retained_per_cluster": args.max_retained_per_cluster,
        },
        "proposal_count": len(metrics),
        "candidate_count": candidate_count,
        "retained_count": len(retained),
        "k_pred": len(retained),
        "reject_reason_counts": reason_counts,
        "retained_group_ids": [metric.group_id for metric in retained],
        "metrics": [asdict(metric) for metric in metrics],
        "retained": [asdict(metric) for metric in retained],
    }
    (out_dir / "feature_scores.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    command = (
        "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\feature_match_stage3_proposals.py "
        f"--proposals-dir {args.proposals_dir} --stage1-dir {args.stage1_dir} "
        f"--dinov2-weights {args.dinov2_weights} --min-instance-similarity {args.min_instance_similarity} "
        f"--min-score {args.min_score} --min-area-pct {args.min_area_pct} --max-area-pct {args.max_area_pct} "
        f"--max-bbox-area-pct {args.max_bbox_area_pct} --min-cluster-coverage {args.min_cluster_coverage} "
        f"--nms-iou {args.nms_iou} --max-retained-per-cluster {args.max_retained_per_cluster} --out-dir {args.out_dir}"
    )
    summary = [
        "# Matcher Stage 3 Feature Matching Summary",
        "",
        f"- Proposals dir: `{result['proposals_dir']}`",
        f"- Stage1 dir: `{result['stage1_dir']}`",
        f"- Proposal count: `{len(metrics)}`",
        f"- Candidate count after thresholds: `{candidate_count}`",
        f"- Retained count / K_pred: `{len(retained)}`",
        f"- Retained group ids: `{result['retained_group_ids']}`",
        f"- Reject reason counts: `{reason_counts}`",
    ]
    if retained:
        best = retained[0]
        summary.extend([
            f"- Best retained instance similarity: `{best.instance_similarity}`",
            f"- Best retained feature score: `{best.feature_filter_score}`",
        ])
    summary.extend([
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Artifacts",
        "",
        "- `feature_scores.csv`",
        "- `feature_scores.json`",
        "- `feature_retained_overlay.png`",
        "- `feature_retained_grid.png`",
    ])
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-level instance matching for Stage 3 SAM proposals.")
    parser.add_argument("--proposals-dir", default="runs/research_direction/stage2_proposals/elephant_pair")
    parser.add_argument("--stage1-dir", default="runs/research_direction/stage1_matching/elephant_pair")
    parser.add_argument("--dinov2-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dinov2-size", default="vit_large")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-instance-similarity", type=float, default=0.45)
    parser.add_argument("--min-score", type=float, default=0.80)
    parser.add_argument("--min-area-pct", type=float, default=5.0)
    parser.add_argument("--max-area-pct", type=float, default=45.0)
    parser.add_argument("--max-bbox-area-pct", type=float, default=45.0)
    parser.add_argument("--min-cluster-coverage", type=float, default=0.30)
    parser.add_argument("--min-prompt-coverage", type=float, default=1.0)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--max-retained-per-cluster", type=int, default=0)
    parser.add_argument("--target-area-pct", type=float, default=25.0)
    parser.add_argument("--target-bbox-area-pct", type=float, default=30.0)
    parser.add_argument("--out-dir", default="runs/research_direction/stage3_feature_matching/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = clean_device(args.device)
    proposals_dir = resolve_path(args.proposals_dir)
    stage1_dir = resolve_path(args.stage1_dir)
    out_dir = resolve_path(args.out_dir)
    proposal_payload, proposals, target_image = load_proposals(proposals_dir)
    stage1_payload = load_stage1_metadata(stage1_dir)
    stage2_dir = resolve_path(proposal_payload["stage2_dir"])
    cluster_points = load_cluster_points(stage2_dir)

    model = load_dinov2(resolve_path(args.dinov2_weights), args.dinov2_size, device)
    reference_proto = compute_reference_prototype(model, stage1_payload, device)
    target_features, _ = compute_target_features(model, resolve_path(proposal_payload["target_image"]), device)
    metrics = [
        feature_metric_for_proposal(
            proposal=proposal,
            target_features=target_features,
            reference_proto=reference_proto,
            cluster_points=cluster_points,
            patch_size=int(model.patch_size),
            device=device,
            args=args,
        )
        for proposal in proposals
    ]
    candidates = filter_feature_candidates(metrics)
    retained = run_feature_nms(candidates, args.nms_iou, args.max_retained_per_cluster)
    save_feature_outputs(out_dir, proposals_dir, stage1_dir, proposal_payload, target_image, metrics, retained, args)
    print(f"Proposal count: {len(metrics)}")
    print(f"Candidates after thresholds: {len(candidates)}")
    print(f"Retained count / K_pred: {len(retained)}")
    print(f"Retained group ids: {[metric.group_id for metric in retained]}")
    if retained:
        print(f"Best instance similarity: {retained[0].instance_similarity:.6f}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
