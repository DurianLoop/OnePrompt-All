from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ClusterPoint:
    rank: int
    x: float
    y: float
    cluster_id: int


@dataclass
class ProposalMetric:
    group_id: int
    cluster_id: int
    sample_size: int
    sample_index: int
    sam_score: float
    mask_area_pct: float
    bbox_area_pct: float
    cluster_coverage: float
    cluster_points_inside: int
    cluster_points_total: int
    prompt_coverage: float
    prompt_points_inside: int
    prompt_points_total: int
    filter_score: float
    filter_status: str
    reject_reasons: list[str]
    retained_rank: int
    bbox_xyxy: list[int]
    mask_path: str
    overlay_path: str


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def load_proposals(proposals_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Image.Image]:
    scores_json = proposals_dir / "proposal_scores.json"
    if not scores_json.exists():
        raise FileNotFoundError(scores_json)
    payload = json.loads(scores_json.read_text(encoding="utf-8"))
    image = Image.open(resolve_path(payload["target_image"])).convert("RGB")
    return payload, payload["proposals"], image


def load_cluster_points(stage2_dir: Path) -> list[ClusterPoint]:
    prompt_json = stage2_dir / "prompt_groups.json"
    if not prompt_json.exists():
        raise FileNotFoundError(prompt_json)
    payload = json.loads(prompt_json.read_text(encoding="utf-8"))
    points = [
        ClusterPoint(
            rank=int(point["rank"]),
            x=float(point["x"]),
            y=float(point["y"]),
            cluster_id=int(point["cluster_id"]),
        )
        for point in payload["points"]
        if int(point["cluster_id"]) >= 0
    ]
    return points


def load_mask(path_text: str) -> np.ndarray:
    mask = Image.open(resolve_path(path_text)).convert("L")
    return np.asarray(mask) > 0


def point_inside(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    ix = max(0, min(w - 1, int(round(x))))
    iy = max(0, min(h - 1, int(round(y))))
    return bool(mask[iy, ix])


def bbox_area_pct(bbox: list[int], image_shape: tuple[int, int]) -> float:
    x0, y0, x1, y1 = bbox
    width = max(0, int(x1) - int(x0) + 1)
    height = max(0, int(y1) - int(y0) + 1)
    total = float(image_shape[0] * image_shape[1])
    return 100.0 * float(width * height) / total if total else 0.0


def compute_proposal_metrics(proposal: dict[str, Any], cluster_points: list[ClusterPoint], args: argparse.Namespace) -> ProposalMetric:
    mask = load_mask(proposal["mask_path"])
    cluster_id = int(proposal["cluster_id"])
    points = [point for point in cluster_points if point.cluster_id == cluster_id]
    inside = sum(1 for point in points if point_inside(mask, point.x, point.y))
    prompt_points = proposal["prompt_points"]
    prompt_inside = sum(1 for point in prompt_points if point_inside(mask, float(point["x"]), float(point["y"])))
    cluster_coverage = float(inside) / float(len(points)) if points else 0.0
    prompt_coverage = float(prompt_inside) / float(len(prompt_points)) if prompt_points else 0.0
    bbox_pct = bbox_area_pct(proposal["bbox_xyxy"], mask.shape)
    area_pct = float(proposal["mask_area_pct"])
    area_penalty = max(0.0, area_pct - args.target_area_pct) / max(args.target_area_pct, 1e-6)
    box_penalty = max(0.0, bbox_pct - args.target_bbox_area_pct) / max(args.target_bbox_area_pct, 1e-6)
    filter_score = float(proposal["sam_score"]) + 0.2 * cluster_coverage + 0.1 * prompt_coverage - 0.2 * area_penalty - 0.2 * box_penalty
    reasons: list[str] = []
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
    status = "candidate" if not reasons else "rejected"
    return ProposalMetric(
        group_id=int(proposal["group_id"]),
        cluster_id=cluster_id,
        sample_size=int(proposal["sample_size"]),
        sample_index=int(proposal["sample_index"]),
        sam_score=round(float(proposal["sam_score"]), 6),
        mask_area_pct=round(area_pct, 4),
        bbox_area_pct=round(float(bbox_pct), 4),
        cluster_coverage=round(cluster_coverage, 4),
        cluster_points_inside=inside,
        cluster_points_total=len(points),
        prompt_coverage=round(prompt_coverage, 4),
        prompt_points_inside=prompt_inside,
        prompt_points_total=len(prompt_points),
        filter_score=round(float(filter_score), 6),
        filter_status=status,
        reject_reasons=reasons,
        retained_rank=0,
        bbox_xyxy=[int(v) for v in proposal["bbox_xyxy"]],
        mask_path=proposal["mask_path"],
        overlay_path=proposal["overlay_path"],
    )


def filter_candidates(metrics: list[ProposalMetric]) -> list[ProposalMetric]:
    return [metric for metric in metrics if metric.filter_status == "candidate"]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union else 0.0


def run_nms(candidates: list[ProposalMetric], nms_iou: float) -> list[ProposalMetric]:
    retained: list[ProposalMetric] = []
    by_cluster = sorted({metric.cluster_id for metric in candidates})
    for cluster_id in by_cluster:
        cluster = [metric for metric in candidates if metric.cluster_id == cluster_id]
        cluster.sort(key=lambda item: (item.filter_score, item.sam_score), reverse=True)
        kept_masks: list[np.ndarray] = []
        for metric in cluster:
            mask = load_mask(metric.mask_path)
            if all(mask_iou(mask, kept) < nms_iou for kept in kept_masks):
                retained.append(metric)
                kept_masks.append(mask)
    retained.sort(key=lambda item: (item.cluster_id, -item.filter_score))
    for rank, metric in enumerate(retained, start=1):
        metric.filter_status = "retained"
        metric.retained_rank = rank
    return retained


def draw_retained_overlay(image: Image.Image, retained: list[ProposalMetric]) -> Image.Image:
    canvas = image.convert("RGBA")
    palette = [(255, 40, 40, 110), (40, 180, 255, 110), (80, 220, 90, 110), (255, 190, 40, 110)]
    draw = ImageDraw.Draw(canvas)
    for idx, metric in enumerate(retained):
        mask = load_mask(metric.mask_path)
        color = Image.new("RGBA", canvas.size, palette[idx % len(palette)])
        alpha = Image.fromarray((mask.astype(np.uint8) * 110), mode="L")
        color.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, color)
        draw = ImageDraw.Draw(canvas)
        if metric.bbox_xyxy != [0, 0, 0, 0]:
            draw.rectangle(tuple(metric.bbox_xyxy), outline=(255, 255, 0), width=3)
        draw.text((max(6, metric.bbox_xyxy[0] + 4), max(6, metric.bbox_xyxy[1] + 4)), f"#{metric.retained_rank} g{metric.group_id}", fill=(255, 255, 255))
    out = canvas.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((4, out.height - 38, min(out.width - 4, 520), out.height - 4), fill=(0, 0, 0))
    draw.text((12, out.height - 28), f"retained masks: {len(retained)}", fill=(255, 255, 255))
    return out


def make_grid(metrics: list[ProposalMetric], columns: int = 3) -> Image.Image:
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


def save_filter_outputs(
    out_dir: Path,
    proposals_dir: Path,
    payload: dict[str, Any],
    image: Image.Image,
    metrics: list[ProposalMetric],
    retained: list[ProposalMetric],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row["reject_reasons"] = json.dumps(metric.reject_reasons, ensure_ascii=False)
        row["bbox_xyxy"] = json.dumps(metric.bbox_xyxy)
        rows.append(row)
    with (out_dir / "filtered_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(ProposalMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    retained_overlay = draw_retained_overlay(image, retained)
    retained_overlay.save(out_dir / "retained_overlay.png")
    make_grid(retained).save(out_dir / "retained_grid.png")

    candidate_count = sum(1 for metric in metrics if not metric.reject_reasons)
    result = {
        "proposals_dir": str(proposals_dir.relative_to(WORKSPACE_ROOT)),
        "target_image": payload["target_image"],
        "thresholds": {
            "min_score": args.min_score,
            "min_area_pct": args.min_area_pct,
            "max_area_pct": args.max_area_pct,
            "max_bbox_area_pct": args.max_bbox_area_pct,
            "min_cluster_coverage": args.min_cluster_coverage,
            "min_prompt_coverage": args.min_prompt_coverage,
            "nms_iou": args.nms_iou,
        },
        "proposal_count": len(metrics),
        "candidate_count": candidate_count,
        "retained_count": len(retained),
        "k_pred": len(retained),
        "metrics": [asdict(metric) for metric in metrics],
        "retained": [asdict(metric) for metric in retained],
    }
    (out_dir / "filtered_scores.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    reason_counts: dict[str, int] = {}
    for metric in metrics:
        for reason in metric.reject_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    command = (
        "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\filter_stage3_proposals.py "
        f"--proposals-dir {args.proposals_dir} --min-score {args.min_score} "
        f"--min-area-pct {args.min_area_pct} --max-area-pct {args.max_area_pct} "
        f"--max-bbox-area-pct {args.max_bbox_area_pct} --min-cluster-coverage {args.min_cluster_coverage} "
        f"--nms-iou {args.nms_iou} --out-dir {args.out_dir}"
    )
    summary = [
        "# Matcher Stage 3 Filtering Summary",
        "",
        f"- Proposals dir: `{result['proposals_dir']}`",
        f"- Proposal count: `{len(metrics)}`",
        f"- Candidate count after thresholds: `{result['candidate_count']}`",
        f"- Retained count / K_pred: `{len(retained)}`",
        f"- Reject reason counts: `{reason_counts}`",
        f"- Retained group ids: `{[metric.group_id for metric in retained]}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Artifacts",
        "",
        "- `filtered_scores.csv`",
        "- `filtered_scores.json`",
        "- `retained_overlay.png`",
        "- `retained_grid.png`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Stage 2 SAM proposals and estimate instance count K.")
    parser.add_argument("--proposals-dir", default="runs/research_direction/stage2_proposals/elephant_pair")
    parser.add_argument("--min-score", type=float, default=0.80)
    parser.add_argument("--min-area-pct", type=float, default=5.0)
    parser.add_argument("--max-area-pct", type=float, default=45.0)
    parser.add_argument("--max-bbox-area-pct", type=float, default=45.0)
    parser.add_argument("--min-cluster-coverage", type=float, default=0.30)
    parser.add_argument("--min-prompt-coverage", type=float, default=1.0)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--target-area-pct", type=float, default=25.0)
    parser.add_argument("--target-bbox-area-pct", type=float, default=30.0)
    parser.add_argument("--out-dir", default="runs/research_direction/stage3_filtering/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposals_dir = resolve_path(args.proposals_dir)
    out_dir = resolve_path(args.out_dir)
    payload, proposals, image = load_proposals(proposals_dir)
    stage2_dir = resolve_path(payload["stage2_dir"])
    cluster_points = load_cluster_points(stage2_dir)
    metrics = [compute_proposal_metrics(proposal, cluster_points, args) for proposal in proposals]
    candidates = filter_candidates(metrics)
    retained = run_nms(candidates, args.nms_iou)
    save_filter_outputs(out_dir, proposals_dir, payload, image, metrics, retained, args)
    print(f"Proposal count: {len(metrics)}")
    print(f"Candidates after thresholds: {len(candidates)}")
    print(f"Retained count / K_pred: {len(retained)}")
    print(f"Retained group ids: {[metric.group_id for metric in retained]}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
