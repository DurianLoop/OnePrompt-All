from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


@dataclass
class ProposalResult:
    group_id: int
    cluster_id: int
    sample_size: int
    sample_index: int
    point_count: int
    sam_score: float
    mask_area_px: int
    mask_area_pct: float
    bbox_xyxy: list[int]
    mask_path: str
    overlay_path: str
    prompt_points: list[dict[str, float]]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def load_prompt_groups(stage2_dir: Path, max_groups: int) -> tuple[dict[str, Any], list[dict[str, Any]], Image.Image]:
    prompt_json = stage2_dir / "prompt_groups.json"
    if not prompt_json.exists():
        raise FileNotFoundError(prompt_json)
    payload = json.loads(prompt_json.read_text(encoding="utf-8"))
    groups = payload["prompt_groups"]
    if max_groups > 0:
        groups = groups[:max_groups]
    target = Image.open(resolve_path(payload["target_image"])).convert("RGB")
    return payload, groups, target


def clean_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_sam_predictor(checkpoint: Path, model_type: str, device_text: str) -> tuple[SamPredictor, torch.device]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = clean_device(device_text)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam), device


def points_from_group(group: dict[str, Any]) -> np.ndarray:
    points = [[float(point["x"]), float(point["y"])] for point in group["prompt_points"]]
    if not points:
        raise ValueError(f"Prompt group {group.get('group_id')} has no points")
    return np.asarray(points, dtype=np.float32)


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def predict_prompt_group(predictor: SamPredictor, group: dict[str, Any]) -> tuple[np.ndarray, float]:
    point_coords = points_from_group(group)
    point_labels = np.ones((len(point_coords),), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def overlay_mask(image: Image.Image, mask: np.ndarray, points: np.ndarray, score: float, group_id: int) -> Image.Image:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (255, 40, 40, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 115), mode="L")
    color.putalpha(alpha)
    merged = Image.alpha_composite(base, color).convert("RGB")
    draw = ImageDraw.Draw(merged)
    for x, y in points.tolist():
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(0, 220, 80), outline=(255, 255, 255), width=2)
    bbox = mask_bbox(mask)
    if bbox != [0, 0, 0, 0]:
        draw.rectangle(tuple(bbox), outline=(255, 255, 0), width=3)
    draw.rectangle((4, merged.height - 34, min(merged.width - 4, 410), merged.height - 4), fill=(0, 0, 0))
    draw.text((12, merged.height - 26), f"group {group_id} | score {score:.3f}", fill=(255, 255, 255))
    return merged


def make_grid(overlays: list[Image.Image], columns: int = 4) -> Image.Image:
    if not overlays:
        return Image.new("RGB", (320, 240), (255, 255, 255))
    thumb_w = 320
    thumb_h = int(round(overlays[0].height * (thumb_w / overlays[0].width)))
    thumbs = [image.resize((thumb_w, thumb_h), Image.BICUBIC) for image in overlays]
    rows = int(np.ceil(len(thumbs) / columns))
    grid = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_w
        y = (idx // columns) * thumb_h
        grid.paste(thumb, (x, y))
    return grid


def save_proposal_outputs(
    out_dir: Path,
    stage2_dir: Path,
    payload: dict[str, Any],
    groups: list[dict[str, Any]],
    target: Image.Image,
    predictor: SamPredictor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[ProposalResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "proposal_masks"
    overlay_dir = out_dir / "proposal_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    predictor.set_image(np.asarray(target))

    results: list[ProposalResult] = []
    overlays: list[Image.Image] = []
    for group in groups:
        mask, score = predict_prompt_group(predictor, group)
        group_id = int(group["group_id"])
        points = points_from_group(group)
        mask_path = mask_dir / f"group_{group_id:03d}.png"
        overlay_path = overlay_dir / f"group_{group_id:03d}.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
        overlay = overlay_mask(target, mask, points, score=score, group_id=group_id)
        overlay.save(overlay_path)
        overlays.append(overlay)
        area_px = int(mask.sum())
        results.append(
            ProposalResult(
                group_id=group_id,
                cluster_id=int(group["cluster_id"]),
                sample_size=int(group["sample_size"]),
                sample_index=int(group["sample_index"]),
                point_count=int(group["point_count"]),
                sam_score=round(float(score), 6),
                mask_area_px=area_px,
                mask_area_pct=round(100.0 * float(area_px) / float(mask.size), 4),
                bbox_xyxy=mask_bbox(mask),
                mask_path=str(mask_path.relative_to(WORKSPACE_ROOT)),
                overlay_path=str(overlay_path.relative_to(WORKSPACE_ROOT)),
                prompt_points=group["prompt_points"],
            )
        )

    rows = []
    for result in results:
        row = asdict(result)
        row["bbox_xyxy"] = json.dumps(row["bbox_xyxy"])
        row["prompt_points"] = json.dumps(row["prompt_points"], ensure_ascii=False)
        rows.append(row)
    with (out_dir / "proposal_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(ProposalResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    scores = [result.sam_score for result in results]
    areas = [result.mask_area_pct for result in results]
    result_payload = {
        "stage2_dir": str(stage2_dir.relative_to(WORKSPACE_ROOT)),
        "target_image": payload["target_image"],
        "checkpoint": str(resolve_path(args.checkpoint).relative_to(WORKSPACE_ROOT)),
        "model_type": args.model_type,
        "device": str(device),
        "max_groups": args.max_groups,
        "proposal_count": len(results),
        "score_mean": round(float(np.mean(scores)), 6) if scores else None,
        "score_min": round(float(np.min(scores)), 6) if scores else None,
        "score_max": round(float(np.max(scores)), 6) if scores else None,
        "area_pct_mean": round(float(np.mean(areas)), 4) if areas else None,
        "area_pct_min": round(float(np.min(areas)), 4) if areas else None,
        "area_pct_max": round(float(np.max(areas)), 4) if areas else None,
        "proposals": [asdict(result) for result in results],
    }
    (out_dir / "proposal_scores.json").write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    make_grid(overlays).save(out_dir / "proposal_grid.png")

    command = (
        "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\generate_stage2_sam_proposals.py "
        f"--stage2-dir {args.stage2_dir} --checkpoint {args.checkpoint} --model-type {args.model_type} "
        f"--device {args.device} --max-groups {args.max_groups} --out-dir {args.out_dir}"
    )
    summary = [
        "# Matcher Stage 2 SAM Proposal Summary",
        "",
        f"- Stage2 dir: `{result_payload['stage2_dir']}`",
        f"- Model type: `{args.model_type}`",
        f"- Device: `{device}`",
        f"- Proposal count: `{len(results)}`",
        f"- Score mean/min/max: `{result_payload['score_mean']}` / `{result_payload['score_min']}` / `{result_payload['score_max']}`",
        f"- Area pct mean/min/max: `{result_payload['area_pct_mean']}` / `{result_payload['area_pct_min']}` / `{result_payload['area_pct_max']}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Artifacts",
        "",
        "- `proposal_masks/`",
        "- `proposal_overlays/`",
        "- `proposal_scores.csv`",
        "- `proposal_scores.json`",
        "- `proposal_grid.png`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM mask proposals from Stage 2 sampled point prompt groups.")
    parser.add_argument("--stage2-dir", default="runs/research_direction/stage2_sampling/elephant_pair")
    parser.add_argument("--checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--out-dir", default="runs/research_direction/stage2_proposals/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2_dir = resolve_path(args.stage2_dir)
    out_dir = resolve_path(args.out_dir)
    payload, groups, target = load_prompt_groups(stage2_dir, max_groups=args.max_groups)
    predictor, device = load_sam_predictor(resolve_path(args.checkpoint), model_type=args.model_type, device_text=args.device)
    results = save_proposal_outputs(
        out_dir=out_dir,
        stage2_dir=stage2_dir,
        payload=payload,
        groups=groups,
        target=target,
        predictor=predictor,
        device=device,
        args=args,
    )
    scores = [result.sam_score for result in results]
    print(f"Proposal count: {len(results)}")
    if scores:
        print(f"Score mean/min/max: {float(np.mean(scores)):.6f}/{float(np.min(scores)):.6f}/{float(np.max(scores)):.6f}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
