from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SamMed2DArgs:
    image_size: int
    sam_checkpoint: str
    encoder_adapter: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    dice = 2.0 * inter / (pred_sum + gt_sum) if pred_sum + gt_sum else 1.0
    iou = inter / union if union else 1.0
    return dice, iou


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        import cv2

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32), int(count - 1)
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        current = 0
        h, w = mask.shape
        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0, x0] or labels[y0, x0] != 0:
                    continue
                current += 1
                stack = [(y0, x0)]
                labels[y0, x0] = current
                while stack:
                    y, x = stack.pop()
                    for ny in range(max(0, y - 1), min(h, y + 2)):
                        for nx in range(max(0, x - 1), min(w, x + 2)):
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((ny, nx))
        return labels, current


def object_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt_labels, gt_count = connected_components(gt)
    pred_labels, pred_count = connected_components(pred)
    gt_hit = 0
    for label in range(1, gt_count + 1):
        region = gt_labels == label
        if np.logical_and(region, pred).any():
            gt_hit += 1
    pred_hit = 0
    merge_count = 0
    for label in range(1, pred_count + 1):
        region = pred_labels == label
        overlapped = np.unique(gt_labels[region & gt])
        overlapped = overlapped[overlapped > 0]
        if len(overlapped) > 0:
            pred_hit += 1
        if len(overlapped) > 1:
            merge_count += 1
    return {
        "object_recall": gt_hit / gt_count if gt_count else 1.0,
        "object_precision": pred_hit / pred_count if pred_count else 0.0,
        "gt_components": float(gt_count),
        "pred_components": float(pred_count),
        "merge_error": merge_count / pred_count if pred_count else 0.0,
    }


def load_sammed2d_predictor(repo_root: Path, checkpoint: Path, device: str, image_size: int, encoder_adapter: bool):
    sys.path.insert(0, str(repo_root))
    from segment_anything import sam_model_registry  # type: ignore

    args = SamMed2DArgs(image_size=image_size, sam_checkpoint=str(checkpoint), encoder_adapter=encoder_adapter)
    original_torch_load = torch.load

    def trusted_checkpoint_load(*load_args: Any, **load_kwargs: Any):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load  # type: ignore[assignment]
    try:
        model = sam_model_registry["vit_b"](args).to(device)
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]
    model.eval()
    return LightweightSamMed2DPredictor(model)


class LightweightSamMed2DPredictor:
    """Minimal SAM-Med2D predictor that avoids the optional albumentations dependency."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.features: torch.Tensor | None = None
        self.original_size: tuple[int, int] | None = None
        self.image_size = int(model.image_encoder.img_size)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def set_image(self, image: np.ndarray) -> None:
        if image.ndim != 3:
            raise ValueError("Expected HWC RGB image")
        self.original_size = (int(image.shape[0]), int(image.shape[1]))
        image = image.astype(np.float32)
        pixel_mean = self.model.pixel_mean.detach().cpu().numpy().reshape(3)
        pixel_std = self.model.pixel_std.detach().cpu().numpy().reshape(3)
        normalized = (image - pixel_mean) / pixel_std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(tensor, (self.image_size, self.image_size), mode="nearest")
        with torch.no_grad():
            self.features = self.model.image_encoder(tensor.to(self.device))

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.features is None or self.original_size is None:
            raise RuntimeError("set_image must be called first")
        old_h, old_w = self.original_size
        coords = point_coords.astype(np.float32).copy()
        coords[:, 0] *= self.image_size / float(old_w)
        coords[:, 1] *= self.image_size / float(old_h)
        coords_torch = torch.as_tensor(coords, dtype=torch.float32, device=self.device)[None, :, :]
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int64, device=self.device)[None, :]
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(coords_torch, labels_torch),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            if multimask_output:
                max_values, max_indexes = torch.max(iou_predictions, dim=1)
                picked = []
                for batch_index, mask_index in enumerate(max_indexes.tolist()):
                    picked.append(low_res_masks[batch_index : batch_index + 1, mask_index : mask_index + 1])
                low_res_masks = torch.cat(picked, dim=0)
                iou_predictions = max_values[:, None]
            masks = F.interpolate(low_res_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, self.original_size, mode="bilinear", align_corners=False)
            binary = (torch.sigmoid(masks) > 0.5).float()
        return (
            binary[0].detach().cpu().numpy(),
            iou_predictions[0].detach().cpu().numpy(),
            low_res_masks[0].detach().cpu().numpy(),
        )


def points_array(points_path: Path, max_points: int | None) -> np.ndarray:
    rows = read_csv(points_path)
    if max_points is not None:
        rows = rows[:max_points]
    return np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)


def save_overlay(image: Image.Image, gt: np.ndarray, pred: np.ndarray, points: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    gt_layer = Image.new("RGBA", base.size, (0, 170, 255, 0))
    pred_layer = Image.new("RGBA", base.size, (255, 40, 40, 0))
    gt_layer.putalpha(Image.fromarray((gt.astype(np.uint8) * 70), mode="L"))
    pred_layer.putalpha(Image.fromarray((pred.astype(np.uint8) * 105), mode="L"))
    out = Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer).convert("RGB")
    draw = ImageDraw.Draw(out)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 255, 90), outline=(255, 255, 255), width=2)
    draw.rectangle((4, out.height - 42, min(out.width - 4, 980), out.height - 4), fill=(0, 0, 0))
    draw.text((12, out.height - 32), title, fill=(255, 255, 255))
    out.save(path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["dice", "iou", "fg_precision", "fg_recall", "object_recall", "object_precision", "merge_error", "pred_area_pct"]
    out: dict[str, Any] = {"case_count": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        out[f"mean_{metric}"] = round(float(values.mean()), 6) if len(values) else 0.0
        out[f"std_{metric}"] = round(float(values.std(ddof=0)), 6) if len(values) else 0.0
    return out


def write_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    top = sorted(rows, key=lambda row: float(row["dice"]), reverse=True)
    cards = []
    for row in top[:3] + top[-3:]:
        overlay_rel = os.path.relpath(resolve_path(row["overlay_path"]), report_dir).replace("\\", "/")
        cards.append(
            f"""
            <section>
              <h2>{html.escape(row['query_case_id'])}</h2>
              <p>Dice {row['dice']} | IoU {row['iou']} | object recall {row['object_recall']} | pred area {row['pred_area_pct']}%</p>
              <a href="{overlay_rel}" target="_blank"><img src="{overlay_rel}" alt="{html.escape(row['query_case_id'])}"></a>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAM-Med2D from Matcher Points</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 0; color: #172033; background: #fff; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 28px 22px 60px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
p {{ color: #526174; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 28px; }}
.metrics div {{ border: 1px solid #d9e2ee; border-radius: 8px; padding: 12px; background: #f7f9fc; }}
.metrics b {{ display: block; font-size: 22px; color: #172033; }}
section {{ border-top: 1px solid #d9e2ee; padding-top: 18px; margin-top: 24px; }}
img {{ display: block; width: 100%; max-height: 760px; object-fit: contain; background: #f1f5f9; border: 1px solid #d9e2ee; }}
@media (max-width: 760px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
<h1>SAM-Med2D 接收 matcher dense points 的快速评估</h1>
<p>Matcher 固定为当前最好配置 DINOv2-L + K=40 + lambda=0.75，只替换 segmenter 为 SAM-Med2D。</p>
<div class="metrics">
  <div><span>Mean Dice</span><b>{summary['mean_dice']}</b></div>
  <div><span>Mean IoU</span><b>{summary['mean_iou']}</b></div>
  <div><span>FG Recall</span><b>{summary['mean_fg_recall']}</b></div>
  <div><span>Pred Area %</span><b>{summary['mean_pred_area_pct']}</b></div>
</div>
{''.join(cards)}
</main></body></html>"""
    (report_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="runs/research_direction/monuseg_official_screening/official_monuseg_screening_manifest.csv")
    parser.add_argument("--source-run", default="runs/research_direction/monuseg_official_screening/dinov2l_samvitb_kxlambda_test14")
    parser.add_argument("--out-dir", default="runs/research_direction/monuseg_official_screening/sammed2d_from_dinov2l_points_test14")
    parser.add_argument("--checkpoint", default="assets/checkpoints/sam_med2d_b.pth")
    parser.add_argument("--repo-root", default="SAM-Med2D")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=40)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--no-encoder-adapter", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(resolve_path(args.manifest))
    if args.case_limit:
        manifest = manifest[: args.case_limit]
    source_rows = read_csv(resolve_path(args.source_run) / "candidates.csv")
    best_by_case = {
        row["case_name"]: row
        for row in source_rows
        if row["round_name"] == "round3" and row["method"] == "best_topk"
    }
    predictor = load_sammed2d_predictor(
        resolve_path(args.repo_root),
        resolve_path(args.checkpoint),
        args.device,
        args.image_size,
        not args.no_encoder_adapter,
    )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(manifest, 1):
        case_name = item["pilot_id"]
        if case_name not in best_by_case:
            continue
        source = best_by_case[case_name]
        image = Image.open(resolve_path(item["query_image_path"])).convert("RGB")
        gt = load_mask(resolve_path(item["query_mask_path"]))
        points = points_array(resolve_path(source["points_path"]), args.max_points)
        labels = np.ones((len(points),), dtype=np.int32)
        predictor.set_image(np.asarray(image))
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
        masks = np.asarray(masks)
        if masks.ndim == 3:
            best = int(np.argmax(np.asarray(scores).reshape(-1)))
            pred = masks[best].astype(bool)
            sam_score = float(np.asarray(scores).reshape(-1)[best])
        else:
            pred = masks.squeeze().astype(bool)
            sam_score = float(np.asarray(scores).reshape(-1)[0]) if np.asarray(scores).size else 0.0
        dice, iou = dice_iou(pred, gt)
        pred_sum = float(pred.sum())
        gt_sum = float(gt.sum())
        inter = float(np.logical_and(pred, gt).sum())
        obj = object_metrics(pred, gt)
        mask_path = out_dir / "masks" / f"{item['query_case_id']}.png"
        overlay_path = out_dir / "overlays" / f"{item['query_case_id']}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred.astype(np.uint8) * 255), mode="L").save(mask_path)
        save_overlay(image, gt, pred, points, overlay_path, f"{item['query_case_id']} | SAM-Med2D | Dice {dice:.3f}")
        row = {
            "case_name": case_name,
            "query_case_id": item["query_case_id"],
            "source_method": source["method"],
            "k": len(points),
            "sammed2d_score": round(sam_score, 6),
            "dice": round(dice, 6),
            "iou": round(iou, 6),
            "fg_precision": round(inter / pred_sum if pred_sum else 0.0, 6),
            "fg_recall": round(inter / gt_sum if gt_sum else 1.0, 6),
            "object_recall": round(float(obj["object_recall"]), 6),
            "object_precision": round(float(obj["object_precision"]), 6),
            "gt_components": int(obj["gt_components"]),
            "pred_components": int(obj["pred_components"]),
            "merge_error": round(float(obj["merge_error"]), 6),
            "pred_area_pct": round(100.0 * pred_sum / float(pred.size), 6),
            "mask_path": os.path.relpath(mask_path, WORKSPACE_ROOT).replace("\\", "/"),
            "overlay_path": os.path.relpath(overlay_path, WORKSPACE_ROOT).replace("\\", "/"),
        }
        rows.append(row)
        print(f"[{index}/{len(manifest)}] {item['query_case_id']} dice={dice:.4f} area={row['pred_area_pct']}")

    write_csv(out_dir / "case_metrics.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "summary.csv", [summary])
    write_report(out_dir, rows, summary)
    print(f"[done] cases={len(rows)} mean_dice={summary['mean_dice']} report={out_dir / 'report' / 'index.html'}")


if __name__ == "__main__":
    main()
