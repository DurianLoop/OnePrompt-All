"""Evaluate native SAM3 one-positive-point multi-instance predictions.

The GT instance masks are used *only* to synthesize one reproducible click and
to score the output.  At inference, SAM3 receives the RGB image plus exactly
one positive point.  Its returned masks are evaluated as separate instances.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SAM3_ROOT = WORKSPACE / "external" / "sam3"
for path in (ROOT, SAM3_ROOT / ".deps_runtime", SAM3_ROOT):
    sys.path.insert(0, str(path))

from demos.medical_sam_click_app import PointPrompt, Sam3TextSegmenter  # noqa: E402
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform  # noqa: E402
from dinov2.models import vision_transformer as vits  # noqa: E402

SIZE, PATCH = 518, 14


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--nms-iou", type=float, default=.65)
    parser.add_argument("--dino-checkpoint", type=Path, help="optional DINOv2-B visual consistency filter")
    parser.add_argument("--dino-size", choices=("vit_base", "vit_large"), default="vit_base")
    parser.add_argument("--dino-mode", choices=("off", "nms", "filter"), default="off")
    parser.add_argument("--dino-margin", type=float, default=.13)
    parser.add_argument("--dino-nms-weight", type=float, default=.2, help="DINO contribution when selecting among overlapping SAM3 masks")
    parser.add_argument("--dino-propagate-points", type=int, default=0, help="additional DINO-discovered same-object virtual queries")
    parser.add_argument("--min-area-ratio", type=float, default=0., help="discard implausibly tiny instance fragments")
    parser.add_argument("--save-viz", action="store_true", help="write predicted-instance overlays and GT comparison sheets")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def nms(masks: np.ndarray, scores: np.ndarray, threshold: float = .65) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for index in np.argsort(-scores):
        mask = masks[index].astype(bool)
        if mask.any() and all(iou(mask, previous) < threshold for previous in kept):
            kept.append(mask)
    return kept


def metrics(pred: list[np.ndarray], truth: list[np.ndarray]) -> tuple[float, float, float, float, int]:
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(truth)), reverse=True)
    used_pred, used_truth, hits = set(), set(), 0
    for score, pi, gi in pairs:
        if score < .5 or pi in used_pred or gi in used_truth:
            continue
        used_pred.add(pi); used_truth.add(gi); hits += 1
    precision = hits / max(len(pred), 1)
    recall = hits / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    pred_union = np.logical_or.reduce(pred) if pred else np.zeros_like(truth[0])
    return precision, recall, f1, iou(pred_union, np.logical_or.reduce(truth)), hits


def centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    return int(round(xs.mean())), int(round(ys.mean()))


PALETTE = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166), (236, 112, 99)]


def save_visualization(image: np.ndarray, prediction: list[np.ndarray], truth: list[np.ndarray], point: tuple[int, int], target: Path, title: str) -> None:
    """Save input, final prediction, and an explicitly separate GT panel."""
    base = Image.fromarray(image).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    label_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for index, mask in enumerate(prediction):
        color = PALETTE[index % len(PALETTE)]
        overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        overlay[mask] = (*color, 105)
        layer.alpha_composite(Image.fromarray(overlay, "RGBA"))
        label_map[mask] = color
    # A black background makes the final segmentation unambiguous: every
    # colored pixel is a predicted target instance, not an annotation source.
    prediction_image = Image.fromarray(label_map, "RGB")
    input_image = Image.fromarray(image).convert("RGB")
    input_draw = ImageDraw.Draw(input_image)
    px, py = point
    input_draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(230, 45, 45), outline=(255, 255, 255), width=2)
    gt_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for mask in truth:
        gt_map[mask] = (61, 143, 230)
    gt_image = Image.fromarray(gt_map, "RGB")
    sheet = Image.new("RGB", (image.shape[1] * 3, image.shape[0] + 24), "white")
    sheet.paste(input_image, (0, 24)); sheet.paste(prediction_image, (image.shape[1], 24)); sheet.paste(gt_image, (image.shape[1] * 2, 24))
    header = ImageDraw.Draw(sheet)
    header.text((5, 5), "Input: one positive point (red)", fill="black")
    header.text((image.shape[1] + 5, 5), f"OnePrompt-All output: {len(prediction)} predicted instance masks | {title}", fill="black")
    header.text((image.shape[1] * 2 + 5, 5), f"GT (evaluation only): {len(truth)} instance masks", fill="black")
    sheet.save(target)


def load_dino(path: Path, device: torch.device, name: str = "vit_base") -> torch.nn.Module:
    model = vits.__dict__[name](img_size=SIZE, patch_size=PATCH, init_values=1e-5, ffn_layer="mlp", block_chunks=0, qkv_bias=True, proj_bias=True, ffn_bias=True)
    state = torch.load(path, map_location="cpu")
    state = state.get("teacher", state)
    state = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def token_features(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([MaybeToTensor(), make_normalize_transform(mean=(.485, .456, .406), std=(.229, .224, .225))])
    resized = Image.fromarray(image).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    with torch.inference_mode():
        result = model.forward_features(transform(resized).unsqueeze(0).to(device))["x_prenorm"][0, 1:]
    return F.normalize(result, dim=1).reshape(37, 37, -1)


def patch_mask(mask: np.ndarray) -> np.ndarray:
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=bool)
    return resized.reshape(37, PATCH, 37, PATCH).mean((1, 3)) > .25


def propagated_points(image: np.ndarray, masks: np.ndarray, scores: np.ndarray, point: tuple[int, int], dino: torch.nn.Module, device: torch.device, count: int) -> list[tuple[int, int]]:
    """Find separated DINO-similar patches outside the clicked SAM3 seed."""
    if not len(masks) or count <= 0:
        return []
    x, y = point
    contains = [i for i, mask in enumerate(masks) if mask[y, x]]
    if not contains:
        return []
    seed = masks[max(contains, key=lambda i: float(scores[i]))]
    cells = patch_mask(seed)
    if not cells.any():
        return []
    feats = token_features(dino, image, device)
    prototype = F.normalize(feats[cells].mean(0), dim=0)
    similarity = (feats @ prototype).detach().cpu().numpy()
    similarity[cells] = -np.inf
    selected: list[tuple[int, int]] = []
    for flat in np.argsort(similarity.ravel())[::-1]:
        ty, tx = np.unravel_index(int(flat), similarity.shape)
        if not np.isfinite(similarity[ty, tx]):
            break
        if any((tx - sx) ** 2 + (ty - sy) ** 2 < 16 for sx, sy in selected):
            continue
        selected.append((int(tx), int(ty)))
        if len(selected) == count:
            break
    return [(min(image.shape[1] - 1, int((tx + .5) * image.shape[1] / 37)), min(image.shape[0] - 1, int((ty + .5) * image.shape[0] / 37))) for tx, ty in selected]


def main() -> None:
    cfg = args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    rows = list(csv.DictReader((cfg.dataset / "manifest.csv").open(encoding="utf-8")))
    rows = rows[cfg.start_index:cfg.start_index + cfg.max_images]
    segmenter = Sam3TextSegmenter(cfg.checkpoint, device=device, threshold=cfg.threshold)
    dino = load_dino(cfg.dino_checkpoint, torch_device, cfg.dino_size) if cfg.dino_checkpoint and (cfg.dino_mode != "off" or cfg.dino_propagate_points > 0) else None
    viz_dir = cfg.out_dir / "visualizations"
    if cfg.save_viz:
        viz_dir.mkdir(exist_ok=True)
    output: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        image = np.asarray(Image.open(cfg.dataset / row["image"]).convert("RGB"))
        truth = [np.asarray(Image.open(path).convert("L"), dtype=bool) for path in sorted((cfg.dataset / row["instance_mask_dir"]).glob("*.png"))]
        x, y = centroid(truth[0])
        masks, scores = segmenter.predict_interactive(image, "click", [PointPrompt(x=x, y=y, label=1)], None, "manual")
        nms_scores = scores.copy()
        if dino is not None and cfg.dino_propagate_points:
            extra_masks, extra_scores = [], []
            for vx, vy in propagated_points(image, masks, scores, (x, y), dino, torch_device, cfg.dino_propagate_points):
                virtual_masks, virtual_scores = segmenter.predict_interactive(image, "click", [PointPrompt(x=vx, y=vy, label=1)], None, "manual")
                extra_masks.append(virtual_masks); extra_scores.append(virtual_scores)
            if extra_masks:
                masks = np.concatenate([masks, *extra_masks], axis=0)
                scores = np.concatenate([scores, *extra_scores], axis=0)
                nms_scores = scores.copy()
        if dino is not None and len(masks):
            feats = token_features(dino, image, torch_device)
            prompt_candidates = [i for i, mask in enumerate(masks) if mask[y, x]]
            exemplar_id = max(prompt_candidates or list(range(len(masks))), key=lambda i: float(scores[i]))
            seed_cells = patch_mask(masks[exemplar_id])
            prototype = F.normalize(feats[seed_cells].mean(0), dim=0)
            visual_scores = []
            for mask in masks:
                cells = patch_mask(mask)
                visual_scores.append(float((feats[cells] @ prototype).mean()) if cells.any() else -1.)
            seed_visual = visual_scores[exemplar_id]
            if cfg.dino_mode == "filter":
                selected_ids = [i for i, score in enumerate(visual_scores) if score >= seed_visual - cfg.dino_margin]
                masks, scores = masks[selected_ids], scores[selected_ids]
                visual_scores = [visual_scores[i] for i in selected_ids]
            if cfg.dino_mode == "nms" and len(masks) > 1:
                def norm(values: np.ndarray) -> np.ndarray:
                    return (values - values.min()) / max(float(values.max() - values.min()), 1e-6)
                weight = float(np.clip(cfg.dino_nms_weight, 0., 1.))
                nms_scores = (1. - weight) * norm(scores) + weight * norm(np.asarray(visual_scores, dtype=np.float32))
            else:
                nms_scores = scores.copy()
        if cfg.min_area_ratio > 0 and len(masks):
            min_area = image.shape[0] * image.shape[1] * cfg.min_area_ratio
            kept_ids = [i for i, mask in enumerate(masks) if float(mask.sum()) >= min_area]
            masks, scores, nms_scores = masks[kept_ids], scores[kept_ids], nms_scores[kept_ids]
        pred = nms(masks, nms_scores, cfg.nms_iou)
        precision, recall, f1, union_iou, matched = metrics(pred, truth)
        item = dict(image=row["image"], gt_instances=len(truth), raw_masks=int(len(masks)), predicted_instances=len(pred), matched_instances=matched, precision=precision, recall=recall, f1=f1, union_iou=union_iou, elapsed_sec=time.perf_counter() - started)
        output.append(item)
        if cfg.save_viz:
            save_visualization(image, pred, truth, (x, y), viz_dir / f"{index:02d}_{Path(row['image']).stem}.jpg", f"F1={f1:.3f}")
        print(f"[{index}/{len(rows)}] {row['image']} F1={f1:.3f} R={recall:.3f} P={precision:.3f} K={len(pred)}/{len(truth)}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    with (cfg.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    summary = {name: float(np.mean([float(row[name]) for row in output])) for name in ("precision", "recall", "f1", "union_iou", "elapsed_sec")}
    summary.update(images=len(output), threshold=cfg.threshold, nms_iou=cfg.nms_iou, start_index=cfg.start_index)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
