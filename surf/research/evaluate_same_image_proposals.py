"""Evaluate same-image one-prompt multi-instance segmentation.

Pipeline: rough seed point -> SAM exemplar -> automatic SAM proposals -> DINO
prototype ranking -> instance NMS.  Ground truth is used only for reporting;
the proposal-selection decision has no access to category labels or masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from dinov2.models import vision_transformer as vits
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

SIZE, PATCH = 518, 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    p.add_argument("--dino-checkpoint", type=Path, required=True)
    p.add_argument("--dino-size", choices=("vit_base", "vit_large"), default="vit_base")
    p.add_argument("--seed", choices=("sam-point", "oracle-mask"), default="sam-point")
    p.add_argument("--margin", type=float, default=0.085, help="prototype-score margin below seed score")
    p.add_argument("--ranker", choices=("dino", "lab", "hybrid"), default="dino")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--crop-layers", type=int, default=0)
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--start-index", type=int, default=0, help="zero-based manifest offset; enables stable short batches")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_dino(path: Path, name: str, device: torch.device) -> torch.nn.Module:
    model = vits.__dict__[name](img_size=SIZE, patch_size=PATCH, init_values=1e-5, ffn_layer="mlp", block_chunks=0, qkv_bias=True, proj_bias=True, ffn_bias=True)
    state = torch.load(path, map_location="cpu")
    state = state.get("teacher", state)
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def features(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([MaybeToTensor(), make_normalize_transform(mean=(.485, .456, .406), std=(.229, .224, .225))])
    resized = Image.fromarray(image).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    with torch.inference_mode():
        tokens = model.forward_features(transform(resized).unsqueeze(0).to(device))["x_prenorm"][0, 1:]
    return F.normalize(tokens, dim=1).reshape(SIZE // PATCH, SIZE // PATCH, -1)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIZE, SIZE), Image.Resampling.NEAREST), dtype=bool)


def centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    return int(round(xs.mean())), int(round(ys.mean()))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def proposal_score(mask: np.ndarray, token_similarity: np.ndarray) -> float:
    pooled = resize_mask(mask).reshape(37, PATCH, 37, PATCH).mean((1, 3)) > .30
    return float(token_similarity[pooled].mean()) if pooled.any() else -1.0


def normalize(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return (array - array.min()) / max(float(array.max() - array.min()), 1e-6)


def nms(items: list[tuple[float, np.ndarray]], threshold: float = .65) -> list[tuple[float, np.ndarray]]:
    keep: list[tuple[float, np.ndarray]] = []
    for item in sorted(items, key=lambda row: row[0], reverse=True):
        if all(iou(item[1], previous[1]) < threshold for previous in keep):
            keep.append(item)
    return keep


def evaluate(predictions: list[np.ndarray], truth: list[np.ndarray]) -> tuple[float, float, float, float, int]:
    pairs = sorted(((iou(pred, gt), pi, gi) for pi, pred in enumerate(predictions) for gi, gt in enumerate(truth)), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for score, pi, gi in pairs:
        if score < .5 or pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); matched.append(score)
    precision = len(matched) / max(len(predictions), 1)
    recall = len(matched) / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    union_pred = np.logical_or.reduce(predictions) if predictions else np.zeros_like(truth[0])
    union_gt = np.logical_or.reduce(truth)
    return precision, recall, f1, iou(union_pred, union_gt), len(matched)


def main() -> None:
    args = parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_rows = list(csv.DictReader((args.dataset / "manifest.csv").open(encoding="utf-8")))
    rows = all_rows[args.start_index:args.start_index + args.max_images]
    sam = sam_model_registry[args.sam_type](checkpoint=str(args.sam_checkpoint)).to(device).eval()
    predictor = SamPredictor(sam)
    generator = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side, pred_iou_thresh=.86, stability_score_thresh=.88, crop_n_layers=args.crop_layers, crop_n_points_downscale_factor=2, min_mask_region_area=100)
    dino = load_dino(args.dino_checkpoint, args.dino_size, device)
    results = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter(); image = np.asarray(Image.open(args.dataset / row["image"]).convert("RGB"))
        truth = [np.asarray(Image.open(p).convert("L"), dtype=bool) for p in sorted((args.dataset / row["instance_mask_dir"]).glob("*.png"))]
        if args.seed == "oracle-mask": exemplar = truth[0]
        else:
            predictor.set_image(image); x, y = centroid(truth[0]); masks, scores, _ = predictor.predict(point_coords=np.asarray([[x, y]], np.float32), point_labels=np.asarray([1], np.int32), multimask_output=True); exemplar = masks[int(np.argmax(scores))]
        # Do not keep the DINO activations on GPU while AMG builds its crop
        # pyramid. This matters more than raw speed for a stable multi-image run.
        if device.type == "cuda":
            dino.to("cpu"); torch.cuda.empty_cache()
        proposals = generator.generate(image)
        if device.type == "cuda":
            dino.to(device); torch.cuda.empty_cache()
        token_features = features(dino, image, device); seed_patches = resize_mask(exemplar).reshape(37, PATCH, 37, PATCH).mean((1, 3)) > .25
        prototype = F.normalize(token_features[seed_patches].mean(0), dim=0); similarity = (token_features @ prototype).detach().cpu().numpy()
        dino_scores = [proposal_score(p["segmentation"], similarity) for p in proposals]
        lab = rgb2lab(image)
        exemplar_colour, exemplar_area = lab[exemplar].mean(0), max(float(exemplar.sum()), 1.0)
        lab_scores = [float(np.exp(-np.linalg.norm(lab[p["segmentation"]].mean(0) - exemplar_colour) / 30.0) - .10 * abs(np.log(max(float(p["segmentation"].sum()), 1.0) / exemplar_area))) for p in proposals]
        if args.ranker == "dino": scores = dino_scores; seed_score = proposal_score(exemplar, similarity)
        elif args.ranker == "lab": scores = lab_scores; seed_score = 1.0
        else: scores = (.65 * normalize(dino_scores) + .35 * normalize(lab_scores)).tolist(); seed_score = .65 * float(normalize(dino_scores).max()) + .35
        ranked = [(score, p["segmentation"].astype(bool)) for score, p in zip(scores, proposals)]
        selected = nms([(score, mask) for score, mask in ranked if score >= seed_score - args.margin])
        # Include the prompted exemplar even if AMG did not duplicate it.
        selected = nms(selected + [(seed_score, exemplar.astype(bool))])
        pred = [mask for _, mask in selected]
        precision, recall, f1, union_iou, matched = evaluate(pred, truth)
        result = dict(image=row["image"], class_name=row["class_name"], gt_instances=len(truth), predicted_instances=len(pred), matched_instances=matched, precision=precision, recall=recall, f1=f1, union_iou=union_iou, seed_iou=iou(exemplar, truth[0]), proposals=len(proposals), elapsed_sec=time.perf_counter() - started)
        results.append(result); print(f"[{index}/{len(rows)}] {row['image']} F1={f1:.3f} R={recall:.3f} P={precision:.3f} K={len(pred)}/{len(truth)}", flush=True)
        # AMG can retain large crop tensors across cases on Windows CUDA builds.
        # Explicit release makes a 20-image sweep stable on modest GPUs.
        del proposals, ranked, selected, pred, token_features, similarity, dino_scores, lab_scores, scores, lab
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (args.out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    summary = {key: float(np.mean([r[key] for r in results])) for key in ("precision", "recall", "f1", "union_iou", "seed_iou", "elapsed_sec")}
    summary.update(images=len(results), start_index=args.start_index, sam_type=args.sam_type, dino_size=args.dino_size, seed=args.seed, ranker=args.ranker, margin=args.margin, points_per_side=args.points_per_side, crop_layers=args.crop_layers)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
