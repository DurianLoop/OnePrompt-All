"""Measure OnePrompt-All sensitivity to the location of one valid click.

For each benchmark image this script simulates several *independent* user
clicks.  Each click is sampled from the interior of a different annotated
target only to make the experiment reproducible; inference receives exactly
one RGB image and one positive point, never a mask, class name, count, or
second prompt.
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
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demos.medical_sam_click_app import PointPrompt, Sam3TextSegmenter  # noqa: E402
from research.evaluate_sam3_onepoint import load_dino, metrics, nms, patch_mask, token_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-size", choices=("vit_base", "vit_large"), default="vit_base")
    parser.add_argument("--prompts-per-image", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None, help="optional cap after deterministic striding, for ablations")
    parser.add_argument("--image-stride", type=int, default=1, help="evaluate every Nth manifest image (>=1)")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threshold", type=float, default=.45)
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--min-area-ratio", type=float, default=.001)
    parser.add_argument("--dino-nms-weight", type=float, default=.2)
    parser.add_argument("--candidate-filter-margin", type=float, default=None, help="keep masks whose DINO cosine is within this margin of the clicked seed; omit for no final filter")
    parser.add_argument("--canonicalize-click", action="store_true", help="use SAM3 seed + DINO feature core to issue one internal re-click")
    parser.add_argument("--negative-points", type=int, default=0, help="DINO-selected background negatives used only after the initial positive click")
    parser.add_argument("--negative-policy", choices=("fixed", "adaptive"), default="fixed", help="fixed uses exactly N points; adaptive chooses 0..N from the seed/background DINO separation")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sampled_point(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Choose an arbitrary but reliable interior point, not the centroid."""
    distance = distance_transform_edt(mask)
    candidates = np.argwhere(distance >= max(2.0, float(distance.max()) * .25))
    if len(candidates) == 0:
        candidates = np.argwhere(mask)
    y, x = candidates[int(rng.integers(len(candidates)))]
    return int(x), int(y)


def dino_candidate_scores(image: np.ndarray, masks: np.ndarray, scores: np.ndarray, point: tuple[int, int], dino: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, int]:
    """Cosine consistency of every SAM3 mask with the clicked seed."""
    if not len(masks):
        return np.empty(0, dtype=np.float32), -1
    x, y = point
    features = token_features(dino, image, device)
    contains_click = [i for i, mask in enumerate(masks) if mask[y, x]]
    exemplar = max(contains_click or list(range(len(masks))), key=lambda i: float(scores[i]))
    cells = patch_mask(masks[exemplar])
    prototype = torch.nn.functional.normalize(features[cells].mean(0), dim=0)
    visual: list[float] = []
    for mask in masks:
        cells = patch_mask(mask)
        visual.append(float((features[cells] @ prototype).mean()) if cells.any() else -1.)
    return np.asarray(visual, dtype=np.float32), exemplar


def soft_dino_nms_scores(scores: np.ndarray, visual: np.ndarray, weight: float) -> np.ndarray:
    """Use DINO only to break overlap ties, never to suppress a candidate."""
    if len(scores) < 2:
        return scores
    def norm(values: np.ndarray) -> np.ndarray:
        return (values - values.min()) / max(float(values.max() - values.min()), 1e-6)
    return (1. - weight) * norm(scores.astype(np.float32)) + weight * norm(visual)


def dino_canonical_point(image: np.ndarray, masks: np.ndarray, scores: np.ndarray, point: tuple[int, int], dino: torch.nn.Module, device: torch.device) -> tuple[int, int]:
    """Turn an arbitrary valid click into a semantically central seed point.

    The first SAM3 response selects only the mask containing the user's click.
    DINO then chooses a high-similarity, interior patch within that mask.  This
    is internal prompt normalization, not an extra user annotation.
    """
    x, y = point
    valid = [i for i, mask in enumerate(masks) if mask[y, x]]
    if not valid:
        return point
    seed = masks[max(valid, key=lambda i: float(scores[i]))]
    cells = patch_mask(seed)
    if not cells.any():
        return point
    features = token_features(dino, image, device)
    prototype = torch.nn.functional.normalize(features[cells].mean(0), dim=0)
    similarity = (features @ prototype).detach().cpu().numpy()
    interior = distance_transform_edt(cells)
    interior = interior / max(float(interior.max()), 1e-6)
    quality = similarity + .12 * interior
    quality[~cells] = -np.inf
    ty, tx = np.unravel_index(int(np.argmax(quality)), quality.shape)
    px = min(image.shape[1] - 1, max(0, int(round((tx + .5) * image.shape[1] / 37))))
    py = min(image.shape[0] - 1, max(0, int(round((ty + .5) * image.shape[0] / 37))))
    if seed[py, px]:
        return px, py
    # The token centre can fall on a resize boundary; return a nearby interior
    # pixel rather than placing a virtual click outside the initial seed.
    ys, xs = np.where(seed)
    nearest = np.argmin((xs - px) ** 2 + (ys - py) ** 2)
    return int(xs[nearest]), int(ys[nearest])


def dino_background_points(image: np.ndarray, masks: np.ndarray, scores: np.ndarray, point: tuple[int, int], dino: torch.nn.Module, device: torch.device, count: int, policy: str = "fixed") -> list[tuple[int, int]]:
    """Select nearby low-similarity patches as automatic background evidence."""
    if count <= 0 or not len(masks):
        return []
    x, y = point
    valid = [i for i, mask in enumerate(masks) if mask[y, x]]
    if not valid:
        return []
    seed = masks[max(valid, key=lambda i: float(scores[i]))]
    cells = patch_mask(seed)
    if not cells.any():
        return []
    features = token_features(dino, image, device)
    prototype = torch.nn.functional.normalize(features[cells].mean(0), dim=0)
    similarity = (features @ prototype).detach().cpu().numpy()
    # Prefer a close exterior ring, but only where DINO says it differs from
    # the clicked object.  This avoids treating a neighboring banana as bg.
    distance = distance_transform_edt(~cells)
    if policy == "adaptive":
        # A crowded ring that resembles the seed is likely another target
        # instance.  Such pixels must never become negative prompts.  Only
        # introduce negatives when DINO observes a genuine semantic boundary;
        # the gap also determines how many are justified (0..count).
        ring = (~cells) & (distance >= 1.) & (distance <= 2.5)
        if not ring.any():
            return []
        contrast = float(np.median(similarity[cells]) - np.median(similarity[ring]))
        if contrast < .08:
            return []
        count = min(count, 1 + int((contrast - .08) / .07))
        if count <= 0:
            return []
    quality = -similarity + .12 / (1. + distance)
    quality[cells] = -np.inf
    chosen: list[tuple[int, int]] = []
    for flat in np.argsort(quality.ravel())[::-1]:
        ty, tx = np.unravel_index(int(flat), quality.shape)
        if not np.isfinite(quality[ty, tx]):
            break
        if any((tx - sx) ** 2 + (ty - sy) ** 2 < 9 for sx, sy in chosen):
            continue
        chosen.append((int(tx), int(ty)))
        if len(chosen) >= count:
            break
    return [(min(image.shape[1] - 1, int((tx + .5) * image.shape[1] / 37)), min(image.shape[0] - 1, int((ty + .5) * image.shape[0] / 37))) for tx, ty in chosen]


def main() -> None:
    cfg = parse_args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"; device = torch.device(device_name)
    rows = list(csv.DictReader((cfg.dataset / "manifest.csv").open(encoding="utf-8")))
    if cfg.image_stride < 1:
        raise ValueError("--image-stride must be at least 1")
    rows = rows[::cfg.image_stride]
    if cfg.max_images is not None:
        rows = rows[:cfg.max_images]
    sam3 = Sam3TextSegmenter(cfg.checkpoint, device_name, cfg.threshold)
    dino = load_dino(cfg.dino_checkpoint, device, cfg.dino_size)
    rng = np.random.default_rng(cfg.seed)
    trials: list[dict[str, object]] = []
    for image_index, row in enumerate(rows, 1):
        image = np.asarray(Image.open(cfg.dataset / row["image"]).convert("RGB"))
        truth = [np.asarray(Image.open(path).convert("L"), dtype=bool) for path in sorted((cfg.dataset / row["instance_mask_dir"]).glob("*.png"))]
        # A real user may click different locations on the same instance.  Use
        # a deterministic uniform draw with replacement so --prompts-per-image
        # is honored even when an image has fewer annotated instances.
        selected = rng.integers(0, len(truth), size=cfg.prompts_per_image)
        for prompt_index, gt_index in enumerate(selected, 1):
            x, y = sampled_point(truth[int(gt_index)], rng); started = time.perf_counter()
            masks, scores = sam3.predict_interactive(image, "click", [PointPrompt(x=x, y=y, label=1)], None, "manual")
            canonical_x, canonical_y = x, y
            if cfg.canonicalize_click:
                canonical_x, canonical_y = dino_canonical_point(image, masks, scores, (x, y), dino, device)
                masks, scores = sam3.predict_interactive(image, "click", [PointPrompt(x=canonical_x, y=canonical_y, label=1)], None, "manual")
            if cfg.negative_points:
                negatives = dino_background_points(image, masks, scores, (canonical_x, canonical_y), dino, device, cfg.negative_points, cfg.negative_policy)
                prompts = [PointPrompt(x=canonical_x, y=canonical_y, label=1), *[PointPrompt(x=nx, y=ny, label=0) for nx, ny in negatives]]
                masks, scores = sam3.predict_interactive(image, "click", prompts, None, "manual")
            visual, exemplar = dino_candidate_scores(image, masks, scores, (canonical_x, canonical_y), dino, device)
            if cfg.candidate_filter_margin is not None and len(visual):
                keep = visual >= visual[exemplar] - cfg.candidate_filter_margin
                masks, scores, visual = masks[keep], scores[keep], visual[keep]
            nms_scores = soft_dino_nms_scores(scores, visual, cfg.dino_nms_weight)
            min_area = image.shape[0] * image.shape[1] * cfg.min_area_ratio
            keep = [i for i, mask in enumerate(masks) if float(mask.sum()) >= min_area]
            masks, nms_scores = masks[keep], nms_scores[keep]
            prediction = nms(masks, nms_scores, cfg.nms_iou)
            precision, recall, f1, union_iou, matched = metrics(prediction, truth)
            result = dict(image=row["image"], class_name=row["class_name"], prompt_index=prompt_index, prompt_gt_instance=int(gt_index) + 1, point_x=x, point_y=y, canonical_x=canonical_x, canonical_y=canonical_y, negative_policy=cfg.negative_policy, max_negative_points=cfg.negative_points, internal_negative_points=len(negatives) if cfg.negative_points else 0, candidate_filter_margin=cfg.candidate_filter_margin, gt_instances=len(truth), predicted_instances=len(prediction), matched_instances=matched, precision=precision, recall=recall, f1=f1, union_iou=union_iou, elapsed_sec=time.perf_counter() - started)
            trials.append(result)
            print(f"[{image_index}/{len(rows)} p{prompt_index}] F1={f1:.3f} K={len(prediction)}/{len(truth)} point=({x},{y})", flush=True)
            if device_name.startswith("cuda"):
                torch.cuda.empty_cache()
    fields = list(trials[0])
    with (cfg.out_dir / "per_prompt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trials)
    image_rows = []
    for image in dict.fromkeys(str(item["image"]) for item in trials):
        group = [item for item in trials if item["image"] == image]
        image_rows.append(dict(image=image, class_name=group[0]["class_name"], prompts=len(group), mean_f1=float(np.mean([item["f1"] for item in group])), std_f1=float(np.std([item["f1"] for item in group])), min_f1=float(np.min([item["f1"] for item in group])), max_f1=float(np.max([item["f1"] for item in group])), mean_precision=float(np.mean([item["precision"] for item in group])), mean_recall=float(np.mean([item["recall"] for item in group]))))
    with (cfg.out_dir / "per_image_prompt_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    summary = dict(images=len(image_rows), prompts=len(trials), prompts_per_image=cfg.prompts_per_image, image_stride=cfg.image_stride, max_images=cfg.max_images, negative_policy=cfg.negative_policy, max_negative_points=cfg.negative_points, mean_internal_negative_points=float(np.mean([item["internal_negative_points"] for item in trials])), mean_f1=float(np.mean([item["f1"] for item in trials])), std_f1=float(np.std([item["f1"] for item in trials])), mean_precision=float(np.mean([item["precision"] for item in trials])), mean_recall=float(np.mean([item["recall"] for item in trials])), mean_per_image_std_f1=float(np.mean([item["std_f1"] for item in image_rows])), worst_prompt_f1=float(np.min([item["f1"] for item in trials])), best_prompt_f1=float(np.max([item["f1"] for item in trials])), seed=cfg.seed)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
