"""Fair SAM3 baselines for the labelled 220-image OnePrompt-All benchmark.

``text`` is an oracle-language baseline: SAM3 sees only the image and the
ground-truth class name (e.g. ``banana``). ``point`` is the native one-click,
one-instance baseline: it sees only one positive point and may return exactly
one mask. Ground truth is used solely to sample reproducible clicks and score
predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SAM3_ROOT = WORKSPACE / "external" / "sam3"
for path in (ROOT, SAM3_ROOT / ".deps_runtime", SAM3_ROOT):
    sys.path.insert(0, str(path))

from demos.medical_sam_click_app import PointPrompt, Sam3TextSegmenter  # noqa: E402
from research.evaluate_sam3_onepoint import metrics, nms  # noqa: E402
from research.evaluate_sam3_multi_click import sampled_point  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("text", "point"), required=True)
    parser.add_argument("--threshold", type=float, default=.45)
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--min-area-ratio", type=float, default=.002)
    parser.add_argument("--prompts-per-image", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def read_truth(dataset: Path, row: dict[str, str]) -> list[np.ndarray]:
    return [np.asarray(Image.open(path).convert("L"), dtype=bool)
            for path in sorted((dataset / row["instance_mask_dir"]).glob("*.png"))]


def text_prompt(class_name: str) -> str:
    """Turn COCO/LVIS catalogue labels into a direct natural-language prompt."""
    return re.sub(r"\\s+", " ", class_name.replace("_", " ").replace("(", " ").replace(")", " ")).strip()


def select_single_click_mask(masks: np.ndarray, scores: np.ndarray, x: int, y: int) -> list[np.ndarray]:
    """Native point-prompt baseline: one output mask, never all candidates."""
    if not len(masks):
        return []
    contains = np.asarray([mask[y, x] for mask in masks], dtype=bool)
    candidates = np.flatnonzero(contains)
    if not len(candidates):
        candidates = np.arange(len(masks))
    best = int(candidates[np.argmax(scores[candidates])])
    return [masks[best].astype(bool)]


def score(prediction: list[np.ndarray], truth: list[np.ndarray]) -> tuple[float, float, float, float, int]:
    return metrics(prediction, truth)


def main() -> None:
    cfg = parse_args(); cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader((cfg.dataset / "manifest.csv").open(encoding="utf-8")))
    sam3 = Sam3TextSegmenter(cfg.checkpoint, device_name, cfg.threshold)
    rng = np.random.default_rng(cfg.seed)
    trials: list[dict[str, object]] = []
    for image_index, row in enumerate(rows, 1):
        image = np.asarray(Image.open(cfg.dataset / row["image"]).convert("RGB"))
        truth = read_truth(cfg.dataset, row)
        min_area = image.shape[0] * image.shape[1] * cfg.min_area_ratio
        repetitions = 1 if cfg.mode == "text" else cfg.prompts_per_image
        selected = rng.integers(0, len(truth), size=repetitions) if cfg.mode == "point" else [-1]
        text_masks = text_scores = None
        if cfg.mode == "text":
            text_masks, _, text_scores = sam3.predict(image, text_prompt(row["class_name"]))
            keep = [i for i, mask in enumerate(text_masks) if float(mask.sum()) >= min_area]
            text_masks, text_scores = text_masks[keep], text_scores[keep]
        for prompt_index, gt_index in enumerate(selected, 1):
            started = time.perf_counter(); x = y = None
            if cfg.mode == "text":
                prediction = nms(text_masks, text_scores, cfg.nms_iou)
            else:
                x, y = sampled_point(truth[int(gt_index)], rng)
                masks, scores = sam3.predict_interactive(image, "click", [PointPrompt(x=x, y=y, label=1)], None, "manual")
                prediction = select_single_click_mask(masks, scores, x, y)
            precision, recall, f1, union_iou, matched = score(prediction, truth)
            trials.append(dict(image=row["image"], class_name=row["class_name"], text_prompt=text_prompt(row["class_name"]) if cfg.mode == "text" else None, prompt_index=prompt_index,
                               prompt_gt_instance=None if cfg.mode == "text" else int(gt_index) + 1,
                               point_x=x, point_y=y, gt_instances=len(truth), predicted_instances=len(prediction),
                               matched_instances=matched, precision=precision, recall=recall, f1=f1,
                               union_iou=union_iou, elapsed_sec=time.perf_counter() - started))
            print(f"[{image_index}/{len(rows)} p{prompt_index}] {cfg.mode} F1={f1:.3f} K={len(prediction)}/{len(truth)}", flush=True)
            if device_name.startswith("cuda"):
                torch.cuda.empty_cache()
    with (cfg.out_dir / "per_prompt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    per_image = []
    for image_name in dict.fromkeys(str(item["image"]) for item in trials):
        group = [item for item in trials if item["image"] == image_name]
        per_image.append(dict(image=image_name, class_name=group[0]["class_name"], prompts=len(group),
                              mean_f1=float(np.mean([item["f1"] for item in group])),
                              std_f1=float(np.std([item["f1"] for item in group]))))
    summary = dict(mode=cfg.mode, images=len(per_image), prompts=len(trials), prompts_per_image=repetitions,
                   mean_f1=float(np.mean([item["f1"] for item in trials])),
                   std_f1=float(np.std([item["f1"] for item in trials])),
                   mean_precision=float(np.mean([item["precision"] for item in trials])),
                   mean_recall=float(np.mean([item["recall"] for item in trials])),
                   mean_union_iou=float(np.mean([item["union_iou"] for item in trials])),
                   mean_per_image_std_f1=float(np.mean([item["std_f1"] for item in per_image])), seed=cfg.seed)
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
