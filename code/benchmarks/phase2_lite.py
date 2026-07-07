from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
SAM2_ROOT = WORKSPACE_ROOT / "sam2"

sys.path.insert(0, str(CODE_ROOT))
if SAM2_ROOT.exists():
    sys.path.insert(0, str(SAM2_ROOT))

DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "phase2_lite_40_cases.csv"
DEFAULT_PROMPTS = DEFAULT_OUTPUT_ROOT / "prompts" / "phase2_lite_prompts.jsonl"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PROMPT_PROTOCOLS = (
    "gt_box",
    "perturbed_box",
    "centroid_point",
    "grid_point",
    "auto_box",
    "auto_medical_box_v1",
    "auto_medical_point_v1",
)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    dataset: str
    split: str
    image_path: str
    mask_path: str
    width: int
    height: int
    target_area: int
    target_area_ratio: float
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    component_count_approx: int
    difficulty_bucket: str


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    else:
        arr = arr > 0
    return arr.astype(bool)


def resize_rgb_array(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(size, Image.BILINEAR), dtype=np.uint8)


def resize_mask_array(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def maybe_resize(image: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    height, width = image.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image, mask, 1.0, 1.0
    scale = max_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return (
        resize_rgb_array(image, (new_width, new_height)),
        resize_mask_array(mask, (new_width, new_height)),
        new_width / float(width),
        new_height / float(height),
    )


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, max(mask.shape[1] - 1, 0), max(mask.shape[0] - 1, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def clamp_box(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(x0, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    if x1 <= x0:
        x1 = min(width - 1.0, x0 + 1.0)
    if y1 <= y0:
        y1 = min(height - 1.0, y0 + 1.0)
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def perturb_box(box: list[float], width: int, height: int, seed_text: str) -> list[float]:
    rng = np.random.default_rng(stable_seed(seed_text))
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    expand = rng.uniform(0.08, 0.22)
    shift_x = rng.uniform(-0.08, 0.08) * bw
    shift_y = rng.uniform(-0.08, 0.08) * bh
    return clamp_box(
        [
            x0 - expand * bw + shift_x,
            y0 - expand * bh + shift_y,
            x1 + expand * bw + shift_x,
            y1 + expand * bh + shift_y,
        ],
        width,
        height,
    )


def centroid_point(mask: np.ndarray, width: int, height: int) -> dict[str, float | int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"x": width / 2.0, "y": height / 2.0, "label": 1}
    return {"x": round(float(xs.mean()), 3), "y": round(float(ys.mean()), 3), "label": 1}


def grid_points(mask: np.ndarray, width: int, height: int) -> list[dict[str, float | int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [centroid_point(mask, width, height)]
    points: list[dict[str, float | int]] = []
    for gy in (0, 1):
        for gx in (0, 1):
            cx = (gx + 0.5) * width / 2.0
            cy = (gy + 0.5) * height / 2.0
            in_cell = (
                (xs >= gx * width / 2.0)
                & (xs < (gx + 1) * width / 2.0)
                & (ys >= gy * height / 2.0)
                & (ys < (gy + 1) * height / 2.0)
            )
            cand_x = xs[in_cell] if np.any(in_cell) else xs
            cand_y = ys[in_cell] if np.any(in_cell) else ys
            idx = int(np.argmin((cand_x - cx) ** 2 + (cand_y - cy) ** 2))
            point = {"x": round(float(cand_x[idx]), 3), "y": round(float(cand_y[idx]), 3), "label": 1}
            if point not in points:
                points.append(point)
    return points or [centroid_point(mask, width, height)]


def approx_component_count(mask: np.ndarray, max_side: int = 384) -> int:
    height, width = mask.shape
    if max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        small = resize_mask_array(mask, (max(1, int(width * scale)), max(1, int(height * scale))))
    else:
        small = mask
    visited = np.zeros(small.shape, dtype=bool)
    count = 0
    h, w = small.shape
    for y, x in np.argwhere(small):
        y = int(y)
        x = int(x)
        if visited[y, x]:
            continue
        count += 1
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and small[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


def difficulty_bucket(area_ratio: float, component_count: int) -> str:
    if area_ratio < 0.002:
        return "tiny_target"
    if area_ratio < 0.02:
        return "small_target"
    if component_count >= 25:
        return "multi_instance"
    if area_ratio > 0.35:
        return "large_region"
    return "standard"


def auto_box_from_image(
    dataset: str,
    image: np.ndarray,
    gt_box: list[float],
) -> tuple[list[float], dict[str, Any]]:
    height, width = image.shape[:2]
    work = image
    sx = sy = 1.0
    if max(height, width) > 1024:
        scale = 1024 / float(max(height, width))
        work = resize_rgb_array(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        sx = work.shape[1] / float(width)
        sy = work.shape[0] / float(height)
    arr = work.astype(np.float32)
    if dataset.lower() == "monuseg":
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        threshold = float(np.percentile(gray, 32.0))
        candidate = gray <= threshold
        source = "dark_tissue_threshold"
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        score = r - 0.45 * g - 0.25 * b
        threshold = float(np.percentile(score, 99.4))
        candidate = (score >= threshold) & (r >= np.percentile(r, 70.0))
        source = "red_lesion_saliency"
    area_ratio = float(candidate.mean())
    if area_ratio < 1e-5 or area_ratio > 0.6:
        return perturb_box(gt_box, width, height, f"auto-fallback:{dataset}:{gt_box}"), {
            "source": source,
            "fallback": "perturbed_gt",
            "candidate_area_ratio": area_ratio,
        }
    x0, y0, x1, y1 = mask_bbox(candidate)
    box = [x0 / sx, y0 / sy, x1 / sx, y1 / sy]
    return clamp_box(box, width, height), {
        "source": source,
        "fallback": "",
        "candidate_area_ratio": area_ratio,
    }


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def robust_box_from_candidate(
    candidate: np.ndarray,
    width: int,
    height: int,
    sx: float,
    sy: float,
    pad_ratio: float,
) -> list[float] | None:
    ys, xs = np.where(candidate)
    if len(xs) == 0:
        return None
    x0, x1 = np.percentile(xs, [1.0, 99.0])
    y0, y1 = np.percentile(ys, [1.0, 99.0])
    bw = max(1.0, float(x1 - x0 + 1.0))
    bh = max(1.0, float(y1 - y0 + 1.0))
    box = [
        (x0 - pad_ratio * bw) / sx,
        (y0 - pad_ratio * bh) / sy,
        (x1 + pad_ratio * bw) / sx,
        (y1 + pad_ratio * bh) / sy,
    ]
    return clamp_box(box, width, height)


def points_from_candidate(
    candidate: np.ndarray,
    score: np.ndarray,
    width: int,
    height: int,
    sx: float,
    sy: float,
    positives: int = 4,
    include_negatives: bool = True,
) -> list[dict[str, float | int]]:
    points: list[dict[str, float | int]] = []
    h, w = candidate.shape
    for gy in (0, 1):
        for gx in (0, 1):
            if len(points) >= positives:
                break
            cell = (
                candidate
                & (np.arange(w)[None, :] >= gx * w / 2.0)
                & (np.arange(w)[None, :] < (gx + 1) * w / 2.0)
                & (np.arange(h)[:, None] >= gy * h / 2.0)
                & (np.arange(h)[:, None] < (gy + 1) * h / 2.0)
            )
            if not np.any(cell):
                continue
            masked = np.where(cell, score, -np.inf)
            y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
            points.append({"x": round(float(x) / sx, 3), "y": round(float(y) / sy, 3), "label": 1})
    if not points and np.any(candidate):
        masked = np.where(candidate, score, -np.inf)
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        points.append({"x": round(float(x) / sx, 3), "y": round(float(y) / sy, 3), "label": 1})
    if include_negatives:
        background = ~candidate
        if np.any(background):
            neg_score = np.where(background, 1.0 - score, -np.inf)
            for y, x in ((h * 0.18, w * 0.18), (h * 0.18, w * 0.82), (h * 0.82, w * 0.18), (h * 0.82, w * 0.82)):
                yy = int(np.clip(round(y), 0, h - 1))
                xx = int(np.clip(round(x), 0, w - 1))
                y0 = max(0, yy - h // 10)
                y1 = min(h, yy + h // 10 + 1)
                x0 = max(0, xx - w // 10)
                x1 = min(w, xx + w // 10 + 1)
                local = neg_score[y0:y1, x0:x1]
                if local.size and np.isfinite(local).any():
                    ly, lx = np.unravel_index(int(np.argmax(local)), local.shape)
                    points.append({"x": round(float(x0 + lx) / sx, 3), "y": round(float(y0 + ly) / sy, 3), "label": 0})
                    break
    if not points:
        points.append({"x": round(width / 2.0, 3), "y": round(height / 2.0, 3), "label": 1})
    return points


def auto_medical_prompt_v1(
    dataset: str,
    image: np.ndarray,
) -> tuple[list[float], list[dict[str, float | int]], dict[str, Any]]:
    height, width = image.shape[:2]
    work = image
    sx = sy = 1.0
    if max(height, width) > 1024:
        scale = 1024 / float(max(height, width))
        work = resize_rgb_array(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        sx = work.shape[1] / float(width)
        sy = work.shape[0] / float(height)

    arr = work.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    fundus_or_tissue = gray > max(8.0, float(np.percentile(gray, 5.0)))

    if dataset.lower() == "idrid":
        red_score = normalize01(r - 0.62 * g - 0.18 * b)
        bright_score = normalize01(0.45 * r + 0.55 * g - 0.65 * b + 0.18 * gray)
        score = np.maximum(red_score, bright_score)
        candidate = fundus_or_tissue & (
            (red_score >= np.percentile(red_score[fundus_or_tissue], 99.15))
            | (bright_score >= np.percentile(bright_score[fundus_or_tissue], 99.05))
        )
        source = "idrid_red_bright_lesion_candidates"
        pad = 0.18
        positives = 4
    else:
        purple = normalize01((b - r) + 0.35 * (b - g))
        dark = normalize01(255.0 - gray)
        score = np.maximum(dark, purple)
        candidate = fundus_or_tissue & (score >= np.percentile(score[fundus_or_tissue], 63.0))
        source = "monuseg_dark_purple_nuclei_candidates"
        pad = 0.08
        positives = 4

    area_ratio = float(candidate.mean())
    fallback = ""
    if area_ratio < 1e-5:
        threshold = float(np.percentile(score[fundus_or_tissue], 98.0)) if np.any(fundus_or_tissue) else float(np.percentile(score, 98.0))
        candidate = score >= threshold
        fallback = "relaxed_score_threshold"
        area_ratio = float(candidate.mean())
    if area_ratio > 0.45:
        threshold = float(np.percentile(score[candidate], 70.0)) if np.any(candidate) else float(np.percentile(score, 85.0))
        candidate = candidate & (score >= threshold)
        fallback = "tightened_large_candidate"
        area_ratio = float(candidate.mean())

    box = robust_box_from_candidate(candidate, width, height, sx, sy, pad)
    if box is None:
        margin_x = width * 0.12
        margin_y = height * 0.12
        box = clamp_box([margin_x, margin_y, width - margin_x, height - margin_y], width, height)
        fallback = "central_image_box"
    points = points_from_candidate(candidate, score, width, height, sx, sy, positives=positives, include_negatives=True)
    meta = {
        "source": source,
        "fallback": fallback,
        "candidate_area_ratio": area_ratio,
        "positive_points": sum(1 for p in points if int(p["label"]) == 1),
        "negative_points": sum(1 for p in points if int(p["label"]) == 0),
        "uses_gt": False,
    }
    return box, points, meta


def find_first_existing(base: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_idrid_cases(limit: int) -> list[tuple[str, str, Path, Path]]:
    cases: list[tuple[str, str, Path, Path]] = []
    for split_name, split_dir in (("train", "DR_Training_Set"), ("test", "DR_Testing_Set")):
        image_dir = CODE_ROOT / "datasets" / "IDRID" / split_dir / "Fundus Images"
        mask_dir = CODE_ROOT / "datasets" / "IDRID" / split_dir / "Combined Masks"
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda p: p.stem):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = find_first_existing(mask_dir, image_path.stem)
            if mask_path is None:
                continue
            cases.append((image_path.stem, split_name, image_path, mask_path))
            if len(cases) >= limit:
                return cases
    return cases


def collect_monuseg_cases(limit: int) -> list[tuple[str, str, Path, Path]]:
    roots = (
        ("kmms", CODE_ROOT / "datasets" / "MonuSeg" / "kmms"),
        ("train", CODE_ROOT / "datasets" / "MonuSeg" / "kmms_training" / "kmms_training"),
        ("test", CODE_ROOT / "datasets" / "MonuSeg" / "kmms_test" / "kmms_test"),
    )
    cases: list[tuple[str, str, Path, Path]] = []
    seen: set[str] = set()
    for split_name, root in roots:
        image_dir = root / "images"
        mask_dir = root / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda p: p.stem):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            case_key = image_path.stem
            if case_key in seen:
                continue
            mask_path = find_first_existing(mask_dir, image_path.stem)
            if mask_path is None:
                continue
            cases.append((case_key, split_name, image_path, mask_path))
            seen.add(case_key)
            if len(cases) >= limit:
                return cases
    return cases


def build_case_record(dataset: str, case_key: str, split: str, image_path: Path, mask_path: Path) -> CaseRecord:
    mask = load_mask(mask_path)
    height, width = mask.shape
    area = int(mask.sum())
    x0, y0, x1, y1 = mask_bbox(mask)
    components = approx_component_count(mask)
    return CaseRecord(
        case_id=f"{dataset}_{case_key}",
        dataset=dataset,
        split=split,
        image_path=rel(image_path),
        mask_path=rel(mask_path),
        width=width,
        height=height,
        target_area=area,
        target_area_ratio=area / max(float(width * height), 1.0),
        bbox_x0=x0,
        bbox_y0=y0,
        bbox_x1=x1,
        bbox_y1=y1,
        component_count_approx=components,
        difficulty_bucket=difficulty_bucket(area / max(float(width * height), 1.0), components),
    )


def write_manifest(output_path: Path, idrid_n: int, monuseg_n: int) -> list[CaseRecord]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[CaseRecord] = []
    for case_key, split, image_path, mask_path in collect_idrid_cases(idrid_n):
        selected.append(build_case_record("IDRID", case_key, split, image_path, mask_path))
    for case_key, split, image_path, mask_path in collect_monuseg_cases(monuseg_n):
        selected.append(build_case_record("MonuSeg", case_key, split, image_path, mask_path))
    fields = list(CaseRecord.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(row.__dict__)
    return selected


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_prompts(manifest_path: Path, prompts_path: Path) -> list[dict[str, Any]]:
    rows = read_manifest(manifest_path)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in rows:
        image = load_rgb(resolve_workspace_path(row["image_path"]))
        mask = load_mask(resolve_workspace_path(row["mask_path"]))
        width = int(row["width"])
        height = int(row["height"])
        gt_box = [
            float(row["bbox_x0"]),
            float(row["bbox_y0"]),
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
        ]
        prompt_defs = {
            "gt_box": {
                "prompt_type": "box",
                "box": clamp_box(gt_box, width, height),
                "points": [],
                "metadata": {"source": "ground_truth_mask"},
            },
            "perturbed_box": {
                "prompt_type": "box",
                "box": perturb_box(gt_box, width, height, row["case_id"]),
                "points": [],
                "metadata": {"source": "deterministic_gt_box_perturbation"},
            },
            "centroid_point": {
                "prompt_type": "point",
                "box": None,
                "points": [centroid_point(mask, width, height)],
                "metadata": {"source": "foreground_centroid"},
            },
            "grid_point": {
                "prompt_type": "point",
                "box": None,
                "points": grid_points(mask, width, height),
                "metadata": {"source": "foreground_2x2_grid"},
            },
        }
        auto_box, auto_meta = auto_box_from_image(row["dataset"], image, gt_box)
        auto_medical_box, auto_medical_points, auto_medical_meta = auto_medical_prompt_v1(row["dataset"], image)
        prompt_defs["auto_box"] = {
            "prompt_type": "box",
            "box": auto_box,
            "points": [],
            "metadata": auto_meta,
        }
        prompt_defs["auto_medical_box_v1"] = {
            "prompt_type": "box",
            "box": auto_medical_box,
            "points": [],
            "metadata": auto_medical_meta,
        }
        prompt_defs["auto_medical_point_v1"] = {
            "prompt_type": "point",
            "box": None,
            "points": auto_medical_points,
            "metadata": auto_medical_meta,
        }
        for protocol in PROMPT_PROTOCOLS:
            record = {
                "prompt_id": f"{row['case_id']}::{protocol}",
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "protocol": protocol,
                **prompt_defs[protocol],
            }
            records.append(record)
    with prompts_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def read_prompts(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            by_case.setdefault(record["case_id"], []).append(record)
    return by_case


def scale_prompt(prompt: dict[str, Any], sx: float, sy: float, width: int, height: int) -> dict[str, Any]:
    copied = dict(prompt)
    if copied.get("box") is not None:
        x0, y0, x1, y1 = copied["box"]
        copied["box"] = clamp_box([x0 * sx, y0 * sy, x1 * sx, y1 * sy], width, height)
    copied["points"] = [
        {"x": round(float(point["x"]) * sx, 3), "y": round(float(point["y"]) * sy, 3), "label": int(point["label"])}
        for point in copied.get("points", [])
    ]
    return copied


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    pred_area = int(pred.sum())
    target_area = int(target.sum())
    union = int(np.logical_or(pred, target).sum())
    return {
        "dice": (2.0 * inter) / max(float(pred_area + target_area), 1.0),
        "iou": inter / max(float(union), 1.0),
        "precision": inter / max(float(pred_area), 1.0),
        "recall": inter / max(float(target_area), 1.0),
        "pred_area": float(pred_area),
        "target_area": float(target_area),
    }


class SamRunner:
    def __init__(self, checkpoint: Path, device: str, model_type: str = "vit_b") -> None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        self.torch = torch
        self.device = pick_device(device)
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=self.device)
        model.eval()
        self.predictor = SamPredictor(model)

    def predict(self, image: np.ndarray, prompt: dict[str, Any]) -> tuple[np.ndarray, float]:
        point_coords, point_labels, box = prompt_arrays(prompt)
        multimask = not (prompt["prompt_type"] == "box" or (point_labels is not None and len(point_labels) > 1))
        with self.torch.inference_mode():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


class MedSamVitBRunner:
    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from segment_anything import sam_model_registry

        self.torch = torch
        self.device = pick_device(device)
        self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        self.model.to(device=self.device)
        self.model.eval()

    def predict(self, image: np.ndarray, prompt: dict[str, Any]) -> tuple[np.ndarray, float]:
        if prompt["prompt_type"] != "box":
            raise UnsupportedPrompt("MedSAM ViT-B is box-prompt only in this runner")
        _, _, box = prompt_arrays(prompt)
        if box is None:
            raise ValueError("MedSAM requires a box prompt")
        h, w = image.shape[:2]
        img_1024 = resize_rgb_array(image, (1024, 1024)).astype(np.float32)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
        img_tensor = self.torch.tensor(img_1024, dtype=self.torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
        box_1024 = box[None, :] / np.array([w, h, w, h], dtype=np.float32) * 1024.0
        with self.torch.no_grad():
            embedding = self.model.image_encoder(img_tensor)
            box_torch = self.torch.as_tensor(box_1024, dtype=self.torch.float, device=self.device)[:, None, :]
            sparse, dense = self.model.prompt_encoder(points=None, boxes=box_torch, masks=None)
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            pred = self.torch.sigmoid(low_res_logits)
            pred = self.torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return (pred.squeeze().detach().cpu().numpy() > 0.5), math.nan


class Sam2TinyRunner:
    def __init__(self, checkpoint: Path, config: str, device: str) -> None:
        import contextlib
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = pick_device(device)
        self.autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else contextlib.nullcontext()
        )
        model = build_sam2(config, str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def predict(self, image: np.ndarray, prompt: dict[str, Any]) -> tuple[np.ndarray, float]:
        point_coords, point_labels, box = prompt_arrays(prompt)
        multimask = not (prompt["prompt_type"] == "box" or (point_labels is not None and len(point_labels) > 1))
        with self.torch.inference_mode(), self.autocast_context:
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


class MedicalSamAdapterRunner:
    def __init__(self, sam_checkpoint: Path, adapter_checkpoint: Path, device: str, image_size: int = 1024) -> None:
        import sys
        import torch
        from types import SimpleNamespace

        adapter_root = WORKSPACE_ROOT / "Medical-SAM-Adapter"
        sys.path.insert(0, str(adapter_root))
        from models.sam import sam_model_registry

        self.torch = torch
        self.device = pick_device(device)
        self.image_size = image_size
        self.args = SimpleNamespace(
            image_size=image_size,
            out_size=image_size,
            mod="sam_adpt",
            multimask_output=3,
            encoder="vit_b",
            mid_dim=None,
            thd=False,
            chunk=None,
        )
        model = sam_model_registry["vit_b"](self.args, checkpoint=str(sam_checkpoint))
        adapter_state = torch.load(str(adapter_checkpoint), map_location="cpu")
        state_dict = adapter_state.get("state_dict", adapter_state)
        model.load_state_dict(state_dict, strict=False)
        model.to(device=self.device)
        model.eval()
        self.model = model

    def _prepare_image(self, image: np.ndarray) -> tuple[Any, tuple[int, int], tuple[int, int], float, float]:
        h, w = image.shape[:2]
        scale = min(float(self.image_size) / max(h, w), 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = resize_rgb_array(image, (new_w, new_h)) if scale != 1.0 else image
        tensor = self.torch.as_tensor(resized, dtype=self.torch.float32, device=self.device).permute(2, 0, 1)
        return tensor, (h, w), (new_h, new_w), new_w / float(w), new_h / float(h)

    def predict(self, image: np.ndarray, prompt: dict[str, Any]) -> tuple[np.ndarray, float]:
        image_tensor, original_size, input_size, sx, sy = self._prepare_image(image)
        scaled_prompt = scale_prompt(prompt, sx, sy, input_size[1], input_size[0])
        point_coords, point_labels, box = prompt_arrays(scaled_prompt)
        with self.torch.inference_mode():
            input_image = self.model.preprocess(image_tensor).unsqueeze(0)
            image_embedding = self.model.image_encoder(input_image)
            points = None
            if point_coords is not None and point_labels is not None:
                points = (
                    self.torch.as_tensor(point_coords, dtype=self.torch.float32, device=self.device)[None, :, :],
                    self.torch.as_tensor(point_labels, dtype=self.torch.int64, device=self.device)[None, :],
                )
            box_torch = None
            if box is not None:
                box_torch = self.torch.as_tensor(box, dtype=self.torch.float32, device=self.device)[None, :]
            sparse, dense = self.model.prompt_encoder(points=points, boxes=box_torch, masks=None)
            low_res_masks, scores = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            masks = self.model.postprocess_masks(low_res_masks, input_size=input_size, original_size=original_size)
        best = int(self.torch.argmax(scores[0]).detach().cpu().item())
        mask = (masks[0, best].detach().cpu().numpy() > float(self.model.mask_threshold))
        return mask.astype(bool), float(scores[0, best].detach().cpu().item())


class UnsupportedPrompt(RuntimeError):
    pass


def pick_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def prompt_arrays(prompt: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    point_coords = None
    point_labels = None
    box = None
    if prompt["prompt_type"] == "point":
        points = prompt.get("points") or []
        if not points:
            raise ValueError("point prompt has no points")
        point_coords = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
        point_labels = np.array([int(p.get("label", 1)) for p in points], dtype=np.int32)
    elif prompt["prompt_type"] == "box":
        if prompt.get("box") is None:
            raise ValueError("box prompt has no box")
        box = np.array(prompt["box"], dtype=np.float32)
    else:
        raise UnsupportedPrompt(f"Unsupported prompt type: {prompt['prompt_type']}")
    return point_coords, point_labels, box


def make_model_runner(name: str, args: argparse.Namespace):
    if name == "sam_vit_b":
        return SamRunner(Path(args.sam_checkpoint), args.device, "vit_b")
    if name == "sam_vit_h":
        return SamRunner(Path(args.sam_h_checkpoint), args.device, "vit_h")
    if name == "medsam_vit_b":
        return MedSamVitBRunner(Path(args.medsam_checkpoint), args.device)
    if name == "lite_medsam":
        return MedSamVitBRunner(Path(args.lite_medsam_checkpoint), args.device)
    if name == "sam2_tiny":
        return Sam2TinyRunner(Path(args.sam2_checkpoint), args.sam2_config, args.device)
    if name == "medsa_opticdisc":
        return MedicalSamAdapterRunner(Path(args.sam_checkpoint), Path(args.medsa_opticdisc_checkpoint), args.device, int(args.medsa_image_size))
    if name == "medsa_microscope":
        return MedicalSamAdapterRunner(Path(args.sam_checkpoint), Path(args.medsa_microscope_checkpoint), args.device, int(args.medsa_image_size))
    raise ValueError(f"Unknown model: {name}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = Path(args.manifest)
    prompts_path = Path(args.prompts)
    output_root = Path(args.output_root)
    pred_root = output_root / "predictions"
    metrics_path = output_root / "metrics" / "phase2_lite_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    cases = read_manifest(manifest_path)
    prompts_by_case = read_prompts(prompts_path)
    if args.limit_cases:
        cases = cases[: int(args.limit_cases)]
    requested_protocols = set(args.protocols.split(",")) if args.protocols else set(PROMPT_PROTOCOLS)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    results: list[dict[str, Any]] = []
    runners: dict[str, Any] = {}
    for model_name in model_names:
        try:
            runners[model_name] = make_model_runner(model_name, args)
        except Exception as exc:
            for case in cases:
                for prompt in prompts_by_case.get(case["case_id"], []):
                    if prompt["protocol"] not in requested_protocols:
                        continue
                    results.append(result_row(case, prompt, model_name, "model_load_failed", str(exc)))
            continue

    for case_index, case in enumerate(cases, start=1):
        image = load_rgb(resolve_workspace_path(case["image_path"]))
        target = load_mask(resolve_workspace_path(case["mask_path"]))
        image_eval, target_eval, sx, sy = maybe_resize(image, target, int(args.max_side))
        print(f"[{case_index}/{len(cases)}] {case['case_id']} eval={image_eval.shape[1]}x{image_eval.shape[0]}", flush=True)
        for prompt in prompts_by_case.get(case["case_id"], []):
            if prompt["protocol"] not in requested_protocols:
                continue
            scaled_prompt = scale_prompt(prompt, sx, sy, image_eval.shape[1], image_eval.shape[0])
            for model_name in model_names:
                if model_name not in runners:
                    continue
                start = time.perf_counter()
                status = "ok"
                error = ""
                score = math.nan
                metrics = {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan, "pred_area": math.nan, "target_area": float(target_eval.sum())}
                mask_path = ""
                try:
                    pred_mask, score = runners[model_name].predict(image_eval, scaled_prompt)
                    metrics = compute_metrics(pred_mask, target_eval)
                    mask_file = pred_root / model_name / case["case_id"] / f"{prompt['protocol']}.png"
                    save_mask(mask_file, pred_mask)
                    mask_path = rel(mask_file)
                except UnsupportedPrompt as exc:
                    status = "unsupported"
                    error = str(exc)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)[-800:]
                elapsed = time.perf_counter() - start
                row = result_row(case, prompt, model_name, status, error)
                row.update(
                    {
                        "score": score,
                        "runtime_sec": round(elapsed, 4),
                        "eval_width": image_eval.shape[1],
                        "eval_height": image_eval.shape[0],
                        "scale_x": round(sx, 6),
                        "scale_y": round(sy, 6),
                        "mask_path": mask_path,
                        **metrics,
                    }
                )
                results.append(row)
                print(
                    f"  {model_name:12s} {prompt['protocol']:14s} {status:11s} dice={row['dice'] if status == 'ok' else 'NA'}",
                    flush=True,
                )

    if results:
        fields = list(results[0].keys())
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    write_summary(output_root, results, manifest_path, prompts_path)
    return results


def result_row(case: dict[str, Any], prompt: dict[str, Any], model: str, status: str, error: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "split": case["split"],
        "difficulty_bucket": case["difficulty_bucket"],
        "prompt_id": prompt["prompt_id"],
        "protocol": prompt["protocol"],
        "prompt_type": prompt["prompt_type"],
        "model": model,
        "status": status,
        "error": error,
        "score": math.nan,
        "runtime_sec": math.nan,
        "eval_width": "",
        "eval_height": "",
        "scale_x": "",
        "scale_y": "",
        "dice": math.nan,
        "iou": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "pred_area": math.nan,
        "target_area": math.nan,
        "mask_path": "",
    }


def write_summary(output_root: Path, results: list[dict[str, Any]], manifest_path: Path, prompts_path: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in results if r["status"] == "ok"]
    by_model: dict[str, list[dict[str, Any]]] = {}
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
        by_protocol.setdefault(str(row["protocol"]), []).append(row)
    lines = [
        "# Phase2-lite Run Summary",
        "",
        f"- Manifest: `{rel(manifest_path)}`",
        f"- Prompts: `{rel(prompts_path)}`",
        f"- Total rows: {len(results)}",
        f"- OK rows: {len(ok_rows)}",
        "",
        "## Mean Dice by model",
        "",
    ]
    for model, rows in sorted(by_model.items()):
        mean_dice = float(np.nanmean([float(r["dice"]) for r in rows])) if rows else math.nan
        lines.append(f"- {model}: {mean_dice:.4f} over {len(rows)} supported runs")
    lines.extend(["", "## Mean Dice by dataset", ""])
    for dataset, rows in sorted(by_dataset.items()):
        mean_dice = float(np.nanmean([float(r["dice"]) for r in rows])) if rows else math.nan
        lines.append(f"- {dataset}: {mean_dice:.4f} over {len(rows)} supported runs")
    lines.extend(["", "## Mean Dice by prompt protocol", ""])
    for protocol, rows in sorted(by_protocol.items()):
        mean_dice = float(np.nanmean([float(r["dice"]) for r in rows])) if rows else math.nan
        lines.append(f"- {protocol}: {mean_dice:.4f} over {len(rows)} supported runs")
    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    lines.extend(["", "## Status counts", ""])
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    prompts = Path(args.prompts)
    cases = write_manifest(manifest, args.idrid_cases, args.monuseg_cases)
    prompt_records = write_prompts(manifest, prompts)
    print(f"Wrote {len(cases)} cases: {manifest}", flush=True)
    print(f"Wrote {len(prompt_records)} prompts: {prompts}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2-lite manifest, prompt generation, and SAM-family benchmark runner.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Create the 40-case manifest and prompt jsonl.")
    build.add_argument("--idrid-cases", type=int, default=20)
    build.add_argument("--monuseg-cases", type=int, default=20)

    run = sub.add_parser("run", help="Run model predictions and write metrics.")
    run.add_argument("--models", default="sam_vit_b,medsam_vit_b,sam2_tiny")
    run.add_argument("--protocols", default=",".join(PROMPT_PROTOCOLS))
    run.add_argument("--limit-cases", type=int, default=0)
    run.add_argument("--max-side", type=int, default=768)
    run.add_argument("--device", default="auto")
    run.add_argument("--sam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"))
    run.add_argument("--sam-h-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth"))
    run.add_argument("--medsam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_vit_b.pth"))
    run.add_argument("--lite-medsam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_download" / "lite_medsam.pth"))
    run.add_argument("--sam2-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    run.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    run.add_argument("--medsa-opticdisc-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medical_adapter_zoo" / "OpticDisc_Fundus_SAM_1024.pth"))
    run.add_argument("--medsa-microscope-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medical_adapter_zoo" / "White Blood Cell_MicroScope_sam_1024.pth"))
    run.add_argument("--medsa-image-size", type=int, default=1024)

    all_cmd = sub.add_parser("all", help="Build manifest/prompts, then run predictions.")
    all_cmd.add_argument("--idrid-cases", type=int, default=20)
    all_cmd.add_argument("--monuseg-cases", type=int, default=20)
    all_cmd.add_argument("--models", default="sam_vit_b,medsam_vit_b,sam2_tiny")
    all_cmd.add_argument("--protocols", default=",".join(PROMPT_PROTOCOLS))
    all_cmd.add_argument("--limit-cases", type=int, default=0)
    all_cmd.add_argument("--max-side", type=int, default=768)
    all_cmd.add_argument("--device", default="auto")
    all_cmd.add_argument("--sam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"))
    all_cmd.add_argument("--sam-h-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth"))
    all_cmd.add_argument("--medsam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_vit_b.pth"))
    all_cmd.add_argument("--lite-medsam-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_download" / "lite_medsam.pth"))
    all_cmd.add_argument("--sam2-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    all_cmd.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    all_cmd.add_argument("--medsa-opticdisc-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medical_adapter_zoo" / "OpticDisc_Fundus_SAM_1024.pth"))
    all_cmd.add_argument("--medsa-microscope-checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "medical_adapter_zoo" / "White Blood Cell_MicroScope_sam_1024.pth"))
    all_cmd.add_argument("--medsa-image-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()
    if args.command == "build":
        build_all(args)
    elif args.command == "run":
        run_benchmark(args)
    elif args.command == "all":
        build_all(args)
        run_benchmark(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
