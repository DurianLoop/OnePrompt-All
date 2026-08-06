from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.phase2_lite import (  # noqa: E402
    MedSamVitBRunner,
    SamRunner,
    UnsupportedPrompt,
    auto_medical_prompt_v1,
    clamp_box,
    compute_metrics,
    load_mask,
    load_rgb,
    mask_bbox,
)


DEFAULT_MANIFEST = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_official_screening" / "official_monuseg_screening_manifest.csv"
DEFAULT_OUT = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_official_screening" / "prompt_baselines_test14"


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prompt_defs(case_id: str, image: np.ndarray, gt: np.ndarray, random_seeds: int) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    gt_box = list(mask_bbox(gt))
    margin_x = width * 0.1
    margin_y = height * 0.1
    auto_box, auto_points, auto_meta = auto_medical_prompt_v1("monuseg", image)

    prompts: list[dict[str, Any]] = [
        {
            "protocol": "center_point",
            "prompt_type": "point",
            "points": [{"x": round(width / 2.0, 3), "y": round(height / 2.0, 3), "label": 1}],
            "box": None,
            "uses_gt": False,
        },
        {
            "protocol": "grid_4_points",
            "prompt_type": "point",
            "points": [
                {"x": round(width * 0.25, 3), "y": round(height * 0.25, 3), "label": 1},
                {"x": round(width * 0.75, 3), "y": round(height * 0.25, 3), "label": 1},
                {"x": round(width * 0.25, 3), "y": round(height * 0.75, 3), "label": 1},
                {"x": round(width * 0.75, 3), "y": round(height * 0.75, 3), "label": 1},
            ],
            "box": None,
            "uses_gt": False,
        },
        {
            "protocol": "full_image_box",
            "prompt_type": "box",
            "points": [],
            "box": [0.0, 0.0, float(width - 1), float(height - 1)],
            "uses_gt": False,
        },
        {
            "protocol": "central_box",
            "prompt_type": "box",
            "points": [],
            "box": clamp_box([margin_x, margin_y, width - margin_x, height - margin_y], width, height),
            "uses_gt": False,
        },
        {
            "protocol": "auto_medical_box_v1",
            "prompt_type": "box",
            "points": [],
            "box": auto_box,
            "uses_gt": False,
            "metadata": json.dumps(auto_meta, ensure_ascii=False),
        },
        {
            "protocol": "auto_medical_point_v1",
            "prompt_type": "point",
            "points": auto_points,
            "box": None,
            "uses_gt": False,
            "metadata": json.dumps(auto_meta, ensure_ascii=False),
        },
        {
            "protocol": "gt_box_oracle",
            "prompt_type": "box",
            "points": [],
            "box": gt_box,
            "uses_gt": True,
        },
    ]
    for seed in range(random_seeds):
        rng = np.random.default_rng(abs(hash(f"{case_id}:{seed}")) % (2**32))
        prompts.append(
            {
                "protocol": f"random_point_seed{seed}",
                "prompt_type": "point",
                "points": [{"x": round(float(rng.uniform(0, width - 1)), 3), "y": round(float(rng.uniform(0, height - 1)), 3), "label": 1}],
                "box": None,
                "uses_gt": False,
            }
        )
    return prompts


def make_runner(model_name: str, args: argparse.Namespace):
    device = args.device
    if model_name == "sam_vit_b":
        return SamRunner(resolve_path(args.sam_b_checkpoint), device, "vit_b")
    if model_name == "sam_vit_l":
        return SamRunner(resolve_path(args.sam_l_checkpoint), device, "vit_l")
    if model_name == "sam_vit_h":
        return SamRunner(resolve_path(args.sam_h_checkpoint), device, "vit_h")
    if model_name == "medsam_vit_b":
        import torch

        original_load = torch.load

        def cpu_load(*load_args: Any, **load_kwargs: Any):
            load_kwargs.setdefault("map_location", "cpu")
            return original_load(*load_args, **load_kwargs)

        torch.load = cpu_load
        try:
            return MedSamVitBRunner(resolve_path(args.medsam_checkpoint), device)
        finally:
            torch.load = original_load
    raise ValueError(f"Unknown model: {model_name}")


def draw_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, prompt: dict[str, Any], path: Path, title: str) -> None:
    base = Image.fromarray(image).convert("RGBA")
    gt_layer = Image.new("RGBA", base.size, (0, 180, 255, 0))
    pred_layer = Image.new("RGBA", base.size, (255, 40, 40, 0))
    gt_layer.putalpha(Image.fromarray((gt.astype(np.uint8) * 70), mode="L"))
    pred_layer.putalpha(Image.fromarray((pred.astype(np.uint8) * 105), mode="L"))
    merged = Image.alpha_composite(Image.alpha_composite(base, gt_layer), pred_layer).convert("RGB")
    draw = ImageDraw.Draw(merged)
    if prompt.get("box") is not None:
        draw.rectangle(tuple(prompt["box"]), outline=(255, 230, 0), width=3)
    for point in prompt.get("points", []):
        x = float(point["x"])
        y = float(point["y"])
        color = (0, 255, 80) if int(point.get("label", 1)) == 1 else (255, 80, 40)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=1)
    draw.rectangle((4, merged.height - 38, min(merged.width - 4, 980), merged.height - 4), fill=(0, 0, 0))
    draw.text((12, merged.height - 28), title, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(path)


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, items in sorted(buckets.items(), key=lambda item: item[0]):
        row = {key: value for key, value in zip(keys, key_values)}
        for metric in ("dice", "iou", "precision", "recall"):
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float32)
            row[f"mean_{metric}"] = round(float(values.mean()), 6)
            row[f"std_{metric}"] = round(float(values.std()), 6)
        row["case_count"] = len({item["case_id"] for item in items})
        row["row_count"] = len(items)
        out.append(row)
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(resolve_path(args.manifest))
    if args.case_limit > 0:
        rows = rows[: args.case_limit]
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    runners = {name: make_runner(name, args) for name in model_names}
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        case_id = row["query_case_id"]
        image = load_rgb(resolve_path(row["query_image_path"]))
        gt = load_mask(resolve_path(row["query_mask_path"]))
        prompts = prompt_defs(case_id, image, gt, args.random_seeds)
        for model_name, runner in runners.items():
            for prompt in prompts:
                if model_name == "medsam_vit_b" and prompt["prompt_type"] != "box":
                    status = "unsupported_prompt"
                    error = "MedSAM box-only baseline"
                    metrics = {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan, "pred_area": math.nan, "target_area": float(gt.sum())}
                    mask_rel = ""
                    overlay_rel = ""
                else:
                    try:
                        pred, score = runner.predict(image, prompt)
                        metrics = compute_metrics(pred, gt)
                        status = "ok"
                        error = ""
                        case_dir = out_dir / "cases" / case_id / model_name / prompt["protocol"]
                        mask_path = case_dir / "mask.png"
                        overlay_path = case_dir / "overlay.png"
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(pred.astype(np.uint8) * 255, mode="L").save(mask_path)
                        draw_overlay(image, gt, pred, prompt, overlay_path, f"{model_name} {prompt['protocol']} dice={metrics['dice']:.3f}")
                        mask_rel = rel(mask_path)
                        overlay_rel = rel(overlay_path)
                    except UnsupportedPrompt as exc:
                        status = "unsupported_prompt"
                        error = str(exc)
                        score = math.nan
                        metrics = {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan, "pred_area": math.nan, "target_area": float(gt.sum())}
                        mask_rel = ""
                        overlay_rel = ""
                    except Exception as exc:
                        status = "error"
                        error = repr(exc)
                        score = math.nan
                        metrics = {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan, "pred_area": math.nan, "target_area": float(gt.sum())}
                        mask_rel = ""
                        overlay_rel = ""
                results.append(
                    {
                        "case_id": case_id,
                        "model": model_name,
                        "protocol": prompt["protocol"],
                        "prompt_type": prompt["prompt_type"],
                        "uses_gt": int(bool(prompt.get("uses_gt", False))),
                        "status": status,
                        "score": round(float(score), 6) if status == "ok" and not math.isnan(float(score)) else "",
                        "dice": round(float(metrics["dice"]), 6) if not math.isnan(float(metrics["dice"])) else "",
                        "iou": round(float(metrics["iou"]), 6) if not math.isnan(float(metrics["iou"])) else "",
                        "precision": round(float(metrics["precision"]), 6) if not math.isnan(float(metrics["precision"])) else "",
                        "recall": round(float(metrics["recall"]), 6) if not math.isnan(float(metrics["recall"])) else "",
                        "pred_area": round(float(metrics["pred_area"]), 3) if not math.isnan(float(metrics["pred_area"])) else "",
                        "target_area": round(float(metrics["target_area"]), 3),
                        "mask_path": mask_rel,
                        "overlay_path": overlay_rel,
                        "error": error,
                    }
                )
        print(f"[case] {idx}/{len(rows)} {case_id}", flush=True)

    save_csv(out_dir / "prompt_baseline_results.csv", results)
    ok_rows = [row for row in results if row["status"] == "ok"]
    save_csv(out_dir / "summary_by_model_protocol.csv", summarize(ok_rows, ["model", "protocol", "prompt_type", "uses_gt"]))
    save_csv(out_dir / "summary_by_protocol.csv", summarize(ok_rows, ["protocol", "prompt_type", "uses_gt"]))
    print(f"[done] rows={len(results)} ok={len(ok_rows)} out={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ordinary prompt baselines on official MoNuSeg screening queries.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--case-limit", type=int, default=14)
    parser.add_argument("--models", default="sam_vit_b,medsam_vit_b")
    parser.add_argument("--random-seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sam-b-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-l-checkpoint", default="assets/checkpoints/sam_vit_l_0b3195.pth")
    parser.add_argument("--sam-h-checkpoint", default="assets/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--medsam-checkpoint", default="assets/checkpoints/medsam_vit_b.pth")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
