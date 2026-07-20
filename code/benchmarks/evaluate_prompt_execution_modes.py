from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib_prompt_execution"))

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2

    if hasattr(cv2, "connectedComponents"):
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    else:
        count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), int(count - 1)


def keep_prompt_component(mask: np.ndarray, x: float, y: float) -> np.ndarray:
    labels, count = connected_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    px = min(mask.shape[1] - 1, max(0, int(round(x))))
    py = min(mask.shape[0] - 1, max(0, int(round(y))))
    label = int(labels[py, px])
    if label == 0:
        ys, xs = np.where(labels > 0)
        nearest = int(np.argmin(np.square(xs - px) + np.square(ys - py)))
        label = int(labels[ys[nearest], xs[nearest]])
    return labels == label


def union_prompt_components(masks: list[np.ndarray], points: list[tuple[float, float]]) -> np.ndarray:
    if len(masks) != len(points):
        raise ValueError("Each single-point mask must have one corresponding prompt point.")
    if not masks:
        raise ValueError("At least one mask is required.")
    union = np.zeros_like(masks[0], dtype=bool)
    for mask, (x, y) in zip(masks, points):
        union |= keep_prompt_component(mask, x, y)
    return union


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.NEAREST)
        array = np.asarray(image)
    return np.any(array > 0, axis=2) if array.ndim == 3 else array > 0


def load_points(path: Path) -> list[tuple[float, float]]:
    return [(float(row["x"]), float(row["y"])) for row in read_csv(path)]


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    denominator = float(pred.sum() + gt.sum())
    return (
        2.0 * intersection / denominator if denominator else 1.0,
        intersection / union if union else 1.0,
    )


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    dice, iou = dice_iou(pred, gt)
    _, components = connected_components(pred)
    return {
        "dice": dice,
        "iou": iou,
        "pred_area_pct": 100.0 * float(pred.mean()),
        "pred_components": components,
    }


def select_best_multimasks(
    masks: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if masks.ndim != 4 or scores.ndim != 2 or masks.shape[:2] != scores.shape:
        raise ValueError("Expected masks [B,C,H,W] and scores [B,C].")
    best = scores.argmax(dim=1)
    indices = torch.arange(masks.shape[0], device=masks.device)
    return masks[indices, best], scores[indices, best]


@torch.inference_mode()
def predict_single_point_masks(
    predictor: SamPredictor,
    points: list[tuple[float, float]],
    original_size: tuple[int, int],
    chunk_size: int,
) -> tuple[list[np.ndarray], list[float]]:
    masks_out: list[np.ndarray] = []
    scores_out: list[float] = []
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        coordinates = torch.as_tensor(chunk, dtype=torch.float32, device=predictor.device).unsqueeze(1)
        transformed = predictor.transform.apply_coords_torch(coordinates, original_size)
        labels = torch.ones((len(chunk), 1), dtype=torch.int64, device=predictor.device)
        masks, scores, _ = predictor.predict_torch(
            point_coords=transformed,
            point_labels=labels,
            boxes=None,
            mask_input=None,
            multimask_output=True,
            return_logits=False,
        )
        selected_masks, selected_scores = select_best_multimasks(masks, scores)
        masks_out.extend(selected_masks.detach().cpu().numpy().astype(bool))
        scores_out.extend(float(value) for value in selected_scores.detach().cpu().numpy())
    return masks_out, scores_out


def draw_points(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, (x, y) in enumerate(points, start=1):
        radius = 10 if index <= 10 else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 225, 0), outline=(90, 0, 0), width=3)
    return canvas


def save_figure(
    query: Image.Image,
    gt: np.ndarray,
    points: list[tuple[float, float]],
    masks: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float | int]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    axes[0, 0].imshow(query)
    axes[0, 0].set_title("Query")
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("GT nuclei")
    axes[0, 2].imshow(draw_points(query, points))
    axes[0, 2].set_title(f"Matched points (K={len(points)})")
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0,
        0.95,
        "Joint: all points mean one object\n"
        "Single union: decode each point separately\n"
        "Component union: keep only the local\ncomponent containing each prompt",
        va="top",
        fontsize=11,
        linespacing=1.5,
    )
    for axis, mode in zip(axes[1, :3], ("joint", "single_union", "component_union")):
        axis.imshow(masks[mode], cmap="gray")
        axis.set_title(f"{mode}\nDice={float(metrics[mode]['dice']):.4f}, area={float(metrics[mode]['pred_area_pct']):.1f}%")
    difference = np.zeros((*gt.shape, 3), dtype=np.uint8)
    difference[np.logical_and(gt, masks["component_union"])] = (35, 190, 90)
    difference[np.logical_and(~gt, masks["component_union"])] = (235, 65, 55)
    difference[np.logical_and(gt, ~masks["component_union"])] = (55, 125, 235)
    axes[1, 3].imshow(difference)
    axes[1, 3].set_title("Component-union error\ngreen TP / red FP / blue FN")
    for axis in axes.reshape(-1):
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare joint versus per-point SAM execution for dense all-instance prompts.")
    parser.add_argument(
        "--candidates",
        default="runs/research_direction/prompt_policy_oracle/test_grid_dinov2l_samvitb_fixedr3/candidates.csv",
    )
    parser.add_argument(
        "--episodes",
        default="runs/research_direction/prompt_policy_splits/monuseg_strict_official/episodes_test.csv",
    )
    parser.add_argument("--action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--case-limit", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        default="runs/research_direction/prompt_execution/k40_lam075_smoke",
    )
    args = parser.parse_args()

    candidates = [row for row in read_csv(resolve(args.candidates)) if row["method"] == args.action]
    candidates = sorted(candidates, key=lambda row: row["query_case_id"])
    if args.case_limit > 0:
        candidates = candidates[: args.case_limit]
    episodes = {row["query_case_id"]: row for row in read_csv(resolve(args.episodes))}
    device = torch.device(args.device)
    sam = sam_model_registry["vit_b"](checkpoint=str(resolve(args.sam_checkpoint)))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    out_dir = resolve(args.out_dir)
    result_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        query_id = candidate["query_case_id"]
        episode = episodes[query_id]
        query = Image.open(resolve(episode["query_image_path"])).convert("RGB")
        gt = load_mask(resolve(episode["query_mask_path"]), query.size)
        points = load_points(resolve(candidate["points_path"]))
        predictor.set_image(np.asarray(query))
        start = time.perf_counter()
        single_masks, single_scores = predict_single_point_masks(
            predictor,
            points,
            original_size=(query.height, query.width),
            chunk_size=args.chunk_size,
        )
        elapsed = time.perf_counter() - start
        modes = {
            "joint": load_mask(resolve(candidate["mask_path"]), query.size),
            "single_union": np.logical_or.reduce(single_masks),
            "component_union": union_prompt_components(single_masks, points),
        }
        case_metrics = {mode: mask_metrics(mask, gt) for mode, mask in modes.items()}
        case_dir = out_dir / "cases" / query_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for mode, mask in modes.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(case_dir / f"{mode}.png")
            result_rows.append({
                "query_case_id": query_id,
                "action": args.action,
                "mode": mode,
                "prompt_count": len(points),
                "mean_single_sam_score": float(np.mean(single_scores)),
                "single_decode_seconds": elapsed,
                **case_metrics[mode],
                "mask_path": str((case_dir / f"{mode}.png").relative_to(ROOT)).replace("\\", "/"),
            })
        save_figure(query, gt, points, modes, case_metrics, case_dir / "comparison.png")
        print(f"{query_id}: " + ", ".join(f"{mode}={float(values['dice']):.4f}" for mode, values in case_metrics.items()))
    write_csv(out_dir / "results.csv", result_rows)
    summary: dict[str, Any] = {}
    for mode in ("joint", "single_union", "component_union"):
        rows = [row for row in result_rows if row["mode"] == mode]
        summary[mode] = {
            "case_count": len(rows),
            "mean_dice": float(np.mean([float(row["dice"]) for row in rows])),
            "mean_iou": float(np.mean([float(row["iou"]) for row in rows])),
            "mean_pred_area_pct": float(np.mean([float(row["pred_area_pct"]) for row in rows])),
            "mean_pred_components": float(np.mean([float(row["pred_components"]) for row in rows])),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
