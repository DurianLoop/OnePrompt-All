from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def binary_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = float(np.logical_or(pred, gt).sum())
    dice = (2.0 * inter) / (pred_sum + gt_sum) if pred_sum + gt_sum > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    return dice, iou


def find_idrid_prediction_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for pred in root.rglob("*_sam_result.png"):
        case_id = pred.name.replace("_sam_result.png", "")
        set_dir = pred.parents[1]
        mask = set_dir / "Combined Masks" / f"{case_id}.png"
        if mask.exists():
            pairs.append((case_id, pred, mask))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate available medical prediction/mask pairs with Dice and IoU.")
    parser.add_argument("--idrid-root", default="code/datasets/IDRID")
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p2_medical_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_path(args.idrid_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_idrid_prediction_pairs(root)
    rows = []
    for case_id, pred_path, gt_path in pairs:
        gt = binary_mask(gt_path)
        pred = binary_mask(pred_path, size=(gt.shape[1], gt.shape[0]))
        dice, iou = dice_iou(pred, gt)
        rows.append({
            "case_id": case_id,
            "prediction": str(pred_path.relative_to(WORKSPACE_ROOT)),
            "ground_truth": str(gt_path.relative_to(WORKSPACE_ROOT)),
            "dice": round(float(dice), 6),
            "iou": round(float(iou), 6),
            "pred_area_px": int(pred.sum()),
            "gt_area_px": int(gt.sum()),
        })
    with (out_dir / "medical_eval.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "prediction", "ground_truth", "dice", "iou", "pred_area_px", "gt_area_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pair_count": len(rows),
        "dice_mean": round(float(np.mean([row["dice"] for row in rows])), 6) if rows else None,
        "iou_mean": round(float(np.mean([row["iou"] for row in rows])), 6) if rows else None,
        "rows": rows,
    }
    (out_dir / "medical_eval.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# Medical Prediction Pair Evaluation",
            "",
            f"- Pair count: `{summary['pair_count']}`",
            f"- Dice mean: `{summary['dice_mean']}`",
            f"- IoU mean: `{summary['iou_mean']}`",
            "",
            "## Artifacts",
            "",
            "- `medical_eval.csv`",
            "- `medical_eval.json`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Pair count: {summary['pair_count']}")
    print(f"Dice mean: {summary['dice_mean']}")
    print(f"IoU mean: {summary['iou_mean']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
