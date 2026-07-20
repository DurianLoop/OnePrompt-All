from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from PIL import Image

from adapromptset.evaluation.metrics import dice_score, iou_score
from adapromptset.matching.runtime import checkpoint_record, configure_external_paths, infer_sam_architecture

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518
CONTROL_ROWS = ("gt_centroid_point", "gt_box", "full_image_box")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def load_mask_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def foreground_centroid_point(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("mask has no foreground")
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    distances = (xs.astype(float) - center_x) ** 2 + (ys.astype(float) - center_y) ** 2
    best = int(np.argmin(distances))
    return int(xs[best]), int(ys[best])


def foreground_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("mask has no foreground")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def box_area_ratio(box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    width = max(0, x1 - x0 + 1)
    height = max(0, y1 - y0 + 1)
    return float(width * height / (IMAGE_SIZE * IMAGE_SIZE))


def _as_bool_list(mask: np.ndarray) -> list[list[bool]]:
    return mask.astype(bool).tolist()


def _metric_row(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _as_bool_list(prediction)
    tgt = _as_bool_list(target)
    pred_area = float(prediction.sum() / prediction.size)
    target_area = float(target.sum() / target.size)
    return {
        "dice": dice_score(pred, tgt),
        "iou": iou_score(pred, tgt),
        "prediction_area_ratio": pred_area,
        "target_area_ratio": target_area,
    }


def _contains_test_path(row: dict[str, str]) -> bool:
    values = [row.get("query_image_path", ""), row.get("query_mask_path", "")]
    lowered = [value.replace("/", "\\").lower() for value in values]
    return any("\\test\\" in value or value.endswith("\\test") for value in lowered)


def load_sam_predictor(checkpoint: Path, workspace_root: Path, device: str):
    configure_external_paths(workspace_root)
    from segment_anything import SamPredictor, sam_model_registry

    architecture = infer_sam_architecture(checkpoint)
    model = sam_model_registry[architecture](checkpoint=str(checkpoint))
    model.eval().to(device)
    return SamPredictor(model)


def _best_sam_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def _predict_point(predictor: Any, point: tuple[int, int]) -> tuple[np.ndarray, float]:
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray([point], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        multimask_output=True,
    )
    return _best_sam_mask(masks, scores)


def _predict_box(predictor: Any, box: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    masks, scores, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=True,
    )
    return _best_sam_mask(masks, scores)


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_control: dict[str, dict[str, Any]] = {}
    for control in CONTROL_ROWS:
        bucket = [row for row in rows if row["control_type"] == control]
        if not bucket:
            continue
        dice_values = [float(row["dice"]) for row in bucket]
        iou_values = [float(row["iou"]) for row in bucket]
        by_control[control] = {
            "n": len(bucket),
            "mean_dice": mean(dice_values),
            "median_dice": median(dice_values),
            "mean_iou": mean(iou_values),
            "median_iou": median(iou_values),
            "failure_fraction_dice_lt_0_05": sum(value < 0.05 for value in dice_values) / len(dice_values),
            "mean_prediction_area_ratio": mean(float(row["prediction_area_ratio"]) for row in bucket),
        }
    return by_control


def run_route_b_ddr_privileged_sam_controls(
    *,
    project_root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    selected_csv: Path | None = None,
    checkpoint: Path | None = None,
    dry_run: bool = False,
    device_text: str = "auto",
) -> dict[str, Any]:
    out_dir = out_dir or (project_root / "reports" / "route_b_ddr_privileged_sam_controls")
    selected_csv = selected_csv or (
        project_root / "reports" / "route_b_ddr_stratified_matcher" / "selected_episodes.csv"
    )
    checkpoint = checkpoint or (project_root.parent / "assets" / "checkpoints" / "sam_vit_b.pth")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _read_csv(selected_csv)
    issues: list[dict[str, str]] = []
    for row in selected:
        if not row.get("query_image_path"):
            issues.append({"query_unit_id": row.get("query_unit_id", ""), "issue": "missing_query_image_path"})
        if not row.get("query_mask_path"):
            issues.append({"query_unit_id": row.get("query_unit_id", ""), "issue": "missing_query_mask_path"})
        if _contains_test_path(row):
            issues.append({"query_unit_id": row.get("query_unit_id", ""), "issue": "test_path_detected"})
    base_payload: dict[str, Any] = {
        "schema_version": "adapromptset.route_b.ddr_privileged_sam_controls.v1",
        "date": "2026-07-19",
        "route": "B_reference_promptability_benchmark",
        "dataset": "DDR",
        "source_selected_episodes": str(selected_csv),
        "selected_episode_count": len(selected),
        "formal_test_access": False,
        "test_files_read": False,
        "segmenter": "SAM-B",
        "segmenter_checkpoint": str(checkpoint),
        "uses_query_gt_for_generation": True,
        "uses_query_gt_for_evaluation": True,
        "automatic_method_result": False,
        "privileged_upper_bound": True,
        "issues": issues,
        "outputs": {
            "query_metrics": str(out_dir / "privileged_sam_query_metrics.csv"),
            "summary": str(out_dir / "privileged_sam_summary.json"),
        },
        "claim_boundary": "privileged_prompt_interface_controls_not_automatic_method_result",
    }
    if dry_run:
        payload = {**base_payload, "status": "DRY_RUN"}
        _write_json(out_dir / "privileged_sam_summary.json", payload)
        return payload
    if issues:
        payload = {**base_payload, "status": "FAIL_ROUTE_B_DDR_PRIVILEGED_SAM_CONTROL_INPUTS"}
        _write_json(out_dir / "privileged_sam_summary.json", payload)
        return payload

    device = torch.device(
        "cuda:0" if device_text == "auto" and torch.cuda.is_available() else ("cpu" if device_text == "auto" else device_text)
    )
    segmenter = checkpoint_record(checkpoint, role="segmenter")
    start = time.perf_counter()
    predictor = load_sam_predictor(checkpoint, project_root.parent, str(device))
    metric_rows: list[dict[str, Any]] = []
    for row in selected:
        image = load_rgb(Path(row["query_image_path"]))
        target = load_mask_array(Path(row["query_mask_path"]))
        point = foreground_centroid_point(target)
        gt_box = foreground_box(target)
        full_box = (0, 0, IMAGE_SIZE - 1, IMAGE_SIZE - 1)
        predictor.set_image(np.asarray(image))
        predictions = [
            ("gt_centroid_point", _predict_point(predictor, point), point, ""),
            ("gt_box", _predict_box(predictor, gt_box), "", gt_box),
            ("full_image_box", _predict_box(predictor, full_box), "", full_box),
        ]
        for control_type, (prediction, sam_score), prompt_point, prompt_box in predictions:
            metrics = _metric_row(prediction, target)
            privileged = control_type != "full_image_box"
            metric_rows.append(
                {
                    "dataset": "DDR",
                    "split": row["split"],
                    "intent": row["intent"],
                    "query_unit_id": row["query_unit_id"],
                    "query_image_id": row["query_image_id"],
                    "control_type": control_type,
                    "segmenter": "SAM-B",
                    "prompt_point": prompt_point,
                    "prompt_box": prompt_box,
                    "prompt_box_area_ratio": box_area_ratio(prompt_box) if prompt_box else "",
                    "sam_score": sam_score,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "prediction_area_ratio": metrics["prediction_area_ratio"],
                    "target_area_ratio": metrics["target_area_ratio"],
                    "uses_query_gt_for_generation": privileged,
                    "uses_query_gt_for_evaluation": True,
                    "privileged_upper_bound": privileged,
                    "automatic_method_result": False,
                    "formal_test_row": False,
                }
            )
    elapsed = time.perf_counter() - start
    _write_csv(out_dir / "privileged_sam_query_metrics.csv", metric_rows)
    summary = _summaries(metric_rows)
    payload = {
        **base_payload,
        "status": "PASS_ROUTE_B_DDR_PRIVILEGED_SAM_CONTROLS",
        "segmenter_checkpoint_record": segmenter,
        "device": str(device),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
        "control_summary": summary,
    }
    _write_json(out_dir / "privileged_sam_summary.json", payload)
    _write_markdown(out_dir / "privileged_sam_summary.md", payload)
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Route B DDR Privileged SAM Controls",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Selected episodes: `{payload['selected_episode_count']}`",
        f"Formal test access: `{str(payload['formal_test_access']).lower()}`",
        f"DDR test files read: `{str(payload['test_files_read']).lower()}`",
        f"Claim boundary: `{payload['claim_boundary']}`",
        "",
        "## Control Summary",
        "",
        "| Control | N | Mean Dice | Median Dice | Mean IoU | Failure Dice<0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for control, row in payload.get("control_summary", {}).items():
        lines.append(
            f"| `{control}` | {row['n']} | {row['mean_dice']:.6f} | {row['median_dice']:.6f} | "
            f"{row['mean_iou']:.6f} | {row['failure_fraction_dice_lt_0_05']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- GT centroid and GT box rows use query GT for prompt generation and are privileged upper bounds.",
            "- Full-image box is a non-privileged control, not an automatic reference-guided method.",
            "- DDR test remains locked.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DDR Route B privileged SAM prompt-interface controls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--selected-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_route_b_ddr_privileged_sam_controls(
        project_root=args.project_root,
        out_dir=args.out_dir,
        selected_csv=args.selected_csv,
        checkpoint=args.checkpoint,
        dry_run=args.dry_run,
        device_text=args.device,
    )


if __name__ == "__main__":
    main()
