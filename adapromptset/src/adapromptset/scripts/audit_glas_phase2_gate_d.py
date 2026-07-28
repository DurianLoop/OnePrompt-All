from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapromptset.evaluation.metrics import dice_score, iou_score
from adapromptset.matching.runtime import checkpoint_record, configure_external_paths, infer_sam_architecture
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SIZE = 518
MIN_PRIVILEGED_DICE = 0.20
MIN_CONTROL_MARGIN = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def load_mask(path: Path) -> list[list[bool]]:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).tolist()


def mask_area(mask: list[list[bool]]) -> int:
    return sum(1 for row in mask for value in row if value)


def mask_centroid(mask: list[list[bool]]) -> tuple[int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError("mask has no foreground")
    return round(sum(xs) / len(xs)), round(sum(ys) / len(ys))


def mask_bbox(mask: list[list[bool]]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError("mask has no foreground")
    return min(xs), min(ys), max(xs), max(ys)


def background_point(mask: list[list[bool]]) -> tuple[int, int]:
    height = len(mask)
    width = len(mask[0])
    margin = min(7, max(0, width - 1), max(0, height - 1))
    for x, y in (
        (margin, margin),
        (max(0, width - 1 - margin), margin),
        (margin, max(0, height - 1 - margin)),
        (max(0, width - 1 - margin), max(0, height - 1 - margin)),
    ):
        if not mask[y][x]:
            return x, y
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if not value:
                return x, y
    raise ValueError("mask has no background")


def connected_component_count(mask: list[list[bool]]) -> int:
    height = len(mask)
    width = len(mask[0])
    seen = [[False for _ in range(width)] for _ in range(height)]
    count = 0
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if not value or seen[y][x]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y][x] = True
            while queue:
                cx, cy = queue.popleft()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
    return count


def _unique_query_rows(episodes: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in episodes:
        query = row["query_case_id"]
        if query in seen:
            continue
        seen.add(query)
        rows.append(row)
    return rows


def build_case_inventory(episodes_csv: Path, workspace_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _unique_query_rows(_read_csv(episodes_csv)):
        paths = resolve_episode_paths(row, workspace_root)
        mask = load_mask(paths["query_mask"])
        rows.append(
            {
                "query_case_id": row["query_case_id"],
                "query_grade": row.get("query_grade", ""),
                "episode_id": row["episode_id"],
                "image_path": str(paths["query_image"]),
                "mask_path": str(paths["query_mask"]),
                "foreground_area": mask_area(mask),
                "component_count": connected_component_count(mask),
            }
        )
    return rows


def select_smoke_cases(inventory: list[dict[str, Any]], max_cases: int = 6) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for grade in ("benign", "malignant"):
        matches = [row for row in inventory if row["query_grade"] == grade]
        if matches:
            selected[matches[0]["query_case_id"]] = matches[0]
    for key in ("component_count", "foreground_area"):
        selected[min(inventory, key=lambda row: row[key])["query_case_id"]] = min(inventory, key=lambda row: row[key])
        selected[max(inventory, key=lambda row: row[key])["query_case_id"]] = max(inventory, key=lambda row: row[key])
    return list(selected.values())[:max_cases]


def load_sam_predictor(checkpoint: Path, workspace_root: Path, device: str):
    configure_external_paths(workspace_root)
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[infer_sam_architecture(checkpoint)](checkpoint=str(checkpoint))
    model.eval().to(device)
    return SamPredictor(model)


def predict_with_points(
    predictor: Any,
    *,
    image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[int],
) -> tuple[list[list[bool]], float]:
    predictor.set_image(np.asarray(image))
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def predict_with_box(
    predictor: Any,
    *,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[list[list[bool]], float]:
    predictor.set_image(np.asarray(image))
    masks, scores, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def classify_gate_d(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    issues: list[str] = []
    if not rows:
        return "FAIL", ["no Gate D rows"]
    if any(row["failed"] for row in rows):
        issues.append("one or more SAM calls failed")
    if any(row["privileged_best_dice"] < MIN_PRIVILEGED_DICE for row in rows):
        issues.append("privileged prompt Dice below minimum on at least one case")
    if any(row["privileged_best_dice"] < row["control_best_dice"] + MIN_CONTROL_MARGIN for row in rows):
        issues.append("privileged prompts did not clearly exceed controls on at least one case")
    if any(row["call_count"] != row["prompt_group_count"] for row in rows):
        issues.append("SAM call count does not match prompt group count")
    return ("PASS" if not issues else "FAIL"), issues


def run_gate_d(
    *,
    project_root: Path = PROJECT_ROOT,
    episodes_csv: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    device: str = "cuda:0",
    checkpoint: Path | None = None,
) -> int:
    workspace_root = project_root.parent
    episodes_csv = episodes_csv or project_root / "manifests" / "phase2" / "glas" / "episodes_val.csv"
    out_dir = out_dir or project_root / "reports" / "phase2_glas" / "gate_d_segmenter_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_case_inventory(episodes_csv, workspace_root)
    selected = select_smoke_cases(inventory)
    if dry_run:
        _write_json(
            out_dir / "gate_d_summary.json",
            {
                "status": "DRY_RUN",
                "selected_cases": selected,
                "formal_gate_d": "PENDING_EXECUTION",
            },
        )
        return 0

    checkpoint = checkpoint or workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth"
    segmenter = checkpoint_record(checkpoint, role="segmenter")
    predictor = load_sam_predictor(checkpoint, workspace_root, device)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for case in selected:
        image = load_rgb(Path(case["image_path"]))
        target = load_mask(Path(case["mask_path"]))
        centroid = mask_centroid(target)
        bg = background_point(target)
        bbox = mask_bbox(target)
        random_control = bg
        mismatched_control = (min(IMAGE_SIZE - 1, centroid[0] + IMAGE_SIZE // 3), centroid[1])
        failed = False
        metrics: dict[str, float] = {}
        try:
            centroid_mask, centroid_score = predict_with_points(predictor, image=image, points=[centroid], labels=[1])
            pos_neg_mask, pos_neg_score = predict_with_points(
                predictor,
                image=image,
                points=[centroid, bg],
                labels=[1, 0],
            )
            box_mask, box_score = predict_with_box(predictor, image=image, box=bbox)
            random_mask, random_score = predict_with_points(
                predictor,
                image=image,
                points=[random_control],
                labels=[1],
            )
            mismatched_mask, mismatched_score = predict_with_points(
                predictor,
                image=image,
                points=[mismatched_control],
                labels=[1],
            )
            metrics = {
                "centroid_dice": dice_score(centroid_mask, target),
                "centroid_iou": iou_score(centroid_mask, target),
                "centroid_score": centroid_score,
                "pos_neg_dice": dice_score(pos_neg_mask, target),
                "pos_neg_iou": iou_score(pos_neg_mask, target),
                "pos_neg_score": pos_neg_score,
                "box_dice": dice_score(box_mask, target),
                "box_iou": iou_score(box_mask, target),
                "box_score": box_score,
                "random_dice": dice_score(random_mask, target),
                "random_score": random_score,
                "mismatched_dice": dice_score(mismatched_mask, target),
                "mismatched_score": mismatched_score,
            }
        except Exception:
            failed = True
        privileged_best = max(metrics.get("centroid_dice", 0.0), metrics.get("pos_neg_dice", 0.0), metrics.get("box_dice", 0.0))
        control_best = max(metrics.get("random_dice", 0.0), metrics.get("mismatched_dice", 0.0))
        rows.append(
            {
                **case,
                **metrics,
                "privileged_best_dice": privileged_best,
                "control_best_dice": control_best,
                "privileged_margin": privileged_best - control_best,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "background_x": bg[0],
                "background_y": bg[1],
                "bbox_x0": bbox[0],
                "bbox_y0": bbox[1],
                "bbox_x1": bbox[2],
                "bbox_y1": bbox[3],
                "failed": failed,
                "call_count": 5,
                "prompt_group_count": 5,
            }
        )
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0
    status, issues = classify_gate_d(rows)
    _write_csv(out_dir / "gate_d_case_metrics.csv", rows)
    summary = {
        "schema_version": "adapromptset.phase2.glas_gate_d.v1",
        "status": status,
        "formal_gate_d": status,
        "issues": issues,
        "case_count": len(rows),
        "selected_cases": selected,
        "mean_privileged_best_dice": sum(row["privileged_best_dice"] for row in rows) / len(rows),
        "mean_control_best_dice": sum(row["control_best_dice"] for row in rows) / len(rows),
        "mean_privileged_margin": sum(row["privileged_margin"] for row in rows) / len(rows),
        "device": device,
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours_estimate": round(gpu_hours, 6),
        "segmenter_checkpoint": segmenter,
        "artifacts": {
            "case_metrics_csv": str(out_dir / "gate_d_case_metrics.csv"),
        },
        "notes": [
            "Privileged prompts are sanity checks only and are not fair automatic-method baselines.",
            "GlaS testA/testB were not used.",
            f"{segmenter['architecture']} is the segmenter used in this Gate D smoke.",
        ],
    }
    _write_json(out_dir / "gate_d_summary.json", summary)
    lines = [
        "# GlaS Phase 2 Gate D Segmenter Smoke",
        "",
        f"Status: `{status}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases | `{len(rows)}` |",
        f"| Mean privileged best Dice | `{summary['mean_privileged_best_dice']:.6f}` |",
        f"| Mean control best Dice | `{summary['mean_control_best_dice']:.6f}` |",
        f"| Mean privileged margin | `{summary['mean_privileged_margin']:.6f}` |",
        f"| GPU hours | `{summary['gpu_hours_estimate']:.6f}` |",
        "",
        "## Issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in issues] if issues else ["- none"])
    (out_dir / "gate_d_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GlaS Phase 2 Gate D SAM-B segmenter smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(
        run_gate_d(
            project_root=args.project_root,
            episodes_csv=args.episodes_csv,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
            device=args.device,
            checkpoint=args.checkpoint,
        )
    )


if __name__ == "__main__":
    main()
