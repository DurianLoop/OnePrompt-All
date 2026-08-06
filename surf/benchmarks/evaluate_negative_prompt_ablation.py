from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def select_negative_points(
    score_map: np.ndarray,
    original_size: tuple[int, int],
    count: int,
    positive_coords: np.ndarray,
    min_distance: float = 18.0,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    height, width = score_map.shape
    target_width, target_height = original_size
    positives = np.asarray(positive_coords, dtype=np.float32).reshape(-1, 2)
    selected: list[list[float]] = []
    for index in np.argsort(score_map.reshape(-1)).tolist():
        patch_y, patch_x = divmod(int(index), width)
        x = (patch_x + 0.5) * target_width / width
        y = (patch_y + 0.5) * target_height / height
        point = np.asarray([x, y], dtype=np.float32)
        if positives.size and float(np.linalg.norm(positives - point, axis=1).min()) < min_distance:
            continue
        if selected:
            previous = np.asarray(selected, dtype=np.float32)
            if float(np.linalg.norm(previous - point, axis=1).min()) < min_distance:
                continue
        selected.append([float(x), float(y)])
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def select_backend_loader(backend: str, loaders: dict[str, Any]) -> Any:
    if backend not in loaders:
        raise ValueError(f"Unsupported SAM backend: {backend}")
    return loaders[backend]


def score_path_from_points(points_path: Path) -> Path:
    return points_path.parent.parent / "t1_matching" / points_path.name.replace("_points.csv", "_score.npy")


def artifact_stem(episode_id: str, query_case_id: str, negative_count: int) -> str:
    safe_episode = "".join(character if character.isalnum() or character in "-_" else "_" for character in episode_id)
    return f"{safe_episode}__{query_case_id}__neg{negative_count}"


def pending_negative_counts(
    episode_id: str,
    negative_counts: list[int],
    completed: set[tuple[str, int]],
) -> list[int]:
    return [count for count in negative_counts if (episode_id, count) not in completed]


def completed_negative_budget_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (
            str(row["episode_id"]),
            int(row.get("requested_negative_count", row["negative_count"])),
        )
        for row in rows
    }


def save_overlay(
    image: Image.Image,
    gt: np.ndarray,
    pred: np.ndarray,
    positive_coords: np.ndarray,
    negative_coords: np.ndarray,
    path: Path,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    rgb[tp] = (0.55 * rgb[tp] + 0.45 * np.asarray([30, 220, 90])).astype(np.uint8)
    rgb[fp] = (0.45 * rgb[fp] + 0.55 * np.asarray([255, 45, 45])).astype(np.uint8)
    rgb[fn] = (0.45 * rgb[fn] + 0.55 * np.asarray([30, 120, 255])).astype(np.uint8)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for x, y in positive_coords:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 245, 0), outline=(0, 0, 0), width=2)
    for x, y in negative_coords:
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=(0, 255, 255), width=3)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=(0, 255, 255), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        requested = int(row.get("requested_negative_count", row["negative_count"]))
        grouped[requested].append(row)
    metrics = ("dice", "iou", "object_recall", "merge_error", "pred_area_pct")
    return [
        {
            "requested_negative_count": requested_count,
            "case_count": len(group),
            "mean_actual_negative_count": mean(float(row["negative_count"]) for row in group),
            **{f"mean_{metric}": mean(float(row[metric]) for row in group) for metric in metrics},
        }
        for requested_count, group in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM positive-only versus explicit negative-point prompts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--negative-counts", default="0,5,10,20")
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-device", default="auto")
    parser.add_argument("--sam-backend", choices=("sam", "sammed2d"), default="sam")
    parser.add_argument("--sammed2d-repo", default="SAM-Med2D")
    parser.add_argument("--sammed2d-image-size", type=int, default=256)
    parser.add_argument("--sammed2d-no-encoder-adapter", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minimal-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = {row["pilot_id"]: row for row in read_csv(resolve(args.manifest))}
    candidates = sorted(
        [row for row in read_csv(resolve(args.candidates)) if row["method"] == args.fixed_action],
        key=lambda row: (row["query_case_id"], row["case_name"]),
    )
    negative_counts = [int(item) for item in args.negative_counts.split(",") if item.strip()]
    out_dir = resolve(args.out_dir)
    progress_path = out_dir / "per_case_progress.csv"
    rows: list[dict[str, Any]] = (
        [dict(row) for row in read_csv(progress_path)]
        if args.resume and progress_path.exists()
        else []
    )
    completed = completed_negative_budget_keys(rows)
    expected = {(candidate["case_name"], count) for candidate in candidates for count in negative_counts}
    if not expected.issuperset(completed):
        unexpected = sorted(completed - expected)
        raise ValueError(f"Progress file contains rows outside the requested experiment: {unexpected[:5]}")
    if args.sam_backend == "sam":
        from run_monuseg_dense_prompt_experiment import dice_iou, load_mask, load_sam_predictor, object_metrics

        predictor, _ = load_sam_predictor(resolve(args.sam_checkpoint), args.sam_model_type, args.sam_device)
    else:
        from run_monuseg_sammed2d_from_points import load_sammed2d_predictor

        device = args.sam_device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = load_sammed2d_predictor(
            resolve(args.sammed2d_repo),
            resolve(args.sam_checkpoint),
            device,
            args.sammed2d_image_size,
            not args.sammed2d_no_encoder_adapter,
        )
        from run_monuseg_dense_prompt_experiment import dice_iou, load_mask, object_metrics
    active_query_id: str | None = None
    for candidate in candidates:
        case_name = candidate["case_name"]
        pending_counts = pending_negative_counts(case_name, negative_counts, completed)
        if not pending_counts:
            print(f"[negative-ablation] resume-skip {case_name}")
            continue
        episode = manifest[case_name]
        image = Image.open(resolve(episode["query_image_path"])).convert("RGB")
        gt = load_mask(resolve(episode["query_mask_path"]), image.size)
        if episode["query_case_id"] != active_query_id:
            predictor.set_image(np.asarray(image))
            active_query_id = episode["query_case_id"]
        points_path = resolve(candidate["points_path"])
        point_rows = read_csv(points_path)
        positive_coords = np.asarray([[float(row["x"]), float(row["y"])] for row in point_rows], dtype=np.float32)
        score_map = np.load(score_path_from_points(points_path))
        for negative_count in pending_counts:
            negative_coords = select_negative_points(score_map, image.size, negative_count, positive_coords)
            coords = np.concatenate([positive_coords, negative_coords], axis=0)
            labels = np.concatenate([
                np.ones(len(positive_coords), dtype=np.int32),
                np.zeros(len(negative_coords), dtype=np.int32),
            ])
            masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
            best = int(np.argmax(scores))
            pred = masks[best].astype(bool)
            dice, iou = dice_iou(pred, gt)
            obj = object_metrics(pred, gt)
            inter = float(np.logical_and(pred, gt).sum())
            pred_sum = float(pred.sum())
            gt_sum = float(gt.sum())
            stem = artifact_stem(case_name, episode["query_case_id"], negative_count)
            mask_path = out_dir / "masks" / f"{stem}.png"
            overlay_path = out_dir / "overlays" / f"{stem}.png"
            if not args.minimal_artifacts:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.astype(np.uint8) * 255).save(mask_path)
                save_overlay(image, gt, pred, positive_coords, negative_coords, overlay_path)
            rows.append({
                "episode_id": case_name,
                "query_case_id": episode["query_case_id"],
                "positive_count": len(positive_coords),
                "requested_negative_count": negative_count,
                "negative_count": len(negative_coords),
                "dice": dice,
                "iou": iou,
                "fg_precision": inter / pred_sum if pred_sum else 0.0,
                "fg_recall": inter / gt_sum if gt_sum else 1.0,
                "object_recall": obj["object_recall"],
                "merge_error": obj["merge_error"],
                "pred_area_pct": 100.0 * pred_sum / pred.size,
                "sam_score": float(scores[best]),
                "mask_path": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
                "overlay_path": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
                "points_path": candidate["points_path"],
            })
            completed.add((case_name, negative_count))
        rows.sort(key=lambda row: (str(row["episode_id"]), int(row["negative_count"])))
        write_csv(progress_path, rows)
        print(
            f"[negative-ablation] {episode['query_case_id']} "
            f"settings={len(pending_counts)} completed={len(completed)}/{len(expected)}"
        )

    summary = summarize(rows)
    write_csv(out_dir / "per_case.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "metadata.json").write_text(json.dumps({
        "fixed_action": args.fixed_action,
        "negative_counts": negative_counts,
        "case_count": len(candidates),
        "selection": "lowest fg-minus-bg contrast patches with spatial exclusion",
        "sam_backend": args.sam_backend,
        "sam_checkpoint": args.sam_checkpoint,
        "minimal_artifacts": args.minimal_artifacts,
        "resume_enabled": args.resume,
        "completed_setting_count": len(completed),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
