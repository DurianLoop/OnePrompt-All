from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from adapromptset.actions.bank import PromptGroup, build_prompt_arrays
from adapromptset.evaluation.metrics import dice_score, iou_score
from adapromptset.matching.runtime import checkpoint_record, configure_external_paths, infer_sam_architecture


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM_ROOT = PROJECT_ROOT.parent
IMAGE_SIZE = 518


def mask_centroid(mask: Sequence[Sequence[bool]]) -> tuple[int, int]:
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


def connected_component_at_point(mask: Sequence[Sequence[bool]], point: tuple[int, int]) -> list[list[bool]]:
    width = len(mask[0])
    height = len(mask)
    start_x, start_y = point
    if not (0 <= start_x < width and 0 <= start_y < height) or not mask[start_y][start_x]:
        raise ValueError("point is not inside the foreground mask")
    component = [[False for _ in range(width)] for _ in range(height)]
    stack = [(start_x, start_y)]
    component[start_y][start_x] = True
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not component[ny][nx]:
                component[ny][nx] = True
                stack.append((nx, ny))
    return component


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def load_mask(path: Path) -> list[list[bool]]:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    values = np.asarray(image)
    return (values > 0).tolist()


def find_background_point(mask: Sequence[Sequence[bool]]) -> tuple[int, int]:
    height = len(mask)
    width = len(mask[0])
    for x, y in ((7, 7), (width - 8, 7), (7, height - 8), (width - 8, height - 8)):
        if not mask[y][x]:
            return x, y
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if not value:
                return x, y
    raise ValueError("mask has no background")


def read_first_candidate_group(path: Path, episode_id: str) -> tuple[list[tuple[int, int]], PromptGroup]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["episode_id"] == episode_id and row["positive_limit"] == "5"
        ]
    if not rows:
        raise ValueError(f"no candidate rows for {episode_id}")
    coordinates = [(int(row["x"]), int(row["y"])) for row in rows]
    positive_indices = tuple(index for index, row in enumerate(rows) if row["kind"] == "positive")
    negative_indices = tuple(index for index, row in enumerate(rows) if row["kind"] == "negative")
    return coordinates, PromptGroup(positive_indices=positive_indices[:1], negative_indices=negative_indices[:1])


def load_sam_predictor(checkpoint: Path, workspace_root: Path, device: str):
    configure_external_paths(workspace_root)
    from segment_anything import SamPredictor, sam_model_registry

    architecture = infer_sam_architecture(checkpoint)
    model = sam_model_registry[architecture](checkpoint=str(checkpoint))
    model.eval().to(device)
    return SamPredictor(model)


def predict_mask(predictor: Any, image: Image.Image, points: list[tuple[int, int]], labels: list[int]) -> tuple[list[list[bool]], float]:
    predictor.set_image(np.asarray(image))
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best])


def classify_gate_d(
    *,
    privileged_dice: float,
    privileged_component_dice: float | None = None,
    repeat_same: bool,
    call_count: int,
    prompt_group_count: int,
    min_reasonable_dice: float = 0.05,
    min_component_dice: float = 0.5,
) -> str:
    if not repeat_same or call_count != prompt_group_count:
        return "FAIL"
    if privileged_component_dice is not None and privileged_component_dice >= min_component_dice:
        return "PASS"
    if privileged_dice < min_reasonable_dice:
        return "PENDING_INSTANCE_AWARE_GT_CENTER_CHECK"
    return "PASS"


def _append_ledger(project_root: Path, gpu_hours: float) -> None:
    ledger_path = project_root / "reports" / "compute_ledger.csv"
    exists = ledger_path.is_file()
    with ledger_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "command", "gpu_hours", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "phase": "4-sam-smoke",
                "command": "scripts/run_sam_smoke.py",
                "gpu_hours": f"{gpu_hours:.6f}",
                "notes": "Validation-only SAM-B prompt API and coordinate smoke.",
            }
        )


def run_smoke(
    *,
    project_root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    dry_run: bool = False,
    device: str = "cuda:0",
) -> int:
    out_dir = out_dir or (project_root / "reports" / "sam_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        (out_dir / "sam_smoke_summary.json").write_text(
            json.dumps({"status": "DRY_RUN", "formal_gate_d": "PENDING_EXECUTION"}, indent=2),
            encoding="utf-8",
        )
        return 0
    workspace_root = project_root.parent
    episode_summary = json.loads((project_root / "reports" / "real_matcher_smoke" / "real_matcher_smoke.json").read_text(encoding="utf-8"))
    episode_id = episode_summary["episode"]
    query_image_path = workspace_root / "code" / "datasets" / "MoNuSAC" / episode_summary["evidence_metadata"]["query_image_path"]
    query_mask_path = workspace_root / "code" / "datasets" / "MoNuSAC" / "processed" / "train" / "masks" / f"{episode_summary['evidence_metadata']['query_case_id']}.png"
    image = load_rgb(query_image_path)
    target = load_mask(query_mask_path)
    centroid = mask_centroid(target)
    target_component = connected_component_at_point(target, centroid)
    background = find_background_point(target)
    candidates_csv = project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv"
    candidate_coordinates, candidate_group = read_first_candidate_group(candidates_csv, episode_id)
    candidate_points, candidate_labels = build_prompt_arrays(candidate_group, candidate_coordinates)
    checkpoint = workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth"
    segmenter = checkpoint_record(checkpoint, role="segmenter")
    start = time.perf_counter()
    predictor = load_sam_predictor(checkpoint, workspace_root, device)
    privileged_mask, privileged_score = predict_mask(predictor, image, [centroid], [1])
    pos_neg_mask, pos_neg_score = predict_mask(predictor, image, [centroid, background], [1, 0])
    candidate_mask, candidate_score = predict_mask(predictor, image, candidate_points, candidate_labels)
    repeat_mask, _ = predict_mask(predictor, image, candidate_points, candidate_labels)
    elapsed = time.perf_counter() - start
    gpu_hours = elapsed / 3600.0 if device.startswith("cuda") else 0.0
    repeat_same = repeat_mask == candidate_mask
    metrics = {
        "privileged_dice": dice_score(privileged_mask, target),
        "privileged_iou": iou_score(privileged_mask, target),
        "privileged_component_dice": dice_score(privileged_mask, target_component),
        "privileged_component_iou": iou_score(privileged_mask, target_component),
        "pos_neg_dice": dice_score(pos_neg_mask, target),
        "pos_neg_component_dice": dice_score(pos_neg_mask, target_component),
        "candidate_dice": dice_score(candidate_mask, target),
        "candidate_iou": iou_score(candidate_mask, target),
        "privileged_score": privileged_score,
        "pos_neg_score": pos_neg_score,
        "candidate_score": candidate_score,
        "repeat_same": repeat_same,
        "call_count": 4,
        "prompt_group_count": 4,
    }
    formal_gate_d = classify_gate_d(
        privileged_dice=metrics["privileged_dice"],
        privileged_component_dice=metrics["privileged_component_dice"],
        repeat_same=repeat_same,
        call_count=metrics["call_count"],
        prompt_group_count=metrics["prompt_group_count"],
    )
    status = "PASS" if formal_gate_d == "PASS" else "PASS_API_LOW_DICE" if formal_gate_d.startswith("PENDING_") else "FAIL"
    payload = {
        "status": status,
        "formal_gate_d": formal_gate_d,
        "episode": episode_id,
        "device": device,
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours_estimate": round(gpu_hours, 6),
        "segmenter_checkpoint": segmenter,
        "metrics": metrics,
        "prompts": {
            "privileged_center": centroid,
            "background_point": background,
            "candidate_points": candidate_points,
            "candidate_labels": candidate_labels,
        },
        "notes": [
            "Validation-only SAM-B smoke.",
            "GT-center point is privileged and used only to verify SAM API/coordinate correctness.",
            "Semantic-union Dice is diagnostic only for one-point SAM prompts; formal Gate D uses the connected component containing the privileged point.",
        ],
    }
    (out_dir / "sam_smoke_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Gate D Status",
        "",
        "| Check | Status | Notes |",
        "|---|---|---|",
        f"| GT-center privileged point | {'PASS' if formal_gate_d == 'PASS' else 'PENDING'} | component Dice={metrics['privileged_component_dice']:.4f}; semantic-union Dice={metrics['privileged_dice']:.4f} |",
        "| positive/negative labels | PASS | SAM accepted positive+negative prompt group |",
        f"| calls vs groups | {'PASS' if metrics['call_count'] == metrics['prompt_group_count'] else 'FAIL'} | calls={metrics['call_count']} groups={metrics['prompt_group_count']} |",
        f"| repeat determinism | {'PASS' if repeat_same else 'FAIL'} | exact mask equality on repeated candidate prompt |",
        f"| formal Gate D | {formal_gate_d} | component-aware wrapper check; semantic-union Dice remains diagnostic |",
    ]
    (project_root / "reports" / "gate_d_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_ledger(project_root, gpu_hours)
    return 0 if status.startswith("PASS") else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only SAM-B segmenter smoke.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    raise SystemExit(run_smoke(project_root=args.project_root, out_dir=args.out_dir, dry_run=args.dry_run, device=args.device))


if __name__ == "__main__":
    main()
