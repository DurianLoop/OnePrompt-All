from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        if size is not None and img.size != size:
            img = img.resize(size, Image.NEAREST)
        arr = np.asarray(img)
    if arr.ndim == 3:
        return np.any(arr > 0, axis=2)
    return arr > 0


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    dice = (2.0 * inter / (pred_sum + target_sum)) if (pred_sum + target_sum) else 1.0
    iou = (inter / union) if union else 1.0
    return dice, iou


def evaluate_case(row: dict[str, str], run_dir: Path) -> dict[str, Any]:
    case_dir = run_dir / row["pilot_id"]
    scores_path = case_dir / "stage3_filtering" / "filtered_scores.json"
    if not scores_path.exists():
        return {
            "pilot_id": row["pilot_id"],
            "dataset": row["dataset"],
            "status": "missing_stage3",
            "proposal_count": 0,
            "retained_count": 0,
            "best_dice": 0.0,
            "best_iou": 0.0,
            "best_retained_dice": 0.0,
            "best_retained_iou": 0.0,
        }
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    target_mask = load_mask(resolve_path(row["query_mask_path"]))
    target_size = (target_mask.shape[1], target_mask.shape[0])
    proposal_rows: list[dict[str, Any]] = []
    for metric in payload["metrics"]:
        pred = load_mask(resolve_path(metric["mask_path"]), size=target_size)
        dice, iou = dice_iou(pred, target_mask)
        proposal_rows.append({
            "group_id": int(metric["group_id"]),
            "filter_status": metric["filter_status"],
            "dice": round(float(dice), 6),
            "iou": round(float(iou), 6),
        })
    retained = [item for item in proposal_rows if item["filter_status"] == "retained"]
    best = max(proposal_rows, key=lambda item: item["dice"], default={"dice": 0.0, "iou": 0.0})
    best_retained = max(retained, key=lambda item: item["dice"], default={"dice": 0.0, "iou": 0.0})
    return {
        "pilot_id": row["pilot_id"],
        "dataset": row["dataset"],
        "status": "ok",
        "proposal_count": len(proposal_rows),
        "retained_count": len(retained),
        "best_dice": best["dice"],
        "best_iou": best["iou"],
        "best_retained_dice": best_retained["dice"],
        "best_retained_iou": best_retained["iou"],
        "proposal_metrics": proposal_rows,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_outputs(out_dir: Path, cases: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_cases = [case for case in cases if case["status"] == "ok"]
    payload = {
        "case_count": len(cases),
        "ok_case_count": len(ok_cases),
        "proposal_count": sum(int(case["proposal_count"]) for case in ok_cases),
        "retained_count": sum(int(case["retained_count"]) for case in ok_cases),
        "best_dice_mean": round(mean([float(case["best_dice"]) for case in ok_cases]), 6),
        "best_iou_mean": round(mean([float(case["best_iou"]) for case in ok_cases]), 6),
        "best_retained_dice_mean": round(mean([float(case["best_retained_dice"]) for case in ok_cases]), 6),
        "best_retained_iou_mean": round(mean([float(case["best_retained_iou"]) for case in ok_cases]), 6),
        "cases": cases,
    }
    (out_dir / "medical_pilot_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "medical_pilot_eval.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "pilot_id",
            "dataset",
            "status",
            "proposal_count",
            "retained_count",
            "best_dice",
            "best_iou",
            "best_retained_dice",
            "best_retained_iou",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key, "") for key in fields})
    lines = [
        "# Matcher Medical Pilot Eval",
        "",
        f"- Cases: `{payload['case_count']}`",
        f"- OK cases: `{payload['ok_case_count']}`",
        f"- Proposals: `{payload['proposal_count']}`",
        f"- Retained: `{payload['retained_count']}`",
        f"- Best Dice mean: `{payload['best_dice_mean']}`",
        f"- Best IoU mean: `{payload['best_iou_mean']}`",
        f"- Best retained Dice mean: `{payload['best_retained_dice_mean']}`",
        f"- Best retained IoU mean: `{payload['best_retained_iou_mean']}`",
        "",
        "## Case Table",
        "",
        "| Case | Dataset | Proposals | Retained | Best Dice | Best Retained Dice |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| {case['pilot_id']} | {case['dataset']} | {case['proposal_count']} | {case['retained_count']} | {case['best_dice']} | {case['best_retained_dice']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical Matcher pilot proposals against query masks.")
    parser.add_argument("--manifest", default="runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv")
    parser.add_argument("--run-dir", default="runs/research_direction/matcher_medical_pilot/run")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_pilot/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = resolve_path(args.manifest)
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    cases = [evaluate_case(row, run_dir) for row in read_manifest(manifest)]
    write_outputs(out_dir, cases)
    print(f"Wrote {rel(out_dir / 'medical_pilot_eval.json')}")


if __name__ == "__main__":
    main()
