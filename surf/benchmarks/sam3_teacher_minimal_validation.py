from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from demos.medical_sam_click_app import (  # noqa: E402
    PointPrompt,
    Sam3TextSegmenter,
    choose_sam_mask,
    make_multi_overlay,
    make_overlay,
    masks_to_xyxy,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    prompt_type: str
    text: str = ""
    points: tuple[tuple[int, int, int], ...] = ()
    note: str = ""


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def summarize_masks(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    h, w = image_shape
    if len(masks) == 0:
        return {
            "instances": 0,
            "score_best": None,
            "scores": [],
            "union_area_px": 0,
            "union_area_pct": 0.0,
            "best_area_px": 0,
            "best_area_pct": 0.0,
            "boxes_xyxy": [],
            "best_box_xyxy": [],
        }

    masks_bool = masks.astype(bool)
    scores_np = np.asarray(scores, dtype=np.float32)
    best = int(np.argmax(scores_np)) if len(scores_np) else 0
    union = masks_bool.any(axis=0)
    best_mask = masks_bool[best]
    total = float(h * w)
    return {
        "instances": int(len(masks_bool)),
        "score_best": float(scores_np[best]) if len(scores_np) else None,
        "scores": [float(x) for x in scores_np.tolist()],
        "union_area_px": int(union.sum()),
        "union_area_pct": round(100.0 * float(union.sum()) / total, 4),
        "best_area_px": int(best_mask.sum()),
        "best_area_pct": round(100.0 * float(best_mask.sum()) / total, 4),
        "boxes_xyxy": np.asarray(boxes, dtype=np.float32).round(2).tolist() if len(boxes) else [],
        "best_box_xyxy": np.asarray(boxes[best], dtype=np.float32).round(2).tolist() if len(boxes) else [],
    }


def run_text_case(segmenter: Sam3TextSegmenter, case: Case, image: np.ndarray, overlay_dir: Path) -> dict[str, Any]:
    masks, boxes, scores = segmenter.predict(image, case.text)
    overlay = make_multi_overlay(image, masks, boxes, scores) if len(masks) else image.copy()
    overlay_path = overlay_dir / f"{case.case_id}.png"
    save_png(overlay, overlay_path)
    summary = summarize_masks(masks, boxes, scores, image.shape[:2])
    summary.update({
        "case_id": case.case_id,
        "image": str(case.image_path.relative_to(WORKSPACE_ROOT)),
        "prompt_type": "text",
        "prompt": case.text,
        "points": "",
        "note": case.note,
        "overlay": str(overlay_path.relative_to(WORKSPACE_ROOT)),
        "error": "",
    })
    return summary


def run_point_case(segmenter: Sam3TextSegmenter, case: Case, image: np.ndarray, overlay_dir: Path) -> dict[str, Any]:
    points = [PointPrompt(x=x, y=y, label=label) for x, y, label in case.points]
    masks, scores = segmenter.predict_interactive(
        image=image,
        prompt_type="click",
        points=points,
        box=None,
        mode="standard",
    )
    if len(masks):
        best = choose_sam_mask(masks, scores, prefer_small=False)
        best_mask = masks[best].astype(bool)
        boxes = masks_to_xyxy(masks.astype(bool))
        overlay = make_overlay(image, best_mask, None)
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        overlay = image.copy()
    overlay_path = overlay_dir / f"{case.case_id}.png"
    save_png(overlay, overlay_path)
    summary = summarize_masks(masks, boxes, scores, image.shape[:2])
    summary.update({
        "case_id": case.case_id,
        "image": str(case.image_path.relative_to(WORKSPACE_ROOT)),
        "prompt_type": "point",
        "prompt": "",
        "points": ";".join(f"{x},{y},{label}" for x, y, label in case.points),
        "note": case.note,
        "overlay": str(overlay_path.relative_to(WORKSPACE_ROOT)),
        "error": "",
    })
    return summary


def build_cases() -> list[Case]:
    elephant = WORKSPACE_ROOT / "code" / "gradio_demo" / "images" / "elephant1.png"
    person = WORKSPACE_ROOT / "sam3" / "outputs" / "sam3_test_image_person.png"
    return [
        Case("elephant_text_elephant", elephant, "text", text="elephant", note="noun text baseline"),
        Case("elephant_text_execute", elephant, "text", text="execute", note="action text negative prompt"),
        Case("elephant_point_single", elephant, "point", points=((120, 220, 1),), note="single foreground point on left elephant head/neck"),
        Case("elephant_point_cumulative_fg", elephant, "point", points=((120, 220, 1), (160, 260, 1)), note="two accumulated foreground points"),
        Case("elephant_point_posneg", elephant, "point", points=((120, 220, 1), (160, 260, 1), (395, 220, 0)), note="foreground points plus background point on other elephant"),
        Case("person_text_person", person, "text", text="person", note="generic noun text baseline"),
        Case("person_text_girl", person, "text", text="girl", note="single noun text prompt"),
        Case("person_text_white_girl", person, "text", text="white girl", note="attribute+noun text prompt from teacher note"),
        Case("person_text_execute", person, "text", text="execute", note="verb/action text prompt from teacher note"),
        Case("person_point_girl_single", person, "point", points=((540, 470, 1),), note="single foreground point on red-shirt girl torso"),
        Case("person_point_girl_posneg", person, "point", points=((540, 470, 1), (630, 470, 0), (430, 460, 0)), note="foreground point plus negative neighboring people"),
    ]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "image",
        "prompt_type",
        "prompt",
        "points",
        "instances",
        "score_best",
        "union_area_pct",
        "best_area_pct",
        "best_box_xyxy",
        "overlay",
        "note",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SAM3 validation for teacher hypotheses.")
    parser.add_argument("--checkpoint", default=str(WORKSPACE_ROOT / "assets" / "checkpoints" / "sam3_modelscope" / "sam3.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default=str(WORKSPACE_ROOT / "runs" / "teacher_minimal_validation"))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    out_dir = Path(args.out_dir).resolve()
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = Sam3TextSegmenter(
        checkpoint=Path(args.checkpoint),
        device=device,
        threshold=args.threshold,
    )

    rows: list[dict[str, Any]] = []
    for case in build_cases():
        image = load_rgb(case.image_path)
        try:
            if case.prompt_type == "text":
                row = run_text_case(segmenter, case, image, overlay_dir)
            elif case.prompt_type == "point":
                row = run_point_case(segmenter, case, image, overlay_dir)
            else:
                raise ValueError(f"Unsupported prompt_type: {case.prompt_type}")
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "image": str(case.image_path.relative_to(WORKSPACE_ROOT)),
                "prompt_type": case.prompt_type,
                "prompt": case.text,
                "points": ";".join(f"{x},{y},{label}" for x, y, label in case.points),
                "instances": 0,
                "score_best": None,
                "union_area_pct": 0.0,
                "best_area_pct": 0.0,
                "best_box_xyxy": [],
                "overlay": "",
                "note": case.note,
                "error": repr(exc),
            }
        rows.append(row)
        print(
            f"{row['case_id']}: instances={row['instances']} "
            f"score={row['score_best']} area={row['union_area_pct']} error={row['error']}",
            flush=True,
        )

    write_csv(rows, out_dir / "results.csv")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_dir / 'results.csv'}", flush=True)
    print(f"Wrote {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
