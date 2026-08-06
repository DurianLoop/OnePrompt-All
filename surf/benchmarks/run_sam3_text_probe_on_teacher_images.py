from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmarks.sam3_teacher_minimal_validation import run_text_case  # noqa: E402
from demos.medical_sam_click_app import Sam3TextSegmenter  # noqa: E402
from benchmarks.sam3_teacher_minimal_validation import Case, load_rgb, write_csv  # noqa: E402


DEFAULT_IMAGES = [
    "图片/000000365419.jpg",
    "图片/000000554354.jpg",
    "图片/000000154497.jpg",
    "图片/000000201322.jpg",
    "图片/000000295440.jpg",
    "图片/000000236762.jpg",
]
DEFAULT_PROMPTS = ["person", "girl", "woman", "white girl", "execute"]


def safe_case_id(image_path: Path, prompt: str) -> str:
    stem = image_path.stem.replace(" ", "_")
    token = prompt.replace(" ", "_").replace("/", "_")
    return f"{stem}_{token}"


def read_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        return [Path(item).resolve() if Path(item).is_absolute() else ROOT / item for item in args.images]
    return [ROOT / item for item in DEFAULT_IMAGES]


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    lines = [
        "# SAM3 text probe on teacher image candidates",
        "",
        "本实验从 `D:\\SAM\\图片` 批量筛出的 text-failure 候选图里抽取少量人物图，真实运行 SAM3 text prompts。",
        "",
        "判定逻辑：如果 `person/girl/woman` 至少一个返回实例，而 `white girl` 或 `execute` 返回 0，则该图可以支撑老师关于复杂属性词/动作词 text prompt 不稳定的判断。",
        "",
        "| Image | person | girl | woman | white girl | execute | Conclusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for image, items in grouped.items():
        by_prompt = {str(item["prompt"]): item for item in items}
        counts = {prompt: int(by_prompt.get(prompt, {}).get("instances", 0) or 0) for prompt in DEFAULT_PROMPTS}
        baseline = max(counts["person"], counts["girl"], counts["woman"])
        if baseline > 0 and (counts["white girl"] == 0 or counts["execute"] == 0):
            conclusion = "supports_text_failure"
        elif baseline == 0:
            conclusion = "baseline_text_failed_too"
        else:
            conclusion = "needs_manual_review"
        lines.append(
            f"| `{image}` | {counts['person']} | {counts['girl']} | {counts['woman']} | "
            f"{counts['white girl']} | {counts['execute']} | {conclusion} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small SAM3 text probe on selected teacher image candidates.")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--checkpoint", default=str(ROOT / "assets" / "checkpoints" / "sam3_modelscope" / "sam3.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "teacher_image_batch" / "sam3_text_probe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    segmenter = Sam3TextSegmenter(checkpoint=Path(args.checkpoint), device=device, threshold=float(args.threshold))

    rows: list[dict[str, Any]] = []
    for image_path in read_images(args):
        image = load_rgb(image_path)
        for prompt in args.prompts:
            case = Case(
                case_id=safe_case_id(image_path, prompt),
                image_path=image_path,
                prompt_type="text",
                text=prompt,
                note="teacher image batch text probe",
            )
            try:
                row = run_text_case(segmenter, case, image, overlay_dir)
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "image": str(image_path.relative_to(ROOT)),
                    "prompt_type": "text",
                    "prompt": prompt,
                    "points": "",
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
            print(f"{case.case_id}: instances={row.get('instances')} error={row.get('error')}", flush=True)

    write_csv(rows, out_dir / "results.csv")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, rows)
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
