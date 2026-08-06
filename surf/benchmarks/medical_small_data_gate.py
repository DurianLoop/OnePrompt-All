from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".nii", ".nii.gz", ".mha", ".npy"}
MEDICAL_HINTS = {
    "monuseg",
    "idrid",
    "fundus",
    "lesion",
    "polyp",
    "cell",
    "nucleus",
    "vessel",
    "medical",
    "medsam",
    "mask",
    "label",
    "gt",
}


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def suffix_name(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def is_image_file(path: Path) -> bool:
    return suffix_name(path) in IMAGE_EXTS


def looks_medical(path: Path) -> bool:
    text = str(path).lower()
    return any(hint in text for hint in MEDICAL_HINTS)


def is_mask_candidate(path: Path) -> bool:
    text = path.name.lower()
    return any(token in text for token in ["mask", "label", "gt", "annotation", "seg"])


def discover_medical_files(search_roots: list[Path], max_files: int) -> tuple[list[Path], list[Path]]:
    image_candidates: list[Path] = []
    mask_candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(image_candidates) >= max_files:
                break
            if not path.is_file() or not is_image_file(path):
                continue
            if not looks_medical(path):
                continue
            image_candidates.append(path)
            if is_mask_candidate(path):
                mask_candidates.append(path)
    return image_candidates, mask_candidates


def modality_summary(paths: list[Path]) -> dict[str, int]:
    counts = Counter()
    for path in paths:
        text = str(path).lower()
        if "fundus" in text or "idrid" in text:
            counts["fundus"] += 1
        elif "monuseg" in text or "nucleus" in text or "cell" in text:
            counts["histology_cell"] += 1
        elif "polyp" in text:
            counts["endoscopy_polyp"] += 1
        elif "nii" in text or "mri" in text or "ct" in text:
            counts["3d_volume"] += 1
        else:
            counts["unknown_medical_like"] += 1
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover whether local medical small-data closure can run with masks.")
    parser.add_argument("--search-roots", default="code,assets,runs,MedSAM,SAM-Med2D,Medical-SAM-Adapter")
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p2_medical_gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [resolve_path(item.strip()) for item in args.search_roots.split(",") if item.strip()]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images, masks = discover_medical_files(roots, max_files=args.max_files)
    can_run_dice_iou = len(images) >= 2 and len(masks) >= 1
    report = {
        "search_roots": [str(path.relative_to(WORKSPACE_ROOT)) if path.is_relative_to(WORKSPACE_ROOT) else str(path) for path in roots],
        "image_candidate_count": len(images),
        "mask_candidate_count": len(masks),
        "can_run_dice_iou": can_run_dice_iou,
        "reason": "medical data with masks found" if can_run_dice_iou else "no sufficient local medical image/mask pair discovered",
        "modality_summary": modality_summary(images),
        "image_candidates": [str(path.relative_to(WORKSPACE_ROOT)) for path in images[:100]],
        "mask_candidates": [str(path.relative_to(WORKSPACE_ROOT)) for path in masks[:100]],
    }
    (out_dir / "medical_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# Medical Small-Data Gate Summary",
            "",
            f"- Image candidates: `{len(images)}`",
            f"- Mask candidates: `{len(masks)}`",
            f"- Can run Dice/IoU now: `{can_run_dice_iou}`",
            f"- Reason: `{report['reason']}`",
            f"- Modality summary: `{report['modality_summary']}`",
            "",
            "## Artifacts",
            "",
            "- `medical_gate_report.json`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Image candidates: {len(images)}")
    print(f"Mask candidates: {len(masks)}")
    print(f"Can run Dice/IoU now: {can_run_dice_iou}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
