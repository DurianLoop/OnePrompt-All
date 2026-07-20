from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDS = [
    "dataset",
    "sample_id",
    "image_path",
    "mask_path",
    "source_folder",
    "extension",
    "width",
    "height",
    "split_guess",
    "is_patch",
    "mask_area_ratio",
]


def normalize_stem(stem: str) -> str:
    stem = re.sub(r"(-checkpoint)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(_mask|mask|label|_label)$", "", stem, flags=re.IGNORECASE)
    return stem


def mask_area_ratio(path: Path) -> float:
    arr = np.array(Image.open(path).convert("L"))
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum()) / float(arr.size)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def audit_monuseg(root: Path) -> list[dict[str, object]]:
    image_files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "image" in str(p.parent).lower()
        and ".ipynb_checkpoints" not in p.parts
    ]
    mask_files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "mask" in str(p.parent).lower()
        and ".ipynb_checkpoints" not in p.parts
    ]

    images_by_stem = {p.stem: p for p in image_files}
    images_by_norm = {normalize_stem(p.stem): p for p in image_files}
    rows: list[dict[str, object]] = []
    seen = set()
    for mask_path in sorted(mask_files):
        sample_id = normalize_stem(mask_path.stem)
        image_path = images_by_stem.get(mask_path.stem) or images_by_norm.get(sample_id)
        if image_path is None:
            continue
        key = (str(image_path), str(mask_path))
        if key in seen:
            continue
        seen.add(key)

        image = Image.open(image_path)
        folder = str(image_path.parent).replace("\\", "/")
        split = "training" if "training" in folder.lower() else "test" if "test" in folder.lower() else "local"
        rows.append(
            {
                "dataset": "MonuSeg",
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "source_folder": str(image_path.parent),
                "extension": image_path.suffix.lower().lstrip("."),
                "width": image.size[0],
                "height": image.size[1],
                "split_guess": split,
                "is_patch": "yes" if image_path.suffix.lower() == ".png" else "unknown_or_original_tif",
                "mask_area_ratio": f"{mask_area_ratio(mask_path):.8f}",
            }
        )
    return rows


def audit_idrid(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_dir in ["DR_Training_Set", "DR_Testing_Set"]:
        image_dir = root / split_dir / "Fundus Images"
        mask_dir = root / split_dir / "Combined Masks"
        if not image_dir.exists() or not mask_dir.exists():
            continue
        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "-checkpoint" not in p.stem
        }
        masks = [
            p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "-checkpoint" not in p.stem
        ]
        split = "training" if "Training" in split_dir else "test"
        for mask_path in sorted(masks):
            image_path = images.get(mask_path.stem)
            if image_path is None:
                continue
            image = Image.open(image_path)
            rows.append(
                {
                    "dataset": "IDRID",
                    "sample_id": mask_path.stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "source_folder": str(image_path.parent),
                    "extension": image_path.suffix.lower().lstrip("."),
                    "width": image.size[0],
                    "height": image.size[1],
                    "split_guess": split,
                    "is_patch": "no",
                    "mask_area_ratio": f"{mask_area_ratio(mask_path):.8f}",
                }
            )
    return rows


def summarize(name: str, rows: list[dict[str, object]]) -> None:
    print(f"{name}_pairs", len(rows))
    print("split", dict(Counter(str(r["split_guess"]) for r in rows)))
    print("extension", dict(Counter(str(r["extension"]) for r in rows)))


def main() -> None:
    out = Path("runs/research_direction/dataset_audit")
    monuseg_rows = audit_monuseg(Path("code/datasets/MonuSeg"))
    idrid_rows = audit_idrid(Path("code/datasets/IDRID"))
    write_rows(out / "all_monu_valid_pairs.csv", monuseg_rows)
    write_rows(out / "all_idrid_valid_pairs.csv", idrid_rows)
    summarize("monuseg", monuseg_rows)
    summarize("idrid", idrid_rows)


if __name__ == "__main__":
    main()
