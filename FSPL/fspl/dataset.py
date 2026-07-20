from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class Episode:
    reference: CaseRecord
    query: CaseRecord


def discover_cases(root: Path, split: str) -> list[CaseRecord]:
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError(f"expected {split!r} images and masks directories below {root}")

    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    images = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.stem,
    )
    records: list[CaseRecord] = []
    for image_path in images:
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            raise ValueError(f"missing mask for image {image_path.name}")
        records.append(CaseRecord(image_path.stem, image_path, mask_path, split))
    if not records:
        raise ValueError(f"no images found in {image_dir}")
    return records


def discover_idrid_cases(root: Path, split: str) -> list[CaseRecord]:
    """Discover IDRiD's native train/test image-mask directory layout.

    Local IDRiD directories can contain auxiliary image outputs without a
    corresponding combined mask.  Only stem-matched image/mask pairs define
    evaluable cases.
    """
    partition = {"train": "DR_Training_Set", "test": "DR_Testing_Set"}.get(split)
    if partition is None:
        raise ValueError("IDRiD split must be 'train' or 'test'")
    image_dir = root / partition / "Fundus Images"
    mask_dir = root / partition / "Combined Masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError(f"expected IDRiD image and combined-mask directories below {root}")

    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    records = [
        CaseRecord(image_path.stem, image_path, masks[image_path.stem], split)
        for image_path in sorted(image_dir.iterdir(), key=lambda path: path.stem)
        if image_path.is_file()
        and image_path.suffix.lower() in IMAGE_SUFFIXES
        and image_path.stem in masks
    ]
    if not records:
        raise ValueError(f"no paired IDRiD images found in {image_dir}")
    return records


def build_smoke_episode(cases: Sequence[CaseRecord]) -> Episode:
    ordered = sorted(cases, key=lambda case: case.case_id)
    if len(ordered) < 2:
        raise ValueError("a smoke episode requires at least two cases")
    return Episode(reference=ordered[0], query=ordered[1])


def write_audit_csv(cases: Sequence[CaseRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "image_path", "mask_path"])
        writer.writeheader()
        for case in sorted(cases, key=lambda row: (row.split, row.case_id)):
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "image_path": str(case.image_path.resolve()),
                    "mask_path": str(case.mask_path.resolve()),
                }
            )
