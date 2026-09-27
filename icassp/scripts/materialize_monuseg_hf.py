"""Materialize the public Hugging Face MoNuSeg release for the FoB protocol.

The release stores each image as bytes and each instance mask independently in
Parquet. This script deterministically converts it into RGB PNG images and a
binary union mask per patient, without mixing train/test rows or using labels
outside the downloaded split.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "monuseg_hf"
PARQUET = DATA / "train-00000-of-00001.parquet"
OUT = DATA / "materialized_train"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blob(record: object) -> bytes:
    if isinstance(record, dict):
        return bytes(record["bytes"])
    value = record.item() if hasattr(record, "item") else record
    if isinstance(value, dict):
        return bytes(value["bytes"])
    raise TypeError(f"unexpected image record type: {type(record)!r}")


def main() -> None:
    if not PARQUET.is_file():
        raise FileNotFoundError(PARQUET)
    frame = pd.read_parquet(PARQUET)
    if len(frame) != 37:
        raise RuntimeError(f"expected 37 train rows, found {len(frame)}")
    if frame["patient"].nunique() != len(frame):
        raise RuntimeError("patient IDs are not unique")

    images = OUT / "images"
    masks = OUT / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row in frame.sort_values("patient").iterrows():
        patient = str(row["patient"])
        instances = row["instances"]
        image_path = images / f"{patient}.png"
        mask_path = masks / f"{patient}.png"
        if not (image_path.is_file() and mask_path.is_file()):
            image = Image.open(io.BytesIO(blob(row["image"]))).convert("RGB")
            union = np.zeros((image.height, image.width), dtype=np.uint8)
            for record in instances.tolist() if hasattr(instances, "tolist") else list(instances):
                instance = Image.open(io.BytesIO(blob(record))).convert("L")
                if instance.size != image.size:
                    raise RuntimeError(f"size mismatch for {patient}: {instance.size} vs {image.size}")
                union |= np.asarray(instance, dtype=np.uint8) > 0
            image.save(image_path, optimize=True)
            Image.fromarray((union * 255).astype(np.uint8)).save(mask_path, optimize=True)
        with Image.open(image_path) as saved_image, Image.open(mask_path) as saved_mask:
            width, height = saved_image.size
            foreground_fraction = float((np.asarray(saved_mask) > 0).mean())
        rows.append(
            {
                "patient": patient,
                "tissue": int(row["tissue"]),
                "image": image_path.relative_to(DATA).as_posix(),
                "mask": mask_path.relative_to(DATA).as_posix(),
                "width": width,
                "height": height,
                "instance_count": len(instances),
                "union_foreground_fraction": f"{foreground_fraction:.8f}",
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
            }
        )
    manifest = OUT / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"materialized {len(rows)} patients to {OUT}")
    print(f"source_sha256={sha256(PARQUET)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
