"""Materialize the pre-registered PanNuke zero-shot diagnostic interface.

This script is data preparation only: it reads no model and calculates no
prediction/metric.  It emits all fold-1 query RGB/union-mask pairs plus the
five fold-2 support pairs chosen before image inspection by SHA-256 order.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pannuke_hf"
OUT = DATA / "materialized_zero_shot_v1"
FOLD1 = DATA / "fold1-00000-of-00001.parquet"
FOLD2 = DATA / "fold2-00000-of-00001.parquet"
SUPPORT_COUNT = 5


def image_bytes(item: dict[str, object]) -> bytes:
    value = item["bytes"]
    if not isinstance(value, bytes):
        raise TypeError("expected embedded image bytes")
    return value


def decode_rgb(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"))


def union_mask(instances: list[dict[str, object]]) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for instance in instances:
        with Image.open(io.BytesIO(image_bytes(instance))) as image:
            value = np.asarray(image.convert("L")) > 0
        merged = value if merged is None else np.logical_or(merged, value)
    return None if merged is None else merged.astype(np.uint8) * 255


def read(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def save(case_id: str, row: dict[str, object], split: str) -> dict[str, object] | None:
    rgb = decode_rgb(image_bytes(row["image"]))
    mask = union_mask(row["instances"])
    if mask is None:
        return None
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"shape mismatch for {case_id}: {rgb.shape} vs {mask.shape}")
    image_path = OUT / "images" / f"{case_id}.png"
    mask_path = OUT / "masks" / f"{case_id}.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return {
        "case_id": case_id, "role": split, "tissue": int(row["tissue"]),
        "instance_count": len(row["instances"]), "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
        "image_sha256": hashlib.sha256(image_bytes(row["image"])).hexdigest(),
        "image_path": f"images/{case_id}.png", "mask_path": f"masks/{case_id}.png",
    }


def main() -> None:
    if not all(path.is_file() for path in (FOLD1, FOLD2)):
        raise FileNotFoundError("both pre-registered PanNuke fold parquet files are required")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True); (OUT / "masks").mkdir(exist_ok=True)
    fold1, fold2 = read(FOLD1), read(FOLD2)
    selected = sorted(
        ((hashlib.sha256(image_bytes(row["image"])).hexdigest(), index, row)
         for index, row in enumerate(fold2) if row["instances"]),
        key=lambda item: item[0],
    )[:SUPPORT_COUNT]
    records = [record for index, row in enumerate(fold1)
               if (record := save(f"fold1_{index:05d}", row, "query")) is not None]
    records.extend(record for _, index, row in selected
                   if (record := save(f"fold2_{index:05d}", row, "support")) is not None)
    with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    supports = [record for record in records if record["role"] == "support"]
    report = {
        "status": "PANNUKE_INTERFACE_READY",
        "protocol": "icassp.pannuke_zero_shot_v1",
        "query_fold": "fold1", "query_input_count": len(fold1),
        "query_count": len([record for record in records if record["role"] == "query"]), "support_fold": "fold2",
        "support_count": len(supports), "support_selection": "ascending SHA-256 of embedded RGB bytes",
        "target": "union of all supplied instance masks", "image_shape": [256, 256, 3],
        "fold1_parquet_sha256": hashlib.sha256(FOLD1.read_bytes()).hexdigest(),
        "fold2_parquet_sha256": hashlib.sha256(FOLD2.read_bytes()).hexdigest(),
        "selected_supports": supports,
    }
    (OUT / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
