from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def decode_mask(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(payload)).convert("L")) > 0


def union_masks(instance_bytes: Iterable[bytes], categories: Iterable[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    all_mask: np.ndarray | None = None
    class_masks: dict[int, np.ndarray] = {}
    for payload, category in zip(instance_bytes, categories):
        mask = decode_mask(payload)
        if all_mask is None:
            all_mask = np.zeros_like(mask, dtype=bool)
        all_mask |= mask
        class_masks.setdefault(int(category), np.zeros_like(mask, dtype=bool))
        class_masks[int(category)] |= mask
    if all_mask is None:
        raise ValueError("A MoNuSAC patch has no annotated instances")
    return all_mask, class_masks


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return (ROOT / path).resolve() if not path.is_absolute() else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MoNuSAC parquet patches into image/mask files for prompt-policy experiments.")
    parser.add_argument("--data-dir", default="code/datasets/MoNuSAC/hf_raw/data")
    parser.add_argument("--out-dir", default="code/datasets/MoNuSAC/processed")
    args = parser.parse_args()
    data_dir, out_dir = resolve(args.data_dir), resolve(args.out_dir)
    manifest: list[dict[str, object]] = []
    for split in ("train", "test"):
        frame = pd.read_parquet(data_dir / f"{split}-00000-of-00001.parquet")
        for index, row in frame.iterrows():
            case_id = f"{split}_{index:03d}"
            image_dir, mask_dir, class_root = (out_dir / split / "images", out_dir / split / "masks", out_dir / split / "class_masks")
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(image_dir / f"{case_id}.png")
            all_mask, class_masks = union_masks((item["bytes"] for item in row["instances"]), row["categories"])
            Image.fromarray((all_mask.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{case_id}.png")
            for category, mask in class_masks.items():
                category_dir = class_root / f"class_{category}"
                category_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(category_dir / f"{case_id}.png")
            manifest.append({"case_id": case_id, "split": split, "patient": row["patient"], "tissue": int(row["tissue"]), "instance_count": len(row["instances"]), "classes_present": ";".join(str(k) for k in sorted(class_masks))})
    with (out_dir / "monusac_processed_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} MoNuSAC patches to {out_dir}")


if __name__ == "__main__":
    main()
