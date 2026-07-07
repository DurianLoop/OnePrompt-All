from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = WORKSPACE_ROOT / "runs" / "phase3_full" / "manifests" / "phase3_100_cases.csv"
DEFAULT_OUT = WORKSPACE_ROOT / "runs" / "nnunet" / "nnUNet_raw" / "Dataset901_OCSAMPhase3"


def read_manifest(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def convert_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("L")
        img.save(dst)


def convert_mask(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "L":
            img = img.convert("L")
        img = img.point(lambda value: 1 if value > 0 else 0)
        img.save(dst)


def split_rows(rows: list[dict[str, str]], train_ratio: float, split_mode: str, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if split_mode == "ordered":
        n_train = int(round(len(rows) * train_ratio))
        return rows[:n_train], rows[n_train:]

    if split_mode != "stratified":
        raise ValueError(f"Unknown split mode: {split_mode}")

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for dataset in sorted(grouped):
        dataset_rows = list(grouped[dataset])
        rng.shuffle(dataset_rows)
        n_train = int(round(len(dataset_rows) * train_ratio))
        train_rows.extend(dataset_rows[:n_train])
        test_rows.extend(dataset_rows[n_train:])
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 2D nnU-Net dataset from the Phase3 OCSAM manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-mode", choices=("ordered", "stratified"), default="ordered")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-name", default="OCSAMPhase3")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = split_rows(rows, args.train_ratio, args.split_mode, args.seed)
    split_records = []
    ordered_rows = [(row, "train") for row in train_rows] + [(row, "test") for row in test_rows]
    for idx, (row, split) in enumerate(ordered_rows):
        case = f"OCSAM_{idx:04d}"
        is_train = split == "train"
        image_dir = "imagesTr" if is_train else "imagesTs"
        label_dir = "labelsTr" if is_train else "labelsTs"
        convert_image(resolve(row["image_path"]), out / image_dir / f"{case}_0000.png")
        convert_mask(resolve(row["mask_path"]), out / label_dir / f"{case}.png")
        split_records.append(
            {
                "case_id": case,
                "source_case_id": row["case_id"],
                "dataset": row["dataset"],
                "split": split,
                "image": f"{image_dir}/{case}_0000.png",
                "label": f"{label_dir}/{case}.png",
            }
        )

    dataset_json = {
        "channel_names": {"0": "gray"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_rows),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": args.dataset_name,
        "description": "IDRID + MonuSeg 2D binary segmentation subset for OCSAM Phase3 supervised nnU-Net baseline.",
        "reference": "OCSAM local Phase3 manifest",
        "licence": "research-only, follows source datasets",
        "release": "0.1",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "ocsam_split_manifest.json").write_text(json.dumps(split_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"train={len(train_rows)} test={len(test_rows)} split_mode={args.split_mode}")


if __name__ == "__main__":
    main()
