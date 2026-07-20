from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "code" / "datasets" / "MonuSeg" / "official"
RAW_ROOT = OFFICIAL_ROOT / "raw"
OUT_ROOT = OFFICIAL_ROOT
MANIFEST_PATH = (
    ROOT
    / "runs"
    / "research_direction"
    / "dataset_audit"
    / "monuseg_official_manifest.csv"
)


@dataclass(frozen=True)
class Sample:
    split: str
    sample_id: str
    image_path: Path
    annotation_path: Path
    source_zip: str


def is_real_file(path: Path) -> bool:
    parts = set(path.parts)
    return "__MACOSX" not in parts and not path.name.startswith("._")


def collect_by_stem(base: Path, suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in base.rglob(f"*{suffix}"):
        if not is_real_file(path):
            continue
        files[path.stem] = path
    return files


def collect_samples(split: str, raw_split_dir: Path, source_zip: str) -> list[Sample]:
    images = collect_by_stem(raw_split_dir, ".tif")
    annotations = collect_by_stem(raw_split_dir, ".xml")
    samples: list[Sample] = []
    for sample_id in sorted(set(images) & set(annotations)):
        samples.append(
            Sample(
                split=split,
                sample_id=sample_id,
                image_path=images[sample_id],
                annotation_path=annotations[sample_id],
                source_zip=source_zip,
            )
        )
    missing_images = sorted(set(annotations) - set(images))
    missing_annotations = sorted(set(images) - set(annotations))
    if missing_images or missing_annotations:
        raise RuntimeError(
            f"{split}: unmatched files, "
            f"missing_images={missing_images}, missing_annotations={missing_annotations}"
        )
    return samples


def rasterize_xml(annotation_path: Path, size: tuple[int, int]) -> tuple[Image.Image, int]:
    root = ET.parse(annotation_path).getroot()
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    region_count = 0

    for region in root.findall(".//Region"):
        points: list[tuple[float, float]] = []
        for vertex in region.findall(".//Vertex"):
            try:
                points.append((float(vertex.attrib["X"]), float(vertex.attrib["Y"])))
            except KeyError as exc:
                raise RuntimeError(f"Malformed vertex in {annotation_path}: {vertex.attrib}") from exc
        if len(points) >= 3:
            draw.polygon(points, outline=255, fill=255)
            region_count += 1

    return mask, region_count


def prepare_split(samples: list[Sample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        image_out_dir = OUT_ROOT / sample.split / "images"
        mask_out_dir = OUT_ROOT / sample.split / "masks"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)

        image_out = image_out_dir / f"{sample.sample_id}.tif"
        mask_out = mask_out_dir / f"{sample.sample_id}.png"

        with Image.open(sample.image_path) as image:
            width, height = image.size
            mask, region_count = rasterize_xml(sample.annotation_path, image.size)
            nonzero = mask.point(lambda px: 1 if px else 0)
            mask_area = sum(nonzero.getdata())
            mask_area_ratio = mask_area / float(width * height)
            mask.save(mask_out)

        shutil.copy2(sample.image_path, image_out)

        rows.append(
            {
                "dataset": "MoNuSeg",
                "split": sample.split,
                "sample_id": sample.sample_id,
                "image_path": str(image_out.relative_to(ROOT)),
                "annotation_path": str(sample.annotation_path.relative_to(ROOT)),
                "mask_path": str(mask_out.relative_to(ROOT)),
                "width": width,
                "height": height,
                "nuclei_region_count": region_count,
                "mask_area_ratio": f"{mask_area_ratio:.8f}",
                "source_zip": sample.source_zip,
                "status": "ok" if region_count > 0 and mask_area > 0 else "empty_mask",
            }
        )
    return rows


def main() -> None:
    train_samples = collect_samples(
        "train",
        RAW_ROOT / "train",
        "monuseg_2018_training_data.zip",
    )
    test_samples = collect_samples(
        "test",
        RAW_ROOT / "test",
        "monuseg_2018_testing_data.zip",
    )
    rows = prepare_split(train_samples) + prepare_split(test_samples)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_split = {
        "train": len(train_samples),
        "test": len(test_samples),
    }
    print(f"wrote {MANIFEST_PATH}")
    print(f"prepared splits: {by_split}")
    print("note: hidden macOS resource-fork files were ignored")


if __name__ == "__main__":
    main()
