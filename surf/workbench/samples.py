"""Sample and scene registry for the phase-1 reproduction workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from workbench.dataset_catalog import registered_sources


Difficulty = Literal["easy", "hard", "mixed"]


@dataclass(frozen=True, slots=True)
class SampleEntry:
    id: int
    name: str
    path: Path
    dataset: str
    scene: str
    difficulty: Difficulty
    media_type: Literal["image", "video", "frames"] = "image"

    def to_dict(self, url: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset": self.dataset,
            "scene": self.scene,
            "difficulty": self.difficulty,
            "media_type": self.media_type,
            "url": url,
        }


def _first(paths: list[Path], limit: int) -> list[Path]:
    return [p for p in paths if p.exists()][:limit]


def _is_readable_image(path: Path) -> bool:
    """Avoid exposing placeholder, corrupt, or incomplete local image files."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _loader_medical_samples(workspace_root: Path, start_id: int) -> list[SampleEntry]:
    """Expose real images preserved by the established medical data loaders.

    The original loader roots are not always retained after an experiment, but
    the nnU-Net export retains the input pixels plus a source manifest.  Keep
    IDRiD fundus and MoNuSeg nuclei separate in the browser tree.
    """
    root = workspace_root / "runs" / "nnunet" / "nnUNet_raw" / "Dataset902_OCSAMPhase3Stratified"
    manifest = root / "ocsam_split_manifest.json"
    if not manifest.is_file():
        return []
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    labels = {
        "IDRID": ("IDRiD Fundus", "medical retinal fundus image from the established IDRiD loader"),
        "MonuSeg": ("MoNuSeg Nuclei", "medical histology nuclei image from the established MoNuSeg loader"),
    }
    samples: list[SampleEntry] = []
    for source_name in ("IDRID", "MonuSeg"):
        label, scene = labels[source_name]
        for entry in (row for row in entries if row.get("dataset") == source_name):
            path = root / str(entry.get("image", ""))
            if not path.is_file():
                continue
            samples.append(SampleEntry(
                id=start_id + len(samples),
                name=str(entry.get("source_case_id") or path.stem),
                path=path,
                dataset=label,
                scene=scene,
                difficulty="mixed",
            ))
    return samples


def find_image_samples(repo_root: Path, workspace_root: Path) -> list[SampleEntry]:
    samples: list[SampleEntry] = []

    # Curated OnePrompt-All interaction set: exactly the 20 natural same-class
    # scenes used during algorithm development.  It is deliberately separate
    # from future user datasets and does not expose masks to the UI.
    repo_oneprompt = repo_root / "datasets" / "oneprompt_all" / "natural20" / "images"
    oneprompt_images: list[Path] = sorted(repo_oneprompt.glob("*.jpg"))[:20]
    # Legacy fallback only for a checkout created before the curated data was
    # copied into this repository.
    dataset_root = workspace_root / "dataset"
    if not oneprompt_images and dataset_root.exists():
        for candidate in dataset_root.glob("*/same_class_20/images"):
            oneprompt_images = sorted(candidate.glob("*.jpg"))[:20]
            if oneprompt_images:
                break
    # Keep discovery separate from the UI.  Adding a new curated set only
    # requires an entry in dataset_catalog.py, never a frontend rewrite.
    for source_spec in registered_sources(repo_root):
        candidates = [source_spec.root] if source_spec.root.is_file() else sorted(source_spec.root.rglob("*"))
        paths = [
            path for path in candidates
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            and not any(token in path.name.lower() for token in ("mask", "_anno", "_result", "_label"))
            and _is_readable_image(path)
        ][:source_spec.limit]
        for path in paths:
            samples.append(SampleEntry(id=len(samples), name=path.name, path=path, dataset=source_spec.label, scene=source_spec.scene, difficulty="mixed"))

    samples.extend(_loader_medical_samples(workspace_root, len(samples)))

    monuseg = _first(sorted((repo_root / "datasets" / "MonuSeg" / "kmms" / "images").glob("*.png")), 8)
    for idx, path in enumerate(monuseg):
        difficulty: Difficulty = "easy" if idx < 3 else "hard"
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="Medical Nuclei",
                scene="sparse nuclei" if difficulty == "easy" else "dense or touching nuclei",
                difficulty=difficulty,
            )
        )

    idrid = _first(
        sorted((repo_root / "datasets" / "IDRID" / "DR_Training_Set" / "Fundus Images").glob("*.jpg")),
        6,
    )
    for idx, path in enumerate(idrid):
        difficulty = "easy" if idx < 2 else "hard"
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="Retinal lesions",
                scene="large bright lesion" if difficulty == "easy" else "small or low-contrast lesion",
                difficulty=difficulty,
            )
        )

    general = _first(sorted((workspace_root / "sam3" / "assets" / "images").glob("*.jpg")), 4)
    general.extend(_first(sorted((repo_root / "gradio_demo" / "images").glob("*.png")), 4))
    for path in general:
        samples.append(
            SampleEntry(
                id=len(samples),
                name=path.name,
                path=path,
                dataset="Custom",
                scene="open-vocabulary sanity check",
                difficulty="mixed",
            )
        )

    return samples


def find_video_samples(workspace_root: Path) -> list[SampleEntry]:
    paths: list[Path] = []
    paths.extend(sorted((workspace_root / "sam2" / "demo" / "data" / "gallery").glob("*.mp4")))
    bedroom = workspace_root / "sam2" / "notebooks" / "videos" / "bedroom"
    if bedroom.exists():
        paths.append(bedroom)
    paths.extend(sorted((workspace_root / "sam2" / "notebooks" / "videos").glob("*.mp4"))[:2])

    samples: list[SampleEntry] = []
    for idx, path in enumerate([p for p in paths if p.exists()]):
        media_type = "frames" if path.is_dir() else "video"
        samples.append(
            SampleEntry(
                id=idx,
                name=path.name,
                path=path,
                dataset="SAM2",
                scene="simple video propagation" if idx == 0 else "video propagation",
                difficulty="mixed",
                media_type=media_type,
            )
        )
    return samples


def sample_manifest_payload(image_samples: list[SampleEntry], video_samples: list[SampleEntry]) -> dict[str, Any]:
    return {
        "scenes": [
            {
                "dataset": "IDRID",
                "easy": "large bright lesion",
                "hard": "small or low-contrast lesion",
                "prompts": ["box", "point", "text: lesion/red lesion/bright lesion/exudate"],
            },
            {
                "dataset": "MonuSeg",
                "easy": "sparse nuclei",
                "hard": "dense or touching nuclei",
                "prompts": ["box", "point", "text: nucleus/cell nucleus/nuclei", "reference"],
            },
            {
                "dataset": "SAM2 video",
                "easy": "short stable object clip",
                "hard": "occlusion, appearance change, pseudo-video slices",
                "prompts": ["video_point", "video_box"],
            },
        ],
        "images": [sample.to_dict(url=f"/example/{sample.id}") for sample in image_samples],
        "videos": [
            {
                **sample.to_dict(url=f"/video/{sample.id}" if sample.path.is_file() else ""),
                "kind": "frames" if sample.path.is_dir() else "mp4",
            }
            for sample in video_samples
        ],
    }
