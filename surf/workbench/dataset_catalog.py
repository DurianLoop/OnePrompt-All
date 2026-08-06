"""Small, editable dataset registry for the interactive workbench.

Entries are deliberately metadata-only: a later curated dataset can be added by
declaring its label, group and image root without changing UI code.  The UI
only renders sources with discovered images, so a missing local checkout never
becomes a misleading "empty" dataset branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetSource:
    label: str
    group: str
    root: Path
    limit: int = 18
    scene: str = "curated image segmentation scene"


def registered_sources(repo_root: Path) -> tuple[DatasetSource, ...]:
    # The research datasets live next to the repository (D:\SAM\dataset),
    # while the application code lives in D:\SAM\surf.  Point directly at
    # their image folders: scanning a dataset root would otherwise pick up
    # ground-truth masks as if they were selectable input images.
    data_root = repo_root.parent / "dataset"
    local_data_root = repo_root / "数据集"

    def first_existing(*paths: Path) -> Path:
        return next((path for path in paths if path.exists()), paths[0])

    return (
        DatasetSource("20张", "OnePrompt-All", data_root / "20张" / "same_class_20" / "images", limit=20, scene="OnePrompt-All natural same-class benchmark"),
        DatasetSource("200张", "OnePrompt-All", data_root / "200张" / "same_class_200" / "images", limit=32, scene="OnePrompt-All extended natural benchmark"),
        DatasetSource("COCO", "Natural images", first_existing(local_data_root / "coco" / "selected_images（coco77张）", data_root / "coco" / "selected_images（coco77张）"), limit=32, scene="COCO natural-image instance scene"),
        DatasetSource("LVIS", "Natural images", first_existing(local_data_root / "LVIS" / "selected_images（LVIS230张）", data_root / "LVIS" / "selected_images（LVIS230张）"), limit=32, scene="LVIS long-tail natural-image scene"),
        DatasetSource("COCO test", "Natural images", repo_root.parent / "external" / "assets" / "coco_test_images", limit=24, scene="COCO quick-test images"),
        DatasetSource("SAM-Med2D demo", "Medical imagery", repo_root.parent / "external" / "SAM-Med2D" / "data_demo" / "images", limit=12, scene="medical multi-modal demo slices"),
        DatasetSource("IDRID preview", "Medical imagery", repo_root.parent / "external" / "FSPL" / "artifacts" / "test-data" / "IDRID" / "DR_Training_Set" / "Fundus Images", limit=12, scene="IDRID retinal-lesion loader preview"),
        DatasetSource("GlaS preview", "Medical imagery", repo_root.parent / "external" / "research_data" / "extracted" / "GlaS" / "Warwick QU Dataset (Released 2016_07_08)", limit=16, scene="GlaS gland-segmentation loader preview"),
        DatasetSource("MedSAM pathology", "Medical imagery", repo_root.parent / "external" / "MedSAM" / "assets" / "pathology.png", limit=1, scene="medical pathology image demo"),
        DatasetSource("MedSAM imaging", "Medical imagery", repo_root.parent / "external" / "MedSAM" / "assets" / "img_demo.png", limit=1, scene="medical imaging demo"),
    )
