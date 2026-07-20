from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_data_profile import export_data_profile_audit


class DataProfileAuditTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_image(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)

    def test_data_profile_records_dimensions_mask_area_and_instances(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_data_profile_audit"
        dataset_root = root / "datasets" / "TinySet"
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        mask = np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 2, 2],
                [3, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        self._write_image(dataset_root / "images" / "case_001.png", image)
        self._write_image(dataset_root / "masks" / "case_001.png", mask)
        self._write_csv(
            root / "manifests" / "tinyset_manifest.csv",
            [
                {
                    "dataset": "TinySet",
                    "case_id": "case_001",
                    "split": "train",
                    "role": "train",
                    "patient_id": "p1",
                    "image_path": "images/case_001.png",
                    "mask_path": "masks/case_001.png",
                }
            ],
        )
        (root / "configs" / "data").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "data" / "tinyset.yaml").write_text(
            f"name: TinySet\nroot: {dataset_root}\n",
            encoding="utf-8",
        )

        outputs = export_data_profile_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["dataset_count"], 1)
        dataset = payload["datasets"][0]
        self.assertEqual(dataset["dataset"], "TinySet")
        self.assertEqual(dataset["case_count"], 1)
        self.assertEqual(dataset["width"]["min"], 4)
        self.assertEqual(dataset["height"]["max"], 3)
        self.assertEqual(dataset["mask_area_pixels"]["mean"], 5.0)
        self.assertEqual(dataset["instance_count"]["mean"], 3.0)
        self.assertEqual(dataset["mask_area_fraction"]["mean"], 5 / 12)
        self.assertEqual(payload["rows"][0]["mask_unique_nonzero_count"], 3)

        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Status: `PASS`", markdown)
        self.assertIn("TinySet", markdown)
        self.assertIn("mask area", markdown)

    def test_data_profile_resolves_manifest_paths_relative_to_dataset_parent(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_data_profile_parent_relative"
        dataset_root = root / "datasets" / "TinySet"
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[0, 4], [0, 0]], dtype=np.uint8)
        self._write_image(dataset_root / "images" / "case_001.png", image)
        self._write_image(dataset_root / "masks" / "case_001.png", mask)
        self._write_csv(
            root / "manifests" / "tinyset_manifest.csv",
            [
                {
                    "dataset": "TinySet",
                    "case_id": "case_001",
                    "split": "train",
                    "role": "train",
                    "patient_id": "p1",
                    "image_path": "TinySet/images/case_001.png",
                    "mask_path": "TinySet/masks/case_001.png",
                }
            ],
        )
        (root / "configs" / "data").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "data" / "tinyset.yaml").write_text(
            f"name: TinySet\nroot: {dataset_root}\n",
            encoding="utf-8",
        )

        outputs = export_data_profile_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["mask_area_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
