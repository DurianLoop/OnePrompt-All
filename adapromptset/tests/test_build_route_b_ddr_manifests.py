from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_route_b_ddr_manifests import (  # noqa: E402
    LESION_INTENTS,
    build_route_b_ddr_manifests,
)


class RouteBDDRManifestTests(unittest.TestCase):
    def _make_ddr_tree(self, root: Path) -> None:
        cases = {
            "train": ["case-train-a", "case-train-b"],
            "valid": ["case-valid-a"],
        }
        for split, case_ids in cases.items():
            (root / split / "image").mkdir(parents=True, exist_ok=True)
            for intent in LESION_INTENTS:
                (root / split / "label" / intent).mkdir(parents=True, exist_ok=True)
            for case_id in case_ids:
                Image.new("RGB", (8, 6), color=(10, 20, 30)).save(root / split / "image" / f"{case_id}.jpg")
                for idx, intent in enumerate(LESION_INTENTS):
                    mask = Image.new("L", (8, 6), color=0)
                    if idx != 3 or case_id.endswith("a"):
                        mask.putpixel((idx + 1, idx + 1), 255)
                    mask.save(root / split / "label" / intent / f"{case_id}.tif")
        (root / "test" / "image").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(1, 2, 3)).save(root / "test" / "image" / "must-not-read.jpg")

    def test_builder_writes_train_valid_positive_query_manifests(self) -> None:
        data_root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_data"
        project_root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_project"
        for path in [data_root, project_root]:
            if path.exists():
                shutil.rmtree(path)
        try:
            self._make_ddr_tree(data_root)

            summary = build_route_b_ddr_manifests(project_root=project_root, data_root=data_root)

            self.assertEqual(summary["status"], "PASS_DDR_ROUTE_B_DEV_MANIFEST")
            self.assertFalse(summary["formal_test_access"])
            self.assertFalse(summary["test_files_read"])
            self.assertEqual(summary["split_summaries"]["train"]["image_count"], 2)
            self.assertEqual(summary["split_summaries"]["valid"]["image_count"], 1)
            positive_path = project_root / "manifests" / "phase4plus" / "ddr_route_b_train_positive_queries.csv"
            with positive_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(row["mask_nonempty"] == "True" for row in rows))
            report = json.loads(
                (project_root / "reports" / "route_b_ddr_manifest" / "ddr_route_b_manifest_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("train_positive_queries", report["outputs"])
        finally:
            for path in [data_root, project_root]:
                if path.exists():
                    shutil.rmtree(path)

    def test_builder_rejects_test_split(self) -> None:
        data_root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_reject_data"
        project_root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_reject_project"
        for path in [data_root, project_root]:
            if path.exists():
                shutil.rmtree(path)
        try:
            self._make_ddr_tree(data_root)

            with self.assertRaises(ValueError):
                build_route_b_ddr_manifests(project_root=project_root, data_root=data_root, splits=("test",))
        finally:
            for path in [data_root, project_root]:
                if path.exists():
                    shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
