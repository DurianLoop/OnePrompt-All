from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_route_b_ddr_matcher_smoke import run_route_b_ddr_matcher_smoke


class RouteBDDRMatcherSmokeTests(unittest.TestCase):
    def test_dry_run_writes_boundary_payload_without_model_inference(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_matcher_smoke"
        if root.exists():
            shutil.rmtree(root)
        try:
            episodes = root / "manifests" / "phase4plus" / "ddr_route_b_valid_reference_query_episodes.csv"
            episodes.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "split",
                "intent",
                "query_unit_id",
                "query_image_id",
                "query_image_path",
                "query_mask_path",
                "query_mask_sha256",
                "reference_unit_id",
                "reference_image_id",
                "reference_image_path",
                "reference_mask_path",
                "reference_mask_sha256",
                "reference_source_split",
            ]
            with episodes.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "valid",
                        "intent": "EX",
                        "query_unit_id": "DDR:valid:q:EX",
                        "query_image_id": "q",
                        "query_image_path": "q.jpg",
                        "query_mask_path": "q.tif",
                        "query_mask_sha256": "mask-q",
                        "reference_unit_id": "DDR:train:r:EX",
                        "reference_image_id": "r",
                        "reference_image_path": "r.jpg",
                        "reference_mask_path": "r.tif",
                        "reference_mask_sha256": "mask-r",
                        "reference_source_split": "train",
                    }
                )

            payload = run_route_b_ddr_matcher_smoke(project_root=root, dry_run=True)

            self.assertEqual(payload["status"], "DRY_RUN")
            self.assertFalse(payload["formal_test_access"])
            self.assertFalse(payload["test_files_read"])
            self.assertFalse(payload["uses_query_gt_for_generation"])
            report = json.loads(
                (root / "reports" / "route_b_ddr_development_smoke" / "route_b_ddr_matcher_smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["reference_source_split"], "train")
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
