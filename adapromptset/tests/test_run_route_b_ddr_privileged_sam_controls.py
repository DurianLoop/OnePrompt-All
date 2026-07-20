from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_route_b_ddr_privileged_sam_controls import (
    foreground_box,
    foreground_centroid_point,
    run_route_b_ddr_privileged_sam_controls,
)


def _write_selected(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RouteBDDRPrivilegedSAMControlsTests(unittest.TestCase):
    def test_geometry_uses_foreground_pixel_and_tight_box(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True
        mask[5, 6] = True

        point = foreground_centroid_point(mask)
        box = foreground_box(mask)

        self.assertIn(point, {(1, 1), (6, 5)})
        self.assertEqual(box, (1, 1, 6, 5))

    def test_dry_run_rejects_missing_query_image_path_without_loading_sam(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_privileged_sam_controls"
        if root.exists():
            shutil.rmtree(root)
        try:
            selected = root / "reports" / "route_b_ddr_stratified_matcher" / "selected_episodes.csv"
            _write_selected(
                selected,
                [
                    {
                        "split": "valid",
                        "intent": "EX",
                        "query_unit_id": "DDR:valid:q:EX",
                        "query_image_id": "q",
                        "query_mask_path": "valid/q.tif",
                        "reference_unit_id": "DDR:train:r:EX",
                        "reference_image_id": "r",
                        "reference_source_split": "train",
                    }
                ],
            )

            payload = run_route_b_ddr_privileged_sam_controls(project_root=root, dry_run=True)

            self.assertEqual(payload["status"], "DRY_RUN")
            self.assertFalse(payload["formal_test_access"])
            self.assertTrue(payload["uses_query_gt_for_generation"])
            self.assertEqual(payload["issues"][0]["issue"], "missing_query_image_path")
            report = json.loads(
                (root / "reports" / "route_b_ddr_privileged_sam_controls" / "privileged_sam_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["claim_boundary"], "privileged_prompt_interface_controls_not_automatic_method_result")
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
