from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.analyze_route_b_ddr_failure_taxonomy import analyze_route_b_ddr_failure_taxonomy


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RouteBDDRFailureTaxonomyTests(unittest.TestCase):
    def test_failure_taxonomy_compares_candidate_to_random_control(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_failure_taxonomy"
        if root.exists():
            shutil.rmtree(root)
        try:
            mask = Image.new("L", (518, 518), 0)
            mask.putpixel((7, 7), 255)
            mask_path = root / "mask.tif"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)
            _write_csv(
                root / "reports" / "route_b_ddr_stratified_matcher" / "selected_episodes.csv",
                [
                    {
                        "split": "valid",
                        "intent": "EX",
                        "query_unit_id": "DDR:valid:q:EX",
                        "query_image_id": "q",
                        "reference_unit_id": "DDR:train:r:EX",
                        "reference_image_id": "r",
                        "reference_source_split": "train",
                        "query_mask_path": str(mask_path),
                    }
                ],
            )
            _write_csv(
                root / "reports" / "route_b_ddr_stratified_matcher" / "query_metrics.csv",
                [
                    {
                        "split": "valid",
                        "intent": "EX",
                        "query_unit_id": "DDR:valid:q:EX",
                        "query_image_id": "q",
                        "point_precision": "0.0",
                        "component_coverage": "0.0",
                        "positive_count": "5",
                        "s_fg_top_score": "0.5",
                        "s_fg_top_margin": "0.001",
                        "s_fg_entropy": "0.999",
                    }
                ],
            )

            summary = analyze_route_b_ddr_failure_taxonomy(project_root=root)

            self.assertEqual(summary["status"], "PASS_ROUTE_B_DDR_FAILURE_TAXONOMY")
            self.assertEqual(summary["n"], 1)
            self.assertFalse(summary["formal_test_access"])
            self.assertIn("no_candidate_hit", summary["failure_flag_counts"])
            self.assertIn("diffuse_matching_map", summary["failure_flag_counts"])
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
