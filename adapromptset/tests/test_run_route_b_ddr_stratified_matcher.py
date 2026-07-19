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

from adapromptset.scripts.run_route_b_ddr_stratified_matcher import run_route_b_ddr_stratified_matcher


FIELDNAMES = [
    "split",
    "intent",
    "query_unit_id",
    "query_image_id",
    "query_image_path",
    "query_mask_path",
    "reference_rank",
    "reference_unit_id",
    "reference_image_id",
    "reference_image_path",
    "reference_mask_path",
    "reference_source_split",
]


def _write_episode_csv(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for intent in ("EX", "HE", "MA", "SE"):
            writer.writerow(
                {
                    "split": split,
                    "intent": intent,
                    "query_unit_id": f"DDR:{split}:q-{intent}:{intent}",
                    "query_image_id": f"q-{intent}",
                    "query_image_path": f"{split}/q-{intent}.jpg",
                    "query_mask_path": f"{split}/q-{intent}.tif",
                    "reference_rank": "1",
                    "reference_unit_id": f"DDR:train:r-{intent}:{intent}",
                    "reference_image_id": f"r-{intent}",
                    "reference_image_path": f"train/r-{intent}.jpg",
                    "reference_mask_path": f"train/r-{intent}.tif",
                    "reference_source_split": "train",
                }
            )


class RouteBDDRStratifiedMatcherTests(unittest.TestCase):
    def test_dry_run_selects_one_query_per_split_intent_without_test_access(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_stratified_matcher"
        if root.exists():
            shutil.rmtree(root)
        try:
            _write_episode_csv(root / "manifests" / "phase4plus" / "ddr_route_b_train_reference_query_episodes.csv", "train")
            _write_episode_csv(root / "manifests" / "phase4plus" / "ddr_route_b_valid_reference_query_episodes.csv", "valid")

            payload = run_route_b_ddr_stratified_matcher(project_root=root, dry_run=True)

            self.assertEqual(payload["status"], "DRY_RUN")
            self.assertEqual(payload["selected_episode_count"], 8)
            self.assertFalse(payload["formal_test_access"])
            self.assertFalse(payload["test_files_read"])
            self.assertEqual(payload["issues"], [])
            report = json.loads(
                (root / "reports" / "route_b_ddr_stratified_matcher" / "stratified_matcher_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["selected_counts"]["valid:MA"], 1)
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
