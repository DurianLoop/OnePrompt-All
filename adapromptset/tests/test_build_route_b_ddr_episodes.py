from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_route_b_ddr_episodes import build_route_b_ddr_episodes  # noqa: E402


FIELDNAMES = [
    "dataset",
    "split",
    "protocol_role",
    "image_id",
    "intent",
    "query_unit_id",
    "image_path",
    "image_sha256",
    "mask_path",
    "mask_sha256",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "mask_foreground_pixels",
    "mask_area_ratio",
    "mask_nonempty",
    "formal_test_locked",
]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(split: str, image_id: str, intent: str) -> dict[str, str]:
    return {
        "dataset": "DDR",
        "split": split,
        "protocol_role": "development_train" if split == "train" else "development_validation",
        "image_id": image_id,
        "intent": intent,
        "query_unit_id": f"DDR:{split}:{image_id}:{intent}",
        "image_path": f"{split}/{image_id}.jpg",
        "image_sha256": f"image-{image_id}",
        "mask_path": f"{split}/{image_id}-{intent}.tif",
        "mask_sha256": f"mask-{image_id}-{intent}",
        "image_width": "8",
        "image_height": "6",
        "mask_width": "8",
        "mask_height": "6",
        "mask_foreground_pixels": "1",
        "mask_area_ratio": "0.1",
        "mask_nonempty": "True",
        "formal_test_locked": "True",
    }


class RouteBDDREpisodeTests(unittest.TestCase):
    def test_builder_uses_train_support_for_valid_and_excludes_train_query(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_ddr_episodes_project"
        if root.exists():
            shutil.rmtree(root)
        try:
            train_rows = [
                _row("train", "train-a", "EX"),
                _row("train", "train-b", "EX"),
                _row("train", "train-c", "EX"),
                _row("train", "train-d", "HE"),
                _row("train", "train-e", "HE"),
            ]
            valid_rows = [_row("valid", "valid-a", "EX"), _row("valid", "valid-b", "HE")]
            _write_manifest(root / "manifests" / "phase4plus" / "ddr_route_b_train_positive_queries.csv", train_rows)
            _write_manifest(root / "manifests" / "phase4plus" / "ddr_route_b_valid_positive_queries.csv", valid_rows)

            summary = build_route_b_ddr_episodes(project_root=root, references_per_query=1)

            self.assertEqual(summary["status"], "PASS_DDR_ROUTE_B_EPISODES")
            self.assertFalse(summary["test_files_read"])
            valid_episode_path = root / "manifests" / "phase4plus" / "ddr_route_b_valid_reference_query_episodes.csv"
            with valid_episode_path.open(newline="", encoding="utf-8") as handle:
                valid_episodes = list(csv.DictReader(handle))
            self.assertTrue(all(row["reference_source_split"] == "train" for row in valid_episodes))
            train_episode_path = root / "manifests" / "phase4plus" / "ddr_route_b_train_reference_query_episodes.csv"
            with train_episode_path.open(newline="", encoding="utf-8") as handle:
                train_episodes = list(csv.DictReader(handle))
            self.assertTrue(all(row["query_image_id"] != row["reference_image_id"] for row in train_episodes))
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
