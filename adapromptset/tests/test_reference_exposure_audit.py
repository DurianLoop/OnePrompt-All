from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_reference_exposure import export_reference_exposure_audit


class ReferenceExposureAuditTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _seed_episode_dir(self, root: Path, dataset: str, *, leaking: bool = False) -> Path:
        slug = dataset.lower()
        episode_dir = root / "manifests" / "episodes" / slug
        self._write_csv(
            episode_dir / "case_assignments.csv",
            [
                {
                    "dataset": dataset,
                    "case_id": "train_001",
                    "split": "train",
                    "role": "train",
                    "patient_id": "patient_train_1",
                    "image_path": "processed/train/images/train_001.png",
                    "mask_path": "processed/train/masks/train_001.png",
                },
                {
                    "dataset": dataset,
                    "case_id": "test_001",
                    "split": "test",
                    "role": "test",
                    "patient_id": "patient_test_1",
                    "image_path": "processed/test/images/test_001.png",
                    "mask_path": "processed/test/masks/test_001.png",
                },
            ],
        )
        support_case = "test_001" if leaking else "train_001"
        support_role = "test" if leaking else "train"
        support_path = "processed/test/images/test_001.png" if leaking else "processed/train/images/train_001.png"
        self._write_csv(
            episode_dir / "episodes_test.csv",
            [
                {
                    "episode_id": f"{slug}_test_episode",
                    "dataset": dataset,
                    "role": "test",
                    "support_role": support_role,
                    "support_case_id": support_case,
                    "support_patient": "patient_test_1" if leaking else "patient_train_1",
                    "support_image_path": support_path,
                    "support_mask_path": support_path.replace("images", "masks"),
                    "query_case_id": "test_001",
                    "query_patient": "patient_test_1",
                    "query_source_split": "test",
                    "query_image_path": "processed/test/images/test_001.png",
                    "query_mask_path": "processed/test/masks/test_001.png",
                }
            ],
        )
        for role in ("train", "val"):
            self._write_csv(
                episode_dir / f"episodes_{role}.csv",
                [
                    {
                        "episode_id": f"{slug}_{role}_episode",
                        "dataset": dataset,
                        "role": role,
                        "support_role": "train",
                        "support_case_id": "train_001",
                        "support_patient": "patient_train_1",
                        "support_image_path": "processed/train/images/train_001.png",
                        "support_mask_path": "processed/train/masks/train_001.png",
                        "query_case_id": "train_001",
                        "query_patient": "patient_train_1",
                        "query_source_split": "train",
                        "query_image_path": "processed/train/images/train_001.png",
                        "query_mask_path": "processed/train/masks/train_001.png",
                    }
                ],
            )
        return episode_dir

    def test_reference_exposure_audit_passes_when_supports_are_train_only(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_reference_exposure_pass"
        self._seed_episode_dir(root, "MoNuSAC", leaking=False)

        outputs = export_reference_exposure_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["total_issues"], 0)
        self.assertTrue(payload["checks"]["reference_pool_train_only"])
        self.assertTrue(payload["checks"]["reference_test_leakage_zero"])
        self.assertEqual(payload["datasets"][0]["dataset"], "MoNuSAC")

        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Status: `PASS`", markdown)
        self.assertIn("reference-test leakage", markdown)

    def test_reference_exposure_audit_fails_when_support_comes_from_test(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_reference_exposure_fail"
        self._seed_episode_dir(root, "MoNuSAC", leaking=True)

        outputs = export_reference_exposure_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FAIL")
        reasons = {issue["reason"] for issue in payload["issues"]}
        self.assertIn("support_role_not_train", reasons)
        self.assertIn("support_case_from_test_role", reasons)
        self.assertIn("support_path_points_to_test_split", reasons)
        self.assertFalse(payload["checks"]["reference_pool_train_only"])
        self.assertFalse(payload["checks"]["reference_test_leakage_zero"])


if __name__ == "__main__":
    unittest.main()
