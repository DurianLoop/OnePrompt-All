from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TEST_TMP = PROJECT_ROOT / ".test_tmp"

from adapromptset.data.audit import (
    DatasetSpec,
    build_glass_manifest,
    build_monusac_manifest,
    build_monuseg_manifest,
    compute_file_sha256,
    summarize_manifest,
    validate_gate_a,
    write_exposure_log,
    write_manifest_csv,
)


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class AuditManifestTests(unittest.TestCase):
    def temp_root(self) -> Path:
        TEST_TMP.mkdir(parents=True, exist_ok=True)
        root = TEST_TMP / self._testMethodName
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_hashing_and_summary_are_deterministic(self) -> None:
        root = self.temp_root()
        image = _write(root / "train" / "images" / "case_a.png", b"image")
        mask = _write(root / "train" / "masks" / "case_a.png", b"mask")

        rows = build_monuseg_manifest(root)

        self.assertEqual(rows[0]["image_sha256"], compute_file_sha256(image))
        self.assertEqual(rows[0]["mask_sha256"], compute_file_sha256(mask))
        self.assertEqual(summarize_manifest(rows)["split_counts"], {"train": 1})

    def test_monuseg_manifest_requires_paired_masks(self) -> None:
        root = self.temp_root()
        _write(root / "train" / "images" / "case_a.png")

        with self.assertRaisesRegex(ValueError, "missing mask"):
            build_monuseg_manifest(root)

    def test_monusac_manifest_uses_existing_patient_manifest(self) -> None:
        root = self.temp_root()
        _write(root / "train" / "images" / "train_000.png")
        _write(root / "train" / "masks" / "train_000.png")
        manifest = root / "monusac_processed_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "patient"])
            writer.writeheader()
            writer.writerow({"case_id": "train_000", "split": "train", "patient": "p1"})

        rows = build_monusac_manifest(root)

        self.assertEqual(rows[0]["dataset"], "MoNuSAC")
        self.assertEqual(rows[0]["patient_id"], "p1")

    def test_glass_manifest_reads_grade_csv_and_preserves_source_split(self) -> None:
        root = self.temp_root()
        (root / "Grade.csv").write_text(
            "name,patient ID, grade (GlaS)\ntrain_1,patient_a,benign\n",
            encoding="utf-8",
        )
        _write(root / "train_1.bmp")
        _write(root / "train_1_anno.bmp")

        rows = build_glass_manifest(root)

        self.assertEqual(rows[0]["dataset"], "GlaS")
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["patient_id"], "patient_a")
        self.assertEqual(rows[0]["grade"], "benign")

    def test_gate_a_rejects_train_test_patient_overlap(self) -> None:
        rows = [
            {"dataset": "X", "case_id": "a", "split": "train", "patient_id": "p1"},
            {"dataset": "X", "case_id": "b", "split": "test", "patient_id": "p1"},
        ]

        result = validate_gate_a(rows)

        self.assertFalse(result.passed)
        self.assertTrue(any("patient overlap" in issue for issue in result.issues))

    def test_manifest_and_exposure_log_outputs(self) -> None:
        root = self.temp_root()
        rows = [
            {
                "dataset": "MoNuSeg",
                "case_id": "case_a",
                "split": "train",
                "patient_id": "case_a",
                "image_path": "image.png",
                "mask_path": "mask.png",
                "image_sha256": "a",
                "mask_sha256": "b",
            }
        ]
        manifest_path = root / "manifests" / "monuseg.csv"
        log_path = root / "docs" / "data_exposure_log.md"

        write_manifest_csv(manifest_path, rows)
        write_exposure_log(
            log_path,
            [
                DatasetSpec(
                    name="MoNuSeg",
                    role="sanity",
                    root=Path("data/monuseg"),
                    tuning_allowed=False,
                    notes="unit test",
                )
            ],
        )

        self.assertIn("case_a", manifest_path.read_text(encoding="utf-8"))
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("MoNuSeg", log_text)
        self.assertIn("tuning_allowed=False", log_text)

        sidecar = manifest_path.with_suffix(".csv.sha256")
        self.assertTrue(sidecar.is_file())
        summary_path = root / "manifests" / "monuseg.summary.json"
        self.assertEqual(len(json.loads(summary_path.read_text())["split_counts"]), 1)


if __name__ == "__main__":
    unittest.main()
