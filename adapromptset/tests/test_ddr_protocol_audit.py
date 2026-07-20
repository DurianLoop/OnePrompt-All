from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_ddr_protocol import (
    LESION_INTENTS,
    analyze_ddr_protocol,
    write_ddr_protocol_audit,
)


class DDRProtocolAuditTests(unittest.TestCase):
    def _make_ddr_tree(self, root: Path) -> None:
        for split, case_ids in {
            "train": ["001-a", "001-b"],
            "valid": ["002-a"],
            "test": ["003-a"],
        }.items():
            (root / split / "image").mkdir(parents=True, exist_ok=True)
            for intent in LESION_INTENTS:
                (root / split / "label" / intent).mkdir(parents=True, exist_ok=True)
            for case_id in case_ids:
                (root / split / "image" / f"{case_id}.jpg").write_bytes(b"image")
                for intent in LESION_INTENTS:
                    (root / split / "label" / intent / f"{case_id}.tif").write_bytes(b"mask")

    def test_analyze_ddr_protocol_records_stage1_union_and_stage2_intents(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "ddr_protocol_tree"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_ddr_tree(root)

            summary = analyze_ddr_protocol(root)

            self.assertEqual(summary["status"], "PASS_PROTOCOL_DEFINED")
            self.assertEqual(summary["stage1_union"]["allowed_use"], "infrastructure_only")
            self.assertEqual(summary["stage1_union"]["mask_rule"], "binary_union_of_EX_HE_MA_SE")
            self.assertEqual(summary["stage2_four_intents"]["intents"], ["EX", "HE", "MA", "SE"])
            self.assertEqual(summary["formal_policy_training_allowed"], False)
            self.assertEqual(summary["splits"]["train"]["image_count"], 2)
            self.assertEqual(summary["splits"]["train"]["complete_case_count"], 2)
            self.assertEqual(summary["total_missing_label_count"], 0)
            self.assertIn("positive-query protocol", summary["stage2_four_intents"]["query_rule"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_analyze_ddr_protocol_reports_missing_labels(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "ddr_protocol_missing"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_ddr_tree(root)
            (root / "valid" / "label" / "MA" / "002-a.tif").unlink()

            summary = analyze_ddr_protocol(root)

            self.assertEqual(summary["status"], "FAIL_INCOMPLETE_LABELS")
            self.assertEqual(summary["splits"]["valid"]["missing_label_count"], 1)
            self.assertEqual(summary["total_missing_label_count"], 1)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_writer_emits_report_files_under_project_reports(self) -> None:
        data_root = PROJECT_ROOT / ".test_tmp" / "ddr_protocol_writer_data"
        project_root = PROJECT_ROOT / ".test_tmp" / "ddr_protocol_writer_project"
        for path in [data_root, project_root]:
            if path.exists():
                shutil.rmtree(path)
        try:
            self._make_ddr_tree(data_root)

            summary = write_ddr_protocol_audit(project_root=project_root, data_root=data_root)

            self.assertEqual(summary["status"], "PASS_PROTOCOL_DEFINED")
            payload = json.loads((project_root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.json").read_text(encoding="utf-8"))
            text = (project_root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.md").read_text(encoding="utf-8")
            self.assertEqual(payload["stage2_four_intents"]["intents"], ["EX", "HE", "MA", "SE"])
            self.assertIn("Stage 1", text)
            self.assertIn("binary_union_of_EX_HE_MA_SE", text)
            self.assertIn("formal policy training remains prohibited", text)
        finally:
            for path in [data_root, project_root]:
                if path.exists():
                    shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
