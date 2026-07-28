from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_route_b_benchmark_freeze import audit_route_b_benchmark_freeze  # noqa: E402


class RouteBBenchmarkFreezeAuditTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _make_project(self, root: Path, *, allow_formal_test: bool = False) -> None:
        self._write_json(
            root / "reports" / "route_b_ddr_manifest" / "ddr_route_b_manifest_report.json",
            {
                "status": "PASS_DDR_ROUTE_B_DEV_MANIFEST",
                "formal_test_access": False,
                "test_files_read": False,
            },
        )
        self._write_json(
            root / "configs" / "experiments" / "route_b_promptability_benchmark_v1.json",
            {
                "formal_test_access_allowed": allow_formal_test,
                "row_families": [
                    "automatic",
                    "matched_control",
                    "privileged_upper_bound",
                    "direct_matching",
                    "diagnostic_baseline",
                ],
                "prompt_representations": [
                    "point",
                    "multi_point",
                    "grouped_point",
                    "box",
                    "multi_box",
                    "hard_mask",
                    "soft_mask",
                    "direct_matching_mask",
                    "privileged_gt_point",
                    "privileged_gt_box",
                    "privileged_gt_mask",
                ],
                "controls": [
                    "spatial_random_point",
                    "size_matched_random_box",
                    "full_image_box",
                    "alternative_reference_control",
                    "strongest_fixed_representation",
                    "raw_matching_baseline",
                    "privileged_gt_upper_bound",
                ],
                "metrics": [
                    {"name": "promptability_gap"},
                    {"name": "localization_advantage"},
                    {"name": "interface_gain"},
                    {"name": "oracle_adaptation_value"},
                    {"name": "promptability_reliability"},
                ],
                "output_schema": {
                    "query_level_required_columns": [
                        "dataset",
                        "split",
                        "image_id",
                        "intent",
                        "representation",
                        "row_family",
                        "control_family",
                        "matcher",
                        "segmenter",
                        "dice",
                        "iou",
                        "sam_calls",
                        "runtime_seconds",
                        "valid_automatic_row",
                        "uses_query_gt_for_generation",
                        "formal_test_row",
                    ]
                },
            },
        )

    def test_audit_passes_complete_freeze(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_freeze_project"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_project(root)

            summary = audit_route_b_benchmark_freeze(project_root=root)

            self.assertEqual(summary["status"], "PASS_ROUTE_B_B1_PREREQ_FREEZE")
            self.assertEqual(summary["issues"], [])
            self.assertEqual(summary["authorizes"], "development_only_evidence_generation")
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_audit_fails_if_config_allows_formal_test(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "route_b_freeze_formal_fail"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_project(root, allow_formal_test=True)

            summary = audit_route_b_benchmark_freeze(project_root=root)

            self.assertEqual(summary["status"], "FAIL_ROUTE_B_B1_PREREQ_FREEZE")
            self.assertIn("config_allows_formal_test", summary["issues"])
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
