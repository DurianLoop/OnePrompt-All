from __future__ import annotations

# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_glas_phase2_gate_d import (
    background_point,
    classify_gate_d,
    connected_component_count,
    mask_bbox,
    mask_centroid,
    select_smoke_cases,
)


class GlasPhase2GateDTests(unittest.TestCase):
    def test_mask_geometry_helpers(self) -> None:
        mask = [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, False, False],
            [False, False, False, True],
        ]

        self.assertEqual(mask_centroid(mask), (2, 2))
        self.assertEqual(mask_bbox(mask), (1, 1, 3, 3))
        self.assertEqual(connected_component_count(mask), 2)
        bg_x, bg_y = background_point(mask)
        self.assertFalse(mask[bg_y][bg_x])

    def test_select_smoke_cases_covers_grades_and_extremes(self) -> None:
        inventory = [
            {"query_case_id": "a", "query_grade": "benign", "component_count": 1, "foreground_area": 10},
            {"query_case_id": "b", "query_grade": "malignant", "component_count": 5, "foreground_area": 40},
            {"query_case_id": "c", "query_grade": "benign", "component_count": 3, "foreground_area": 100},
        ]

        selected = select_smoke_cases(inventory)
        selected_ids = {row["query_case_id"] for row in selected}

        self.assertIn("a", selected_ids)
        self.assertIn("b", selected_ids)
        self.assertIn("c", selected_ids)

    def test_classify_gate_d_requires_privileged_margin(self) -> None:
        rows = [
            {
                "failed": False,
                "privileged_best_dice": 0.75,
                "control_best_dice": 0.10,
                "call_count": 5,
                "prompt_group_count": 5,
            }
        ]

        status, issues = classify_gate_d(rows)

        self.assertEqual(status, "PASS")
        self.assertEqual(issues, [])

    def test_classify_gate_d_fails_when_control_is_too_close(self) -> None:
        rows = [
            {
                "failed": False,
                "privileged_best_dice": 0.30,
                "control_best_dice": 0.29,
                "call_count": 5,
                "prompt_group_count": 5,
            }
        ]

        status, issues = classify_gate_d(rows)

        self.assertEqual(status, "FAIL")
        self.assertTrue(any("controls" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
