from __future__ import annotations

# ruff: noqa: E402
import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_phase3_interpretation import run_audit


class Phase3InterpretationAuditTests(unittest.TestCase):
    def test_audit_relabels_c2_and_preserves_operational_failure(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_phase3_interpretation"
        source = root / "phase2d"
        out = root / "out"
        source.mkdir(parents=True, exist_ok=True)
        rows = []
        morph_rows = []
        for segmenter in ("SAM-B", "SAM-L"):
            for query_index in range(2):
                query_id = f"q{query_index}"
                for family, dice, c1, c2, iou in (
                    ("P2_fixed_multi_centroid_one_group", 0.60, 0.55, 0.56, 0.43),
                    ("P3_component_grouped_centroids", 0.30, 0.50, 0.52, 0.18),
                    ("P4_union_mask_bounding_box", 0.75, 0.70, 0.68, 0.60),
                ):
                    rows.append(
                        {
                            "segmenter": segmenter,
                            "query_case_id": query_id,
                            "query_grade": "benign",
                            "prompt_family": family,
                            "privileged_dice": dice,
                            "privileged_iou": iou,
                            "matched_control_dice": (c1 + c2) / 2.0,
                            "c1_random_dice": c1,
                            "c2_mismatched_dice": c2,
                            "privileged_margin": dice - (c1 + c2) / 2.0,
                            "mask_area_bias": -0.05,
                            "privileged_catastrophic": family == "P3_component_grouped_centroids",
                        }
                    )
                morph_rows.append(
                    {
                        "segmenter": segmenter,
                        "query_case_id": query_id,
                        "prompt_family": "P3_component_grouped_centroids",
                        "foreground_area_ratio": 0.2,
                        "component_count": 4,
                    }
                )

        self._write_csv(source / "query_level_metrics.csv", rows)
        self._write_csv(source / "prompt_family_metrics.csv", [])
        self._write_csv(source / "morphology_analysis.csv", morph_rows)

        exit_code = run_audit(project_root=PROJECT_ROOT, phase2d_dir=source, out_dir=out)

        self.assertEqual(exit_code, 0)
        summary = json.loads((out / "interpretation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PHASE3_INTERPRETATION_AUDIT_COMPLETE")
        self.assertEqual(summary["source_phase2d_state"], "STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED")
        self.assertEqual(summary["interpretation_state"], "PIVOT_GLAS_REGION_PROMPT_REQUIRED")
        self.assertEqual(summary["c2_relabel"], "alternative_reference_control")
        self.assertIn("region prompting remains viable", summary["scientific_correction"])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("placeholder\n", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
