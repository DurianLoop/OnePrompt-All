from __future__ import annotations

# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_glas_phase3_region_splits import build_outputs


class BuildGlasPhase3RegionSplitsTests(unittest.TestCase):
    def test_build_outputs_uses_only_utility_train_pool(self) -> None:
        out_dir = PROJECT_ROOT / "tmp" / "test_phase3_region_split" / "manifests"
        report_dir = PROJECT_ROOT / "tmp" / "test_phase3_region_split" / "reports"

        summary = build_outputs(out_dir=out_dir, report_dir=report_dir)

        self.assertEqual(summary["status"], "PASS_PHASE3_GLAS_REGION_40_15_SPLIT")
        self.assertEqual(summary["counts"]["region_calibration"], 40)
        self.assertEqual(summary["counts"]["region_selection"], 15)
        self.assertEqual(summary["counts"]["references"], 165)
        self.assertFalse(summary["testA_testB_access"])
        self.assertTrue((out_dir / "glas_region_calibration_40.csv").is_file())
        self.assertTrue((out_dir / "glas_region_selection_15.csv").is_file())
        self.assertTrue((out_dir / "glas_region_references.csv").is_file())


if __name__ == "__main__":
    unittest.main()
