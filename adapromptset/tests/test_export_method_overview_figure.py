from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.export_method_overview_figure import write_method_overview_figure  # noqa: E402


class ExportMethodOverviewFigureTests(unittest.TestCase):
    def test_write_method_overview_figure_records_frozen_learned_train_only_paths(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_export_method_overview_figure"

        outputs = write_method_overview_figure(root)

        for path in outputs.values():
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0)
        tex = outputs["tex"].read_text(encoding="utf-8")
        self.assertIn("Frozen matcher", tex)
        self.assertIn("Frozen segmenter", tex)
        self.assertIn("Train/validation only", tex)
        self.assertIn("Test-time safe", tex)
        self.assertIn("Policy is prohibited after Gate F", tex)
        self.assertIn(r"\begin{tikzpicture}", tex)
        self.assertNotIn("SIMULATED PLACEHOLDER", tex)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS_METHOD_OVERVIEW_FIGURE")
        self.assertEqual(payload["claim_boundary"], "oracle_analysis_only")
        self.assertTrue(payload["checks"]["frozen_matcher_present"])
        self.assertTrue(payload["checks"]["frozen_segmenter_present"])
        self.assertTrue(payload["checks"]["train_only_utility_labels_present"])
        self.assertTrue(payload["checks"]["test_time_safe_path_present"])


if __name__ == "__main__":
    unittest.main()
