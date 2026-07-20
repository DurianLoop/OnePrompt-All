from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.reporting.related_work import (  # noqa: E402
    build_related_work_summary,
    write_related_work_assets,
)


class RelatedWorkAssetsTests(unittest.TestCase):
    def test_summary_records_four_boundaries_without_first_prompt_claim(self) -> None:
        summary = build_related_work_summary()

        self.assertEqual(summary["status"], "PASS_RELATED_WORK_BOUNDARIES")
        self.assertEqual(summary["work_count"], 4)
        self.assertFalse(summary["claims"]["first_to_learn_prompts"])
        labels = {row["label"] for row in summary["rows"]}
        self.assertEqual(labels, {"GF-SAM", "SynPo", "AlignSAM", "Point Prompt Defender"})
        alignsam = next(row for row in summary["rows"] if row["label"] == "AlignSAM")
        self.assertEqual(alignsam["learned_prompt_policy"], "sequential_rl")
        self.assertEqual(alignsam["adapromptset_boundary"], "one_shot_offline_set_level")
        gfsam = next(row for row in summary["rows"] if row["label"] == "GF-SAM")
        self.assertEqual(gfsam["grouping"], "graph_rule")
        self.assertEqual(gfsam["budgeted_complete_action_selection"], False)

    def test_writer_emits_json_markdown_and_tex_table(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "related_work_assets"
        if root.exists():
            shutil.rmtree(root)
        try:
            outputs = write_related_work_assets(root)

            payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            markdown = outputs["markdown"].read_text(encoding="utf-8")
            table = outputs["table"].read_text(encoding="utf-8")
            self.assertEqual(payload["work_count"], 4)
            self.assertIn("GF-SAM", markdown)
            self.assertIn("Point Prompt Defender", markdown)
            self.assertIn(r"\begin{table*}", table)
            self.assertIn(r"\scriptsize", table)
            self.assertIn(r"\setlength{\tabcolsep}{3pt}", table)
            self.assertIn("Budgeted complete action", table)
            self.assertIn("AdaPromptSet boundary", table)
            self.assertIn("background ref.", table)
            self.assertNotIn("background_reference_evidence", table)
            self.assertNotIn("dual_space_graph_environment", table)
            self.assertNotIn("first to learn prompts", table.lower())
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
