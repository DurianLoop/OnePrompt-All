from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.export_oracle_analysis_figures import write_oracle_analysis_figures  # noqa: E402


class ExportOracleAnalysisFiguresTests(unittest.TestCase):
    def _fixture_root(self) -> Path:
        root = PROJECT_ROOT / "tmp" / "test_export_oracle_analysis_figures"
        analysis = root / "reports" / "fixed_oracle_analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        with (analysis / "budget_pareto.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "budget",
                    "best_fixed_action_id",
                    "mean_best_fixed_dice",
                    "mean_oracle_dice",
                    "dice_oracle_gap",
                    "aji_oracle_gap",
                    "episode_count",
                    "action_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "budget": "1",
                    "best_fixed_action_id": "B_top5_pos3neg",
                    "mean_best_fixed_dice": "0.37123847211635835",
                    "mean_oracle_dice": "0.3759059098495395",
                    "dice_oracle_gap": "0.004667437733181123",
                    "aji_oracle_gap": "0.003550037081815921",
                    "episode_count": "10",
                    "action_count": "7",
                }
            )
            writer.writerow(
                {
                    "budget": "5",
                    "best_fixed_action_id": "B_top5_pos3neg",
                    "mean_best_fixed_dice": "0.37123847211635835",
                    "mean_oracle_dice": "0.376249802278007",
                    "dice_oracle_gap": "0.0050113301616486194",
                    "aji_oracle_gap": "0.0038202062471413",
                    "episode_count": "10",
                    "action_count": "12",
                }
            )
        with (analysis / "action_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "action_id",
                    "action_family",
                    "mean_dice",
                    "mean_iou",
                    "mean_aji",
                    "mean_sam_calls",
                    "mean_runtime_ms",
                    "oracle_count",
                    "episode_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "action_id": "B_top5_pos3neg",
                    "action_family": "single_group_positive_negative",
                    "mean_dice": "0.37123847211635835",
                    "mean_iou": "0.22834095079987268",
                    "mean_aji": "0.22834095079987268",
                    "mean_sam_calls": "1.0",
                    "mean_runtime_ms": "41.4",
                    "oracle_count": "5",
                    "episode_count": "10",
                }
            )
            writer.writerow(
                {
                    "action_id": "R_split5_2groups_localneg",
                    "action_family": "rescue_group_specific_negatives",
                    "mean_dice": "0.2865910793308625",
                    "mean_iou": "0.17159767189916633",
                    "mean_aji": "0.17159767189916633",
                    "mean_sam_calls": "2.0",
                    "mean_runtime_ms": "107.1",
                    "oracle_count": "1",
                    "episode_count": "10",
                }
            )
        return root

    def test_write_oracle_analysis_figures_exports_pngs_and_provenance(self) -> None:
        root = self._fixture_root()

        outputs = write_oracle_analysis_figures(root)

        for key in ["pareto_png", "oracle_wins_png", "json", "markdown"]:
            self.assertTrue(outputs[key].exists(), outputs[key])
            self.assertGreater(outputs[key].stat().st_size, 0)
        summary = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS_ORACLE_ANALYSIS_FIGURES")
        self.assertEqual(summary["source_rows"]["budget_pareto"], 2)
        self.assertEqual(summary["source_rows"]["action_summary"], 2)
        self.assertEqual(summary["claim_boundary"], "oracle_analysis_only")
        self.assertIn("paper/figures/oracle_pareto.png", summary["figures"]["pareto_png"]["path"])
        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Oracle Analysis Figures", markdown)
        self.assertIn("oracle_analysis_only", markdown)
        self.assertNotIn("policy improves", markdown.lower())


if __name__ == "__main__":
    unittest.main()
