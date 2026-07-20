from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.export_oracle_analysis_tables import (
    build_pivot_rows,
    format_decimal,
    write_oracle_analysis_assets,
)


class ExportOracleAnalysisTablesTests(unittest.TestCase):
    def test_format_decimal_uses_fixed_precision_and_na_for_missing(self) -> None:
        self.assertEqual(format_decimal(0.371238, digits=3), "0.371")
        self.assertEqual(format_decimal(None, digits=3), "N/A")

    def test_build_pivot_rows_keeps_policy_training_prohibited(self) -> None:
        monusac_summary = {
            "episode_count": 10,
            "action_count": 12,
            "best_fixed_action_id": "B_top5_pos3neg",
            "mean_best_fixed_dice": 0.37123847211635835,
            "mean_oracle_dice": 0.376249802278007,
            "mean_regret_to_oracle": 0.0050113301616486194,
            "mean_aji_regret_to_oracle": 0.006,
            "top_oracle_action_fraction": 0.5,
            "deterministic_promotion_status": "NOT_PROMOTED",
            "recommended_claim": "oracle_analysis_only",
        }
        glas_summary = {
            "gate_f_status": "PENDING_INSUFFICIENT_EPISODES",
            "episode_count": 1,
            "gate_min_episodes": 10,
            "best_fixed_action_id": "R_top5_independent_localneg",
            "mean_best_fixed_dice": 0.7180205482567982,
            "mean_oracle_dice": 0.7180205482567982,
            "dice_oracle_gap": 0.0,
        }

        rows = build_pivot_rows(monusac_summary, glas_summary)

        self.assertEqual(rows[0]["dataset"], "MoNuSAC")
        self.assertEqual(rows[0]["decision"], "PIVOT_ORACLE_ANALYSIS")
        self.assertEqual(rows[0]["allowed_claim"], "oracle_analysis_only")
        self.assertEqual(rows[0]["aji_oracle_gap"], 0.006)
        self.assertEqual(rows[0]["policy_training"], "prohibited")
        self.assertEqual(rows[1]["dataset"], "GlaS")
        self.assertEqual(rows[1]["decision"], "PENDING_INSUFFICIENT_EPISODES")
        self.assertEqual(rows[1]["policy_training"], "not_started")

    def test_write_oracle_analysis_assets_writes_claim_safe_tex_and_report(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_export_oracle_analysis_tables"
        root.mkdir(parents=True, exist_ok=True)
        (root / "reports" / "fixed_oracle_analysis").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "oracle_gap_pilot_glas").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "representation_analysis").mkdir(parents=True, exist_ok=True)
        (root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json").write_text(
            json.dumps(
                {
                    "episode_count": 10,
                    "action_count": 12,
                    "best_fixed_action_id": "B_top5_pos3neg",
                    "mean_best_fixed_dice": 0.37123847211635835,
                    "mean_oracle_dice": 0.376249802278007,
                    "mean_regret_to_oracle": 0.0050113301616486194,
                    "mean_aji_regret_to_oracle": 0.006,
                    "top_oracle_action_fraction": 0.5,
                    "deterministic_promotion_status": "NOT_PROMOTED",
                    "recommended_claim": "oracle_analysis_only",
                }
            ),
            encoding="utf-8",
        )
        (root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json").write_text(
            json.dumps(
                {
                    "gate_f_status": "PENDING_INSUFFICIENT_EPISODES",
                    "episode_count": 1,
                    "gate_min_episodes": 10,
                    "best_fixed_action_id": "R_top5_independent_localneg",
                    "mean_best_fixed_dice": 0.7180205482567982,
                    "mean_oracle_dice": 0.7180205482567982,
                    "dice_oracle_gap": 0.0,
                }
            ),
            encoding="utf-8",
        )
        (root / "reports" / "representation_analysis" / "representation_summary.json").write_text(
            json.dumps(
                {
                    "status": "PIVOT_REPRESENTATION_ANALYSIS",
                    "episode_count": 10,
                    "action_count": 12,
                    "best_fixed_action_id": "B_top5_pos3neg",
                    "mean_regret_to_oracle": 0.005011330161648608,
                    "mean_top2_dice_margin": 0.02044071753259889,
                    "near_tie_episode_fraction": 0.4,
                    "oracle_differs_from_best_fixed_fraction": 0.5,
                    "mean_candidate_point_precision": 0.52,
                    "mean_candidate_component_coverage": 0.052004904966278354,
                    "pearson_precision_vs_regret": -0.21408983836601292,
                    "pearson_coverage_vs_regret": -0.14407203442078728,
                    "representation_claim": "oracle_analysis_only_insufficient_separation",
                    "prohibited_next_actions": [
                        "monusac_policy_training",
                        "posthoc_monusac_action_redesign",
                        "test_time_action_search",
                    ],
                }
            ),
            encoding="utf-8",
        )
        with (root / "reports" / "fixed_oracle_analysis" / "action_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "action_id",
                    "action_family",
                    "mean_dice",
                    "mean_iou",
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
                    "mean_sam_calls": "1.0",
                    "mean_runtime_ms": "52.8733",
                    "oracle_count": "5",
                    "episode_count": "10",
                }
            )

        outputs = write_oracle_analysis_assets(root)

        for path in outputs.values():
            self.assertTrue(path.exists(), path)
        macros = (root / "paper" / "generated" / "oracle_analysis_macros.tex").read_text(encoding="utf-8")
        report = (root / "reports" / "oracle_analysis_paper_assets.md").read_text(encoding="utf-8")
        self.assertIn(r"\OracleAnalysisAJIGap", macros)
        self.assertIn("AJI oracle gap", report)
        pivot_table = outputs["pivot_table"].read_text(encoding="utf-8")
        action_table = outputs["action_table"].read_text(encoding="utf-8")
        representation_table = outputs["representation_table"].read_text(encoding="utf-8")
        macros = outputs["macros"].read_text(encoding="utf-8")
        report = outputs["report"].read_text(encoding="utf-8")
        self.assertIn(r"\begin{table*}[t]", pivot_table)
        self.assertIn(r"\end{table*}", pivot_table)
        self.assertIn(r"\small", pivot_table)
        self.assertIn("Pivot oracle analysis", pivot_table)
        self.assertIn("Pending insufficient episodes", pivot_table)
        self.assertNotIn(r"PENDING\_INSUFFICIENT\_EPISODES", pivot_table)
        self.assertIn(r"\begin{table*}[t]", action_table)
        self.assertIn(r"\end{table*}", action_table)
        self.assertIn(r"\small", action_table)
        self.assertIn(r"\begin{table*}[t]", representation_table)
        self.assertIn("Pivot representation analysis", representation_table)
        self.assertIn("0.020", representation_table)
        self.assertIn("0.400", representation_table)
        self.assertIn("0.520", representation_table)
        self.assertIn("0.052", representation_table)
        self.assertIn(r"\newcommand{\RepresentationAnalysisStatus}{PIVOT\_REPRESENTATION\_ANALYSIS}", macros)
        self.assertIn(r"\newcommand{\RepresentationAnalysisClaim}{oracle\_analysis\_only\_insufficient\_separation}", macros)
        self.assertIn(r"\newcommand{\RepresentationAnalysisTopTwoMargin}{0.020}", macros)
        self.assertIn("Representation pivot", report)
        self.assertIn("monusac_policy_training", report)
        combined = "\n".join(path.read_text(encoding="utf-8") for path in outputs.values())
        self.assertIn("PIVOT_ORACLE_ANALYSIS", combined)
        self.assertIn("PIVOT_REPRESENTATION_ANALYSIS", combined)
        self.assertIn("B_top5_pos3neg", combined)
        self.assertNotIn("SIMULATED PLACEHOLDER", combined)
        self.assertNotIn("\\MainPolicyDice", combined)
        self.assertNotIn("policy improves", combined.lower())


if __name__ == "__main__":
    unittest.main()
