from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.reporting.pdf_metadata import EXPECTED_PAPER_METADATA  # noqa: E402
from adapromptset.scripts.build_paper import (  # noqa: E402
    build_paper,
    build_paper_draft,
    compile_pdf,
    guard_submission_placeholders,
    write_paper_skeleton,
)


class BuildPaperTests(unittest.TestCase):
    def _write_action_bank_fixture(self, root: Path) -> None:
        path = root / "configs" / "actions" / "prompt_set_bank_v2.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": "prompt_set_bank_v2",
                    "action_count": 5,
                    "valid_action_count": 4,
                    "invalid_action_count": 1,
                    "bank_hash": "F" * 64,
                    "source": {"positive_limit": 5},
                    "actions": [
                        {"action_id": "A_top1_one_group", "family": "single_group_positive", "valid": True},
                        {
                            "action_id": "B_top5_pos3neg",
                            "family": "single_group_positive_negative",
                            "valid": True,
                        },
                        {
                            "action_id": "C_top3_independent",
                            "family": "independent_positive_groups",
                            "valid": True,
                        },
                        {
                            "action_id": "R_split5_2groups_localneg",
                            "family": "rescue_group_specific_negatives",
                            "valid": True,
                        },
                        {"action_id": "A_top10_one_group", "family": "single_group_positive", "valid": False},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _fixture_root(self) -> Path:
        root = PROJECT_ROOT / "tmp" / "test_build_paper"
        self._write_action_bank_fixture(root)
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
                    "prohibited_next_actions": ["monusac_policy_training"],
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
        with (root / "reports" / "fixed_oracle_analysis" / "budget_pareto.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
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
        (root / "reports").mkdir(parents=True, exist_ok=True)
        return root

    def test_write_paper_skeleton_creates_claim_free_main_tex(self) -> None:
        root = self._fixture_root()

        outputs = write_paper_skeleton(root)

        main_tex = outputs["main"]
        self.assertTrue(main_tex.exists())
        text = main_tex.read_text(encoding="utf-8")
        self.assertIn("AdaPromptSet: Cost-Aware Prompt Set Selection", text)
        self.assertIn(r"\input{generated/oracle_analysis_macros}", text)
        self.assertIn(r"\usepackage{graphicx}", text)
        self.assertIn("PIVOT_ORACLE_ANALYSIS", text)
        self.assertIn(r"\simulatedfalse", text)
        self.assertNotIn(r"\simulatedtrue", text)
        self.assertNotIn(r"\usepackage{hyperref}", text)
        self.assertNotIn("policy improves", text.lower())
        self.assertNotIn("state of the art", text.lower())
        results = (root / "paper" / "sections" / "05_results.tex").read_text(encoding="utf-8")
        self.assertIn(r"\RepresentationAnalysisClaim{}", results)
        self.assertIn(r"\input{tables/table_representation_pivot}", results)
        self.assertIn("figures/oracle_pareto.png", results)
        self.assertIn("figures/oracle_action_wins.png", results)
        related = (root / "paper" / "sections" / "02_related.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{tables/table_related_work}", related)
        method = (root / "paper" / "sections" / "03_method.tex").read_text(encoding="utf-8")
        self.assertIn(r"\input{figures/method_overview}", method)
        self.assertIn(r"\input{figures/action_bank}", method)
        self.assertIn("current v2 MVP action-bank subset", method)
        self.assertTrue((root / "paper" / "tables" / "table_related_work.tex").exists())
        self.assertTrue((root / "paper" / "references.bib").exists())

    def test_guard_submission_placeholders_rejects_simulated_tokens(self) -> None:
        root = self._fixture_root()
        paper = root / "paper"
        paper.mkdir(parents=True, exist_ok=True)
        (paper / "main.tex").write_text(r"\simulatedtrue SIMULATED PLACEHOLDER", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "simulated placeholders remain"):
            guard_submission_placeholders(root)

    def test_draft_build_marks_watermark_and_submission_build_clears_it(self) -> None:
        root = self._fixture_root()

        build_paper(root, mode="draft")
        draft_text = (root / "paper" / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\simulatedtrue", draft_text)
        self.assertIn("DRAFT WATERMARK", draft_text)
        with self.assertRaisesRegex(ValueError, "simulated placeholders remain"):
            guard_submission_placeholders(root)

        build_paper(root, mode="submission")
        submission_text = (root / "paper" / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\simulatedfalse", submission_text)
        self.assertIn(r"\pdfinfo{", submission_text)
        self.assertIn(f"/Title ({EXPECTED_PAPER_METADATA.title})", submission_text)
        self.assertIn(f"/Author ({EXPECTED_PAPER_METADATA.author})", submission_text)
        self.assertIn(f"\\author{{{EXPECTED_PAPER_METADATA.author}}}", submission_text)
        self.assertIn(f"\\date{{{EXPECTED_PAPER_METADATA.date}}}", submission_text)
        self.assertNotIn(r"\simulatedtrue", submission_text)
        self.assertNotIn("DRAFT WATERMARK", submission_text)
        guard_submission_placeholders(root)

    def test_build_paper_draft_exports_assets_and_skeleton(self) -> None:
        root = self._fixture_root()

        outputs = build_paper_draft(root)

        self.assertTrue(outputs["main"].exists())
        self.assertTrue((root / "paper" / "generated" / "oracle_analysis_macros.tex").exists())
        self.assertTrue((root / "paper" / "tables" / "table_oracle_pivot.tex").exists())
        self.assertTrue((root / "paper" / "tables" / "table_action_summary.tex").exists())
        self.assertTrue((root / "paper" / "tables" / "table_representation_pivot.tex").exists())
        self.assertTrue((root / "paper" / "figures" / "oracle_pareto.png").exists())
        self.assertTrue((root / "paper" / "figures" / "oracle_action_wins.png").exists())
        self.assertTrue((root / "paper" / "figures" / "method_overview.tex").exists())
        self.assertTrue((root / "paper" / "figures" / "action_bank.tex").exists())

    def test_compile_pdf_runs_twice_to_resolve_cross_references(self) -> None:
        root = self._fixture_root()

        with patch("adapromptset.scripts.build_paper.subprocess.run") as run:
            compile_pdf(root)

        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
