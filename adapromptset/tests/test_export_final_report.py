from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.export_final_report import export_final_report  # noqa: E402


class ExportFinalReportTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_export_final_report_records_pivot_boundaries_and_repro_commands(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_export_final_report"
        self._write_json(
            root / "reports" / "decision_checkpoint" / "phase5_decision.json",
            {
                "phase5_status": "PIVOT_ORACLE_ANALYSIS",
                "v2_dice_oracle_gap": 0.0050113301616486194,
                "v2_gate_f_min_dice_gap": 0.01,
                "v2_best_fixed_action_id": "B_top5_pos3neg",
                "allowed_next_actions": ["fixed_rule_action_oracle_analysis"],
                "prohibited_next_actions": [
                    "policy_training_on_monusac",
                    "posthoc_monusac_action_redesign",
                ],
            },
        )
        self._write_json(
            root / "reports" / "gates" / "gate_f_oracle_gap.json",
            {
                "status": "FAIL_ORACLE_GAP_SMALL",
                "pass_fail": "FAIL",
                "diagnostics": {
                    "episode_count": 10,
                    "record_count": 120,
                    "dice_oracle_gap": 0.0050113301616486194,
                    "aji_available": True,
                    "aji_oracle_gap": 0.0042,
                    "gate_f_min_dice_gap": 0.01,
                },
            },
        )
        self._write_json(
            root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json",
            {
                "best_fixed_action_id": "B_top5_pos3neg",
                "mean_best_fixed_dice": 0.37123847211635835,
                "mean_oracle_dice": 0.376249802278007,
                "mean_regret_to_oracle": 0.0050113301616486194,
                "aji_available": True,
                "mean_best_fixed_aji": 0.208,
                "mean_oracle_aji": 0.214,
                "mean_aji_regret_to_oracle": 0.006,
                "deterministic_promotion_status": "NOT_PROMOTED",
                "recommended_claim": "oracle_analysis_only",
            },
        )
        self._write_json(
            root / "reports" / "representation_analysis" / "representation_summary.json",
            {
                "status": "PIVOT_REPRESENTATION_ANALYSIS",
                "mean_top2_dice_margin": 0.02044071753259889,
                "near_tie_episode_fraction": 0.4,
                "representation_claim": "oracle_analysis_only_insufficient_separation",
                "prohibited_next_actions": [
                    "monusac_policy_training",
                    "test_time_action_search",
                ],
            },
        )
        self._write_json(
            root / "reports" / "gate_f_failure_diagnostics" / "gate_f_failure_diagnostics.json",
            {
                "status": "PASS_DIAGNOSTIC_ONLY",
                "policy_training_allowed": False,
                "coverage_diagnostic": {
                    "status": "LOW_COMPONENT_COVERAGE",
                    "mean_component_coverage_at_max_pool": 0.052,
                },
                "proposal_diagnostic": {
                    "status": "ACTIONS_SEPARATED",
                    "mean_top2_dice_margin": 0.0204,
                    "near_tie_episode_fraction": 0.4,
                },
                "sam_sensitivity_diagnostic": {
                    "status": "NO_HIGHER_CALL_GAIN",
                    "best_high_call_minus_best_one_call": -0.02,
                },
                "interpretation_flags": [
                    "candidate_pool_limited_coverage",
                    "no_clear_sam_call_grouping_gain",
                ],
            },
        )
        self._write_json(
            root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json",
            {
                "status": "PENDING_INSUFFICIENT_EPISODES",
                "episode_count": 1,
                "gate_min_episodes": 10,
                "dice_oracle_gap": 0.0,
            },
        )
        self._write_json(
            root / "reports" / "bootstrap" / "bootstrap_summary.json",
            {
                "status": "PENDING_INSUFFICIENT_CLUSTERS",
                "bootstrap": {
                    "cluster_count": 1,
                    "query_count": 2,
                    "observed_mean_delta": 0.0050113301616486194,
                },
                "paired_significance": {"p_value_two_sided": 0.5},
            },
        )
        self._write_json(
            root / "reports" / "reference_exposure_audit.json",
            {
                "status": "PASS",
                "checks": {
                    "reference_pool_train_only": True,
                    "reference_test_leakage_zero": True,
                },
                "dataset_count": 3,
                "total_issues": 0,
            },
        )
        self._write_json(
            root / "reports" / "data_profile_audit.json",
            {
                "status": "PASS",
                "dataset_count": 3,
                "row_count": 526,
                "datasets": [
                    {
                        "dataset": "MoNuSAC",
                        "case_count": 310,
                        "width": {"mean": 602.37},
                        "height": {"mean": 558.70},
                        "mask_area_pixels": {"mean": 92084.89},
                        "mask_area_fraction": {"mean": 0.220377},
                        "instance_count": {"mean": 159.07},
                    }
                ],
                "issues": [],
            },
        )
        self._write_json(
            root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.json",
            {
                "status": "PASS_PROTOCOL_DEFINED",
                "total_image_count": 757,
                "total_complete_case_count": 757,
                "total_missing_label_count": 0,
                "stage1_union": {
                    "mask_rule": "binary_union_of_EX_HE_MA_SE",
                    "allowed_use": "infrastructure_only",
                },
                "stage2_four_intents": {
                    "intents": ["EX", "HE", "MA", "SE"],
                },
                "formal_policy_training_allowed": False,
            },
        )
        self._write_json(
            root / "reports" / "new_domain_pilot_readiness.json",
            {
                "status": "READY_WITH_BLOCKED_FORMAL_DATASETS",
                "blocked_datasets": ["glas", "cpm17"],
                "infrastructure_ready_datasets": ["ddr"],
                "datasets": {
                    "glas": {
                        "status": "BLOCKED_INSUFFICIENT_EPISODES",
                        "gate": {"episode_count": 1, "gate_min_episodes": 10},
                    },
                    "ddr": {"status": "READY_FOR_INFRASTRUCTURE_ONLY"},
                    "idrid": {"status": "READ_ONLY_NON_TUNABLE"},
                    "cpm17": {"status": "DEFERRED_MISSING_OR_INCOMPLETE"},
                },
                "global_allowed_next_actions": [
                    "predeclare_glas_trainval_diagnostic_without_official_test",
                    "ddr_stage1_infrastructure_smoke",
                ],
                "global_prohibited_next_actions": [
                    "formal_new_domain_policy_training_without_predeclared_gate_f",
                    "promote_insufficient_episode_smoke_to_gate_pass",
                ],
            },
        )
        self._write_json(
            root / "reports" / "oracle_analysis_figures.json",
            {
                "status": "PASS_ORACLE_ANALYSIS_FIGURES",
                "claim_boundary": "oracle_analysis_only",
                "source_rows": {"budget_pareto": 4, "action_summary": 12},
                "figures": {
                    "pareto_png": {
                        "path": "paper/figures/oracle_pareto.png",
                        "bytes": 100,
                        "sha256": "A" * 64,
                    },
                    "oracle_wins_png": {
                        "path": "paper/figures/oracle_action_wins.png",
                        "bytes": 200,
                        "sha256": "B" * 64,
                    },
                },
            },
        )
        self._write_json(
            root / "reports" / "method_overview_figure.json",
            {
                "status": "PASS_METHOD_OVERVIEW_FIGURE",
                "claim_boundary": "oracle_analysis_only",
                "figure": {
                    "path": "paper/figures/method_overview.tex",
                    "bytes": 2310,
                    "sha256": "C" * 64,
                },
                "checks": {
                    "frozen_matcher_present": True,
                    "frozen_segmenter_present": True,
                    "train_only_utility_labels_present": True,
                    "test_time_safe_path_present": True,
                    "gate_f_policy_prohibition_present": True,
                    "tikz_present": True,
                },
            },
        )
        self._write_json(
            root / "reports" / "action_bank_figure.json",
            {
                "status": "PASS_ACTION_BANK_FIGURE",
                "claim_boundary": "oracle_analysis_only_current_mvp_subset",
                "source_bank": {
                    "path": "configs/actions/prompt_set_bank_v2.json",
                    "sha256": "D" * 64,
                    "name": "prompt_set_bank_v2",
                    "bank_hash": "F" * 64,
                    "action_count": 13,
                    "valid_action_count": 12,
                    "invalid_action_count": 1,
                },
                "family_counts": {
                    "independent_positive_groups": 2,
                    "rescue_group_specific_negatives": 2,
                    "rescue_split_groups": 1,
                    "single_group_positive": 3,
                    "single_group_positive_negative": 4,
                },
                "figure": {
                    "path": "paper/figures/action_bank.tex",
                    "bytes": 2500,
                    "sha256": "E" * 64,
                },
                "checks": {
                    "tikz_present": True,
                    "source_bank_present": True,
                    "topk_family_present": True,
                    "positive_negative_family_present": True,
                    "independent_groups_family_present": True,
                    "rescue_family_present": True,
                    "deferred_families_disclosed": True,
                    "sam_calls_and_points_present": True,
                    "complete_prompt_set_boundary_present": True,
                    "valid_counts_match_source": True,
                },
                "deferred_families": [
                    "kmeans_groups",
                    "density_groups",
                    "gf_sam_style_graph_groups",
                ],
            },
        )
        self._write_json(
            root / "reports" / "phase0_scope_structure" / "phase0_scope_structure.json",
            {
                "status": "PASS_SCOPE_STRUCTURE",
                "policy_training_allowed": False,
                "structure": {"status": "PASS", "missing_paths": []},
                "toolchain": {"status": "PASS_WITH_LINT_DEFERRED"},
                "research_question": {
                    "text": "Under frozen DINOv2-L and SAM-B, does budget-constrained structured prompt-set selection show enough query-specific oracle heterogeneity to justify learned selection?",
                    "does_not_claim_first_to_learn_prompts": True,
                },
                "similar_work_boundaries": {
                    "gf_sam": {"label": "GF-SAM", "present": True},
                    "synpo": {"label": "SynPo", "present": True},
                    "alignsam": {"label": "AlignSAM", "present": True},
                    "ppd": {"label": "Point Prompt Defender", "present": True},
                },
                "rq_hypotheses": {"status": "PASS", "rq_count": 3, "hypothesis_count": 5},
                "scope": {
                    "status": "PASS",
                    "non_mainline_exclusions": {
                        "sequential_rl": {"present": True, "mainline": False},
                        "continuous_coordinate_generation": {"present": True, "mainline": False},
                        "online_sam_response_refinement": {"present": True, "mainline": False},
                    },
                },
            },
        )
        self._write_json(
            root / "reports" / "paper_section_audit.json",
            {
                "status": "PASS",
                "checks": {
                    "abstract_claim_free": True,
                    "related_work_boundaries_present": True,
                    "method_no_end_to_end_differentiable": True,
                    "experiments_statistical_units_present": True,
                    "results_cost_and_accuracy_present": True,
                    "oracle_privileged_caption_boundaries": True,
                    "limitations_action_bank_upper_bound": True,
                    "limitations_reference_quality": True,
                    "limitations_frozen_matcher_irrecoverable": True,
                    "limitations_utility_generation_cost": True,
                    "conclusion_oracle_analysis_only": True,
                },
                "violations": [],
            },
        )
        self._write_json(
            root / "reports" / "latexmk_audit.json",
            {
                "status": "PASS",
                "checks": {
                    "latexmk_exit_zero": True,
                    "clean_copy_pdf_exists": True,
                    "clean_copy_log_exists": True,
                    "log_problem_scan_clean": True,
                },
                "pdf": {
                    "exists": True,
                    "bytes": 1234,
                    "pages": "4",
                },
                "clean_copy_dir": "tmp/latexmk_clean_abc",
                "violations": [],
            },
        )
        self._write_json(
            root / "reports" / "pdf_metadata_audit.json",
            {
                "status": "PASS",
                "expected": {
                    "title": "AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation",
                    "author": "Internal MVP Draft",
                    "date": "2026-07-17",
                    "subject": "Oracle-analysis-only AdaPromptSet MVP artifact after Gate-F pivot",
                    "keywords": "AdaPromptSet, oracle analysis, prompt set selection, medical segmentation",
                },
                "checks": {
                    "main_tex_exists": True,
                    "pdf_exists": True,
                    "visible_title_present": True,
                    "visible_author_present": True,
                    "visible_date_present": True,
                    "pdfinfo_available": True,
                    "pdf_title_matches": True,
                    "pdf_author_matches": True,
                    "pdf_subject_matches": True,
                    "pdf_keywords_match": True,
                },
                "violations": [],
            },
        )
        self._write_json(
            root / "reports" / "run_metadata" / "phase0_reproducibility" / "run_metadata.json",
            {
                "run_id": "phase0_reproducibility_abc",
                "config_hash": "CONFIGHASH",
                "environment": {
                    "python_executable": "D:\\SAM\\conda_envs\\sam_gpu\\python.exe",
                    "gpu_name": "NVIDIA Test GPU",
                    "torch": "2.x",
                },
            },
        )
        (root / "reports" / "compute_ledger.csv").parent.mkdir(parents=True, exist_ok=True)
        (root / "reports" / "compute_ledger.csv").write_text(
            "phase,command,gpu_hours,notes\n"
            "6-oracle-gap-pilot,scripts/run_oracle_gap_pilot.py,0.008595,MoNuSAC validation\n",
            encoding="utf-8",
        )
        (root / "paper").mkdir(parents=True, exist_ok=True)
        (root / "paper" / "main.pdf").write_bytes(b"%PDF-final-report-test")

        outputs = export_final_report(root)

        self.assertTrue(outputs["json"].exists())
        self.assertTrue(outputs["markdown"].exists())
        summary = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PIVOT_ORACLE_ANALYSIS")
        self.assertEqual(summary["gate_f"]["status"], "FAIL_ORACLE_GAP_SMALL")
        self.assertEqual(summary["gate_f"]["aji_oracle_gap"], 0.0042)
        self.assertEqual(summary["claim_boundary"], "oracle_analysis_only")
        self.assertEqual(summary["fixed_oracle"]["mean_aji_regret_to_oracle"], 0.006)
        self.assertIn("policy_training_on_monusac", summary["prohibited_next_actions"])
        self.assertIn("test_time_action_search", summary["prohibited_next_actions"])
        self.assertEqual(summary["external_pilots"]["glas"]["status"], "PENDING_INSUFFICIENT_EPISODES")
        self.assertEqual(summary["statistics"]["status"], "PENDING_INSUFFICIENT_CLUSTERS")
        self.assertEqual(summary["reference_exposure"]["status"], "PASS")
        self.assertTrue(summary["reference_exposure"]["checks"]["reference_test_leakage_zero"])
        self.assertEqual(summary["data_profile"]["status"], "PASS")
        self.assertEqual(summary["data_profile"]["row_count"], 526)
        self.assertEqual(summary["ddr_protocol"]["status"], "PASS_PROTOCOL_DEFINED")
        self.assertEqual(summary["ddr_protocol"]["stage1_mask_rule"], "binary_union_of_EX_HE_MA_SE")
        self.assertEqual(summary["ddr_protocol"]["stage2_intents"], ["EX", "HE", "MA", "SE"])
        self.assertFalse(summary["ddr_protocol"]["formal_policy_training_allowed"])
        self.assertEqual(summary["new_domain_pilot_readiness"]["status"], "READY_WITH_BLOCKED_FORMAL_DATASETS")
        self.assertEqual(summary["new_domain_pilot_readiness"]["glas_status"], "BLOCKED_INSUFFICIENT_EPISODES")
        self.assertEqual(summary["new_domain_pilot_readiness"]["ddr_status"], "READY_FOR_INFRASTRUCTURE_ONLY")
        self.assertEqual(summary["new_domain_pilot_readiness"]["idrid_status"], "READ_ONLY_NON_TUNABLE")
        self.assertIn(
            "formal_new_domain_policy_training_without_predeclared_gate_f",
            summary["new_domain_pilot_readiness"]["global_prohibited_next_actions"],
        )
        self.assertEqual(summary["oracle_analysis_figures"]["status"], "PASS_ORACLE_ANALYSIS_FIGURES")
        self.assertEqual(summary["oracle_analysis_figures"]["claim_boundary"], "oracle_analysis_only")
        self.assertEqual(summary["oracle_analysis_figures"]["source_rows"]["action_summary"], 12)
        self.assertIn("pareto_png", summary["oracle_analysis_figures"]["figures"])
        self.assertEqual(summary["method_overview_figure"]["status"], "PASS_METHOD_OVERVIEW_FIGURE")
        self.assertTrue(summary["method_overview_figure"]["checks"]["frozen_matcher_present"])
        self.assertTrue(summary["method_overview_figure"]["checks"]["train_only_utility_labels_present"])
        self.assertEqual(summary["action_bank_figure"]["status"], "PASS_ACTION_BANK_FIGURE")
        self.assertEqual(summary["action_bank_figure"]["source_bank"]["name"], "prompt_set_bank_v2")
        self.assertTrue(summary["action_bank_figure"]["checks"]["deferred_families_disclosed"])
        self.assertIn("gf_sam_style_graph_groups", summary["action_bank_figure"]["deferred_families"])
        self.assertEqual(summary["phase0_scope_structure"]["status"], "PASS_SCOPE_STRUCTURE")
        self.assertEqual(summary["phase0_scope_structure"]["toolchain_status"], "PASS_WITH_LINT_DEFERRED")
        self.assertFalse(summary["phase0_scope_structure"]["policy_training_allowed"])
        self.assertTrue(summary["phase0_scope_structure"]["research_question_does_not_claim_first"])
        self.assertTrue(summary["phase0_scope_structure"]["non_mainline_exclusions"]["sequential_rl"]["present"])
        self.assertEqual(summary["paper_section_audit"]["status"], "PASS")
        self.assertTrue(summary["paper_section_audit"]["checks"]["limitations_reference_quality"])
        self.assertTrue(summary["paper_section_audit"]["checks"]["results_cost_and_accuracy_present"])
        self.assertTrue(summary["paper_section_audit"]["checks"]["oracle_privileged_caption_boundaries"])
        self.assertEqual(summary["latexmk_audit"]["status"], "PASS")
        self.assertTrue(summary["latexmk_audit"]["checks"]["log_problem_scan_clean"])
        self.assertEqual(summary["latexmk_audit"]["pdf"]["pages"], "4")
        self.assertEqual(summary["pdf_metadata_audit"]["status"], "PASS")
        self.assertTrue(summary["pdf_metadata_audit"]["checks"]["pdf_title_matches"])
        self.assertEqual(summary["gate_f_failure_diagnostics"]["coverage_status"], "LOW_COMPONENT_COVERAGE")
        self.assertEqual(summary["gate_f_failure_diagnostics"]["sam_sensitivity_status"], "NO_HIGHER_CALL_GAIN")
        self.assertFalse(summary["gate_f_failure_diagnostics"]["policy_training_allowed"])
        self.assertEqual(summary["pdf"]["bytes"], len(b"%PDF-final-report-test"))
        self.assertRegex(summary["pdf"]["sha256"], r"^[A-F0-9]{64}$")
        self.assertGreaterEqual(len(summary["reproduction_commands"]), 8)
        self.assertNotIn("{root}", "\n".join(summary["reproduction_commands"]))
        self.assertIn(str(root / "src"), summary["reproduction_commands"][1])
        self.assertIn("scripts\\audit_reference_exposure.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_ddr_protocol.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_new_domain_pilot_readiness.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_phase0_scope_structure.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_gate_f_failure_diagnostics.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\export_oracle_analysis_figures.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\export_method_overview_figure.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\export_action_bank_figure.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_latexmk_build.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_pdf_metadata.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\export_final_report.py", "\n".join(summary["reproduction_commands"]))
        self.assertIn("scripts\\audit_scientific_claims.py", summary["reproduction_commands"][-1])

        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Mean oracle AJI", markdown)
        self.assertIn("AJI oracle gap", markdown)
        self.assertIn("PIVOT_ORACLE_ANALYSIS", markdown)
        self.assertIn("Policy training on MoNuSAC remains prohibited", markdown)
        self.assertIn("PENDING_INSUFFICIENT_EPISODES", markdown)
        self.assertIn("PENDING_INSUFFICIENT_CLUSTERS", markdown)
        self.assertIn("Reference exposure", markdown)
        self.assertIn("Data profile", markdown)
        self.assertIn("DDR protocol", markdown)
        self.assertIn("New-domain readiness", markdown)
        self.assertIn("READY_WITH_BLOCKED_FORMAL_DATASETS", markdown)
        self.assertIn("BLOCKED_INSUFFICIENT_EPISODES", markdown)
        self.assertIn("Oracle-analysis figures", markdown)
        self.assertIn("PASS_ORACLE_ANALYSIS_FIGURES", markdown)
        self.assertIn("Method overview figure", markdown)
        self.assertIn("PASS_METHOD_OVERVIEW_FIGURE", markdown)
        self.assertIn("Action-bank figure", markdown)
        self.assertIn("PASS_ACTION_BANK_FIGURE", markdown)
        self.assertIn("prompt_set_bank_v2", markdown)
        self.assertIn("binary_union_of_EX_HE_MA_SE", markdown)
        self.assertIn("Phase 0 scope", markdown)
        self.assertIn("PASS_SCOPE_STRUCTURE", markdown)
        self.assertIn("Gate F failure diagnostics", markdown)
        self.assertIn("LOW_COMPONENT_COVERAGE", markdown)
        self.assertIn("Paper section audit", markdown)
        self.assertIn("Results cost and accuracy", markdown)
        self.assertIn("Limitations reference quality", markdown)
        self.assertIn("Oracle/privileged captions", markdown)
        self.assertIn("latexmk clean build", markdown)
        self.assertIn("no privileged-input row is included", markdown)
        self.assertIn("PDF metadata", markdown)
        self.assertIn("Internal MVP Draft", markdown)
        self.assertIn("Oracle rows are analysis upper bounds, not deployable method outputs", markdown)
        self.assertIn("Background reference evidence is not explicit SAM negative point prompting", markdown)
        self.assertIn("Multi-reference episodes are repeated measurements and not independent samples", markdown)
        self.assertIn("No in-domain retraining or transfer claim is made", markdown)
        self.assertIn("Failure results and limitations are disclosed", markdown)
        self.assertIn("```powershell", markdown)
        self.assertNotIn("SIMULATED PLACEHOLDER", markdown)
        self.assertNotIn("policy improves", markdown.lower())


if __name__ == "__main__":
    unittest.main()
