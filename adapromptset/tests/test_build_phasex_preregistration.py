from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_phasex_preregistration import (
    PHASEX_BRANCH,
    PHASEX_COMMIT_MESSAGE,
    PHASEX_NAME,
    POSITIVE_K_GRID,
    NEGATIVE_K_GRID,
    POSITIVE_SELECTORS,
    NEGATIVE_SELECTORS,
    GROUPINGS,
    CALL_BUDGETS,
    POINT_BUDGETS,
    A_SMALL_LIMITS,
    A_MEDIUM_LIMITS,
    A_LARGE_LIMITS,
    SEARCH_BEAM_WIDTHS,
    SEARCH_DEPTHS,
    SEARCH_CALL_BUDGETS,
    SEARCH_POINT_BUDGETS,
    ADAPTIVE_VALUE_DICE_THRESHOLD,
    ADAPTIVE_VALUE_MIN_QUERY_FRACTION,
    COMPUTE_CAPS,
    build_phasex_preregistration,
    load_historical_artifacts,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_bank_v1() -> dict[str, object]:
    return {
        "name": "prompt_set_bank_v1",
        "action_count": 1,
        "valid_action_count": 1,
        "bank_hash": "V1HASH" * 12 + "ABCD",
        "actions": [
            {
                "action_id": "A_top1_one_group",
                "family": "single_group_positive",
                "groups": [{"positive_indices": [0], "negative_indices": []}],
                "valid": True,
                "invalid_reason": "",
                "sam_calls": 1,
                "prompt_points": 1,
                "content_hash": "C1" * 32,
            }
        ],
    }


def _make_bank_v2() -> dict[str, object]:
    return {
        "name": "prompt_set_bank_v2",
        "action_count": 2,
        "valid_action_count": 2,
        "bank_hash": "V2HASH" * 12 + "ABCD",
        "actions": [
            {
                "action_id": "A_top1_one_group",
                "family": "single_group_positive",
                "groups": [{"positive_indices": [0], "negative_indices": []}],
                "valid": True,
                "invalid_reason": "",
                "sam_calls": 1,
                "prompt_points": 1,
                "content_hash": "D1" * 32,
            },
            {
                "action_id": "A_top5_one_group",
                "family": "single_group_positive",
                "groups": [{"positive_indices": [0, 1, 2, 3, 4], "negative_indices": []}],
                "valid": True,
                "invalid_reason": "",
                "sam_calls": 1,
                "prompt_points": 5,
                "content_hash": "D5" * 32,
            },
        ],
    }


def _make_oracle_v2_summary() -> dict[str, object]:
    return {
        "gate_f_status": "FAIL_ORACLE_GAP_SMALL",
        "episode_count": 10,
        "record_count": 20,
        "valid_record_count": 20,
        "best_fixed_action_id": "A_top5_one_group",
        "mean_oracle_dice": 0.300000,
        "mean_best_fixed_dice": 0.295000,
        "dice_oracle_gap": 0.005000,
    }


def _make_fixed_oracle_summary() -> dict[str, object]:
    return {
        "episode_count": 10,
        "action_count": 2,
        "best_fixed_action_id": "A_top5_one_group",
        "mean_best_fixed_dice": 0.295000,
        "mean_oracle_dice": 0.300000,
        "mean_regret_to_oracle": 0.005000,
        "aji_available": True,
        "best_fixed_aji_action_id": "A_top5_one_group",
        "mean_best_fixed_aji": 0.190000,
        "mean_oracle_aji": 0.194000,
        "mean_aji_regret_to_oracle": 0.004000,
        "median_regret_to_oracle": 0.000500,
        "max_regret_to_oracle": 0.018000,
        "near_oracle_tolerance": 0.005,
        "near_oracle_episode_fraction": 0.6,
        "oracle_action_counts": {"A_top5_one_group": 8, "A_top1_one_group": 2},
        "oracle_family_counts": {"single_group_positive": 10},
        "top_oracle_action_fraction": 0.8,
        "deterministic_promotion_status": "NOT_PROMOTED",
        "recommended_claim": "oracle_analysis_only",
    }


def _make_phase5_decision() -> dict[str, object]:
    return {
        "phase5_status": "PIVOT_ORACLE_ANALYSIS",
        "rationale": "test rationale",
        "v2_dice_oracle_gap": 0.005,
    }


def _make_gate_f_report() -> dict[str, object]:
    return {
        "gate": "F",
        "status": "FAIL_ORACLE_GAP_SMALL",
        "pass_fail": "FAIL",
    }


def _seed_tmp_project(tmp_root: Path) -> None:
    _write_json(tmp_root / "configs" / "actions" / "prompt_set_bank_v1.json", _make_bank_v1())
    _write_json(tmp_root / "configs" / "actions" / "prompt_set_bank_v2.json", _make_bank_v2())
    _write_json(
        tmp_root / "reports" / "oracle_gap_pilot_v2" / "oracle_gap_pilot_summary.json",
        _make_oracle_v2_summary(),
    )
    _write_json(
        tmp_root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json",
        _make_fixed_oracle_summary(),
    )
    _write_json(
        tmp_root / "reports" / "decision_checkpoint" / "phase5_decision.json",
        _make_phase5_decision(),
    )
    _write_json(
        tmp_root / "reports" / "gates" / "gate_f_oracle_gap.json",
        _make_gate_f_report(),
    )

    _write_csv(
        tmp_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv",
        [
            "episode_id",
            "action_id",
            "action_family",
            "valid",
            "invalid_reason",
            "sam_calls",
            "decoder_calls",
            "cache_hits",
            "num_groups",
            "num_positive",
            "num_negative",
            "dice",
            "iou",
            "aji",
            "retained_masks",
            "runtime_ms",
        ],
        [
            {
                "episode_id": "e0",
                "action_id": "A_top1_one_group",
                "action_family": "single_group_positive",
                "valid": "True",
                "invalid_reason": "",
                "sam_calls": "1",
                "decoder_calls": "1",
                "cache_hits": "0",
                "num_groups": "1",
                "num_positive": "1",
                "num_negative": "0",
                "dice": "0.05",
                "iou": "0.03",
                "aji": "0.03",
                "retained_masks": "1",
                "runtime_ms": "10.5",
            },
            {
                "episode_id": "e0",
                "action_id": "A_top5_one_group",
                "action_family": "single_group_positive",
                "valid": "True",
                "invalid_reason": "",
                "sam_calls": "1",
                "decoder_calls": "1",
                "cache_hits": "0",
                "num_groups": "1",
                "num_positive": "5",
                "num_negative": "0",
                "dice": "0.295",
                "iou": "0.18",
                "aji": "0.18",
                "retained_masks": "1",
                "runtime_ms": "41.4",
            },
        ],
    )

    _write_csv(
        tmp_root / "reports" / "fixed_oracle_analysis" / "action_summary.csv",
        [
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
        [
            {
                "action_id": "A_top1_one_group",
                "action_family": "single_group_positive",
                "mean_dice": "0.05",
                "mean_iou": "0.03",
                "mean_aji": "0.03",
                "mean_sam_calls": "1.0",
                "mean_runtime_ms": "10.5",
                "oracle_count": "2",
                "episode_count": "10",
            },
            {
                "action_id": "A_top5_one_group",
                "action_family": "single_group_positive",
                "mean_dice": "0.295",
                "mean_iou": "0.18",
                "mean_aji": "0.18",
                "mean_sam_calls": "1.0",
                "mean_runtime_ms": "41.4",
                "oracle_count": "8",
                "episode_count": "10",
            },
        ],
    )

    _write_csv(
        tmp_root / "reports" / "fixed_oracle_analysis" / "budget_pareto.csv",
        [
            "budget",
            "best_fixed_action_id",
            "mean_best_fixed_dice",
            "mean_oracle_dice",
            "dice_oracle_gap",
            "aji_oracle_gap",
            "episode_count",
            "action_count",
        ],
        [
            {
                "budget": "1",
                "best_fixed_action_id": "A_top5_one_group",
                "mean_best_fixed_dice": "0.295",
                "mean_oracle_dice": "0.300",
                "dice_oracle_gap": "0.005",
                "aji_oracle_gap": "0.004",
                "episode_count": "10",
                "action_count": "2",
            },
        ],
    )

    _write_csv(
        tmp_root / "reports" / "decision_checkpoint" / "oracle_action_heatmap.csv",
        ["episode_id", "action_id", "dice", "is_oracle"],
        [
            {"episode_id": "e0", "action_id": "A_top1_one_group", "dice": "0.05", "is_oracle": "0"},
            {"episode_id": "e0", "action_id": "A_top5_one_group", "dice": "0.295", "is_oracle": "1"},
        ],
    )

    _write_csv(
        tmp_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv",
        ["episode_id", "dataset", "role", "query_case_id"],
        [
            {"episode_id": "e0", "dataset": "MoNuSAC", "role": "val", "query_case_id": "q0"},
        ],
    )

    _write_csv(
        tmp_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv",
        ["episode_id", "positive_limit", "kind", "grid_row", "grid_col", "x", "y", "score"],
        [
            {"episode_id": "e0", "positive_limit": "5", "kind": "positive", "grid_row": "0", "grid_col": "0", "x": "1", "y": "1", "score": "1.0"},
        ],
    )

    # Seed the live docs so the append code has something to preserve.
    (tmp_root / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_root / "docs" / "protocol_history.md").write_text("# Protocol History\n", encoding="utf-8")
    (tmp_root / "docs" / "decision_log.md").write_text("# Decision Log\n", encoding="utf-8")
    (tmp_root / "docs" / "data_exposure_log.md").write_text("# Data Exposure Log\n", encoding="utf-8")
    (tmp_root / "checklist.md").write_text("# Checklist\n", encoding="utf-8")


class PhaseXPreregistrationConstantsTests(unittest.TestCase):
    """Verify the frozen protocol constants match the directive file."""

    def test_positive_k_grid_matches_directive(self) -> None:
        self.assertEqual(POSITIVE_K_GRID, (1, 3, 5, 10, 20, 30, 50, 75, 100))

    def test_negative_k_grid_matches_directive(self) -> None:
        self.assertEqual(NEGATIVE_K_GRID, (0, 1, 3, 5, 10, 20, 30, 50))

    def test_positive_selectors_count(self) -> None:
        self.assertEqual(len(POSITIVE_SELECTORS), 6)
        self.assertIn("P_TOPK", POSITIVE_SELECTORS)
        self.assertIn("P_DIVERSITY_GREEDY", POSITIVE_SELECTORS)

    def test_negative_selectors_count(self) -> None:
        self.assertEqual(len(NEGATIVE_SELECTORS), 6)
        self.assertIn("N_NONE", NEGATIVE_SELECTORS)
        self.assertIn("N_DIVERSITY", NEGATIVE_SELECTORS)

    def test_groupings_count(self) -> None:
        self.assertEqual(len(GROUPINGS), 12)
        for g in ("G_ONE_GROUP", "G_KMEANS_3", "G_SPATIAL_GRID", "G_GRAPH", "G_CANDIDATE_COMPONENT"):
            self.assertIn(g, GROUPINGS)

    def test_call_and_point_budgets(self) -> None:
        self.assertEqual(CALL_BUDGETS, (1, 3, 5, 10, 20))
        self.assertEqual(POINT_BUDGETS, (5, 10, 20, 50, 100, 150))

    def test_nested_bank_limits(self) -> None:
        self.assertEqual(A_SMALL_LIMITS, {"max_positive": 5, "max_negative": 4, "max_calls": 3})
        self.assertEqual(A_MEDIUM_LIMITS, {"max_positive": 20, "max_negative": 10, "max_calls": 5})
        self.assertEqual(A_LARGE_LIMITS, {"max_positive": 100, "max_negative": 30, "max_calls": 20})

    def test_search_budgets(self) -> None:
        self.assertEqual(SEARCH_BEAM_WIDTHS, (4, 8))
        self.assertEqual(SEARCH_DEPTHS, (10, 20, 50))
        self.assertEqual(SEARCH_CALL_BUDGETS, (1, 3, 5, 10, 20))
        self.assertEqual(SEARCH_POINT_BUDGETS, (10, 20, 50, 100))

    def test_adaptive_value_thresholds(self) -> None:
        self.assertEqual(ADAPTIVE_VALUE_DICE_THRESHOLD, 0.010)
        self.assertEqual(ADAPTIVE_VALUE_MIN_QUERY_FRACTION, 0.25)

    def test_compute_caps(self) -> None:
        self.assertEqual(COMPUTE_CAPS["total_initial_cap"], 132.0)
        self.assertEqual(COMPUTE_CAPS["interim_report_threshold"], 40.0)
        self.assertEqual(COMPUTE_CAPS["reevaluate_search_value_threshold"], 90.0)
        self.assertEqual(COMPUTE_CAPS["greedy_beam_search_oracle"], 40.0)

    def test_branch_and_commit_message(self) -> None:
        self.assertEqual(PHASEX_BRANCH, "phaseX-action-space-oracle-audit")
        self.assertEqual(
            PHASEX_COMMIT_MESSAGE,
            "phaseX: preregister action-space adequacy and search-oracle audit",
        )
        self.assertTrue(PHASEX_NAME.startswith("Phase X"))


class LoadHistoricalArtifactsTests(unittest.TestCase):
    def test_load_raises_on_missing_artifact(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".test_tmp" / "phasex_missing"
        if tmp_root.exists():
            # Wipe the dir so the missing-file path actually triggers.
            for path in tmp_root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(tmp_root.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        tmp_root.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(FileNotFoundError) as ctx:
            load_historical_artifacts(tmp_root)
        self.assertIn("PHASEX_BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE", str(ctx.exception))


class BuildPhaseXPreregistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / ".test_tmp" / "phasex_build"
        if self.tmp_root.exists():
            for path in self.tmp_root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(self.tmp_root.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        _seed_tmp_project(self.tmp_root)

    def test_build_phasex_preregistration_produces_all_artifacts(self) -> None:
        summary = build_phasex_preregistration(self.tmp_root)

        self.assertEqual(summary["phase"], "phaseX")
        self.assertEqual(summary["stage"], "preregistration")
        self.assertEqual(summary["status"], "PREREGISTRATION_ARTIFACTS_READY")
        self.assertEqual(summary["branch"], PHASEX_BRANCH)
        self.assertEqual(summary["commit_message"], PHASEX_COMMIT_MESSAGE)

        # --- 7 preregistration docs ---
        docs_dir = self.tmp_root / "docs"
        for name in (
            "phaseX_goal_read_receipt.md",
            "action_space_problem_statement.md",
            "historical_oracle_correction.md",
            "dense_prompt_scaling_protocol.md",
            "search_oracle_protocol.md",
            "action_space_saturation_protocol.md",
            "formal_test_lock_notice.md",
        ):
            path = docs_dir / name
            self.assertTrue(path.is_file(), f"missing doc: {name}")
            self.assertGreater(path.stat().st_size, 0)

        # --- 4 reports scaffolding ---
        reports_dir = self.tmp_root / "reports"
        self.assertTrue((reports_dir / "phaseX_gate_status.md").is_file())
        self.assertTrue((reports_dir / "phaseX_compute_ledger.csv").is_file())
        self.assertTrue((reports_dir / "phaseX_protocol_snapshot.json").is_file())
        self.assertTrue((reports_dir / "phaseX_input_hashes.json").is_file())

        # --- 8 historical bank audit files ---
        audit_dir = reports_dir / "phaseX_historical_bank_audit"
        for name in (
            "action_bank_exact_definitions.csv",
            "action_bank_exact_definitions.md",
            "fixed_action_absolute_metrics.csv",
            "fixed_action_rankings.csv",
            "old_bank_oracle_query_metrics.csv",
            "old_bank_oracle_summary.md",
            "cost_distribution.csv",
            "historical_claim_correction.md",
            "audit_summary.json",
        ):
            self.assertTrue((audit_dir / name).is_file(), f"missing audit file: {name}")

        # --- 4 live docs were appended to ---
        for path in (
            docs_dir / "protocol_history.md",
            docs_dir / "decision_log.md",
            docs_dir / "data_exposure_log.md",
            self.tmp_root / "checklist.md",
        ):
            self.assertTrue(path.is_file())
            content = path.read_text(encoding="utf-8")
            self.assertIn("Phase X", content)

    def test_protocol_snapshot_has_frozen_grids_and_historical_anchors(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        snapshot_path = self.tmp_root / "reports" / "phaseX_protocol_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(snapshot["phase"], "phaseX")
        self.assertEqual(snapshot["branch"], PHASEX_BRANCH)
        self.assertEqual(snapshot["frozen_grids"]["positive_k"], list(POSITIVE_K_GRID))
        self.assertEqual(snapshot["frozen_grids"]["negative_k"], list(NEGATIVE_K_GRID))
        self.assertEqual(snapshot["frozen_grids"]["positive_selectors"], list(POSITIVE_SELECTORS))
        self.assertEqual(snapshot["frozen_grids"]["negative_selectors"], list(NEGATIVE_SELECTORS))
        self.assertEqual(snapshot["frozen_grids"]["groupings"], list(GROUPINGS))
        self.assertEqual(snapshot["frozen_grids"]["call_budgets"], list(CALL_BUDGETS))
        self.assertEqual(snapshot["frozen_grids"]["point_budgets"], list(POINT_BUDGETS))

        self.assertEqual(snapshot["nested_banks"]["A_small"], A_SMALL_LIMITS)
        self.assertEqual(snapshot["nested_banks"]["A_medium"], A_MEDIUM_LIMITS)
        self.assertEqual(snapshot["nested_banks"]["A_large"], A_LARGE_LIMITS)

        self.assertEqual(
            snapshot["search_oracle"]["beam_widths"], list(SEARCH_BEAM_WIDTHS)
        )
        self.assertEqual(snapshot["search_oracle"]["depths"], list(SEARCH_DEPTHS))

        self.assertEqual(
            snapshot["adaptive_value_thresholds"]["min_dice_gap"],
            ADAPTIVE_VALUE_DICE_THRESHOLD,
        )
        self.assertEqual(
            snapshot["adaptive_value_thresholds"]["min_query_fraction"],
            ADAPTIVE_VALUE_MIN_QUERY_FRACTION,
        )

        anchors = snapshot["historical_anchors"]
        self.assertEqual(anchors["v2_valid_action_count"], 2)
        self.assertEqual(anchors["strongest_fixed_action_id"], "A_top5_one_group")
        self.assertEqual(anchors["historical_phase5_status"], "PIVOT_ORACLE_ANALYSIS")

    def test_input_hashes_record_sha256_for_each_historical_input(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        hashes_path = self.tmp_root / "reports" / "phaseX_input_hashes.json"
        payload = json.loads(hashes_path.read_text(encoding="utf-8"))

        keys = {
            "directive",
            "bank_v1",
            "bank_v2",
            "oracle_v2_summary",
            "fixed_oracle_summary",
            "phase5_decision",
            "gate_f_report",
            "action_utility",
            "fixed_action_summary",
            "budget_pareto",
            "oracle_heatmap",
            "episodes_val",
            "candidate_pilot",
        }
        self.assertEqual(set(payload["input_hashes"].keys()), keys)
        # All inputs except the directive file exist in the tmp project; the
        # directive path is hardcoded to the real D:\ path so it will resolve
        # to a real hash on disk in dev, but the test must only require the
        # remaining keys to be hashed.
        for key, entry in payload["input_hashes"].items():
            if key == "directive":
                continue
            self.assertIsNotNone(entry["sha256"], f"missing sha256 for {key}")
            self.assertEqual(len(entry["sha256"]), 64)

    def test_historical_claim_correction_renames_old_wording(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        correction = (
            self.tmp_root
            / "reports"
            / "phaseX_historical_bank_audit"
            / "historical_claim_correction.md"
        ).read_text(encoding="utf-8")
        self.assertIn("FINITE_BANK_ORACLE_GAP_SMALL", correction)
        self.assertIn("Oracle gap = 0.005 proves adaptive prompting is ineffective", correction)

        docs_correction = (
            self.tmp_root / "docs" / "historical_oracle_correction.md"
        ).read_text(encoding="utf-8")
        self.assertIn("FINITE_BANK_ORACLE_GAP_SMALL", docs_correction)

    def test_audit_summary_records_untested_grid_values(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        audit = json.loads(
            (self.tmp_root / "reports" / "phaseX_historical_bank_audit" / "audit_summary.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(audit["renamed_claim"], "FINITE_BANK_ORACLE_GAP_SMALL")
        self.assertEqual(
            audit["historical_bank"]["untested_positive_k"],
            [10, 20, 30, 50, 75, 100],
        )
        self.assertEqual(
            audit["historical_bank"]["untested_negative_k"], [10, 20, 30, 50]
        )
        self.assertEqual(audit["historical_bank"]["untested_call_budgets"], [10, 20])
        self.assertIn("G_KMEANS_3", audit["historical_bank"]["untested_groupings"])
        self.assertIn("G_GRAPH", audit["historical_bank"]["untested_groupings"])

    def test_compute_ledger_marks_preregistration_as_cpu_only(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        ledger = (self.tmp_root / "reports" / "phaseX_compute_ledger.csv").read_text(
            encoding="utf-8"
        )
        self.assertIn("phaseX-preregistration", ledger)
        self.assertIn("historical_audit", ledger)
        self.assertIn("CPU-only", ledger)

    def test_decision_log_appendix_records_phaseX_decisions(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        decisions = (self.tmp_root / "docs" / "decision_log.md").read_text(encoding="utf-8")
        self.assertIn("FINITE_BANK_ORACLE_GAP_SMALL", decisions)
        self.assertIn("Freeze Phase X grids", decisions)
        self.assertIn("Prohibit PPO/DQN", decisions)

    def test_checklist_marks_preregistration_steps_done(self) -> None:
        build_phasex_preregistration(self.tmp_root)
        checklist = (self.tmp_root / "checklist.md").read_text(encoding="utf-8")
        self.assertIn("Phase X", checklist)
        self.assertIn("PREREGISTRATION_PENDING_COMMIT", checklist)
        # Steps still pending after preregistration
        self.assertIn("Run candidate coverage scaling", checklist)
        self.assertIn("Run GT-guided greedy/beam search oracle", checklist)


if __name__ == "__main__":
    unittest.main()
