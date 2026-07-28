from __future__ import annotations

# ruff: noqa: E402
import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_glas_gate_d_confirmation import (
    IMAGE_SIZE,
    _assert_no_test_access,
    _bootstrap,
    _final_state,
    _mismatched_control_spec,
    _prompt_spec_from_mask,
    _random_control_spec,
    run_confirmation,
)


def toy_mask() -> list[list[bool]]:
    mask = [[False for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
    for y in range(10, 16):
        for x in range(10, 18):
            mask[y][x] = True
    for y in range(40, 46):
        for x in range(60, 66):
            mask[y][x] = True
    for y in range(90, 95):
        for x in range(120, 124):
            mask[y][x] = True
    return mask


def spec_cost(spec: dict) -> tuple[int, int, int]:
    if spec["kind"] == "box":
        return 4, 1, 1
    prompt_count = sum(len(group) for group in spec["groups"])
    group_count = len(spec["groups"])
    call_count = group_count
    return prompt_count, group_count, call_count


class GlasGateDConfirmationTests(unittest.TestCase):
    def test_prompt_specs_follow_registered_families(self) -> None:
        mask = toy_mask()

        p1 = _prompt_spec_from_mask(mask, "P1_largest_component_centroid")
        p2 = _prompt_spec_from_mask(mask, "P2_fixed_multi_centroid_one_group")
        p3 = _prompt_spec_from_mask(mask, "P3_component_grouped_centroids")
        p4 = _prompt_spec_from_mask(mask, "P4_union_mask_bounding_box")
        p5 = _prompt_spec_from_mask(mask, "P5_positive_plus_controlled_negative")

        self.assertEqual(spec_cost(p1), (1, 1, 1))
        self.assertEqual(spec_cost(p2), (3, 1, 1))
        self.assertEqual(spec_cost(p3), (3, 3, 3))
        self.assertEqual(spec_cost(p4), (4, 1, 1))
        self.assertEqual(spec_cost(p5), (2, 1, 1))
        self.assertEqual(p5["labels"], [[1, 0]])

    def test_controls_are_cost_matched_to_privileged_specs(self) -> None:
        mask = toy_mask()
        reference_mask = toy_mask()

        for family in (
            "P1_largest_component_centroid",
            "P2_fixed_multi_centroid_one_group",
            "P3_component_grouped_centroids",
            "P4_union_mask_bounding_box",
            "P5_positive_plus_controlled_negative",
        ):
            privileged = _prompt_spec_from_mask(mask, family)
            random_control = _random_control_spec(privileged, query_case_id="train_1", family=family)
            mismatched_control = _mismatched_control_spec(reference_mask, privileged, family)
            self.assertEqual(spec_cost(random_control), spec_cost(privileged))
            self.assertEqual(spec_cost(mismatched_control), spec_cost(privileged))

    def test_manifest_guard_rejects_test_and_non_train_queries(self) -> None:
        valid = [{"episode_id": "ok", "query_source_split": "train", "query_image_path": "train_1.bmp"}]
        _assert_no_test_access(valid)

        with self.assertRaises(ValueError):
            _assert_no_test_access(
                [{"episode_id": "bad", "query_source_split": "train", "query_image_path": "testA_1.bmp"}]
            )
        with self.assertRaises(ValueError):
            _assert_no_test_access(
                [{"episode_id": "bad", "query_source_split": "test", "query_image_path": "train_1.bmp"}]
            )

    def test_final_state_selection_is_predeclared(self) -> None:
        samb_pass = {
            "status": "PASS",
            "p3_mean_privileged_dice": 0.70,
            "p3_catastrophic_failure_rate": 0.10,
        }
        saml_pass = {
            "status": "PASS",
            "p3_mean_privileged_dice": 0.72,
            "p3_catastrophic_failure_rate": 0.08,
        }
        saml_clear = {
            "status": "PASS",
            "p3_mean_privileged_dice": 0.74,
            "p3_catastrophic_failure_rate": 0.04,
        }
        failed = {
            "status": "FAIL",
            "p3_mean_privileged_dice": 0.20,
            "p3_catastrophic_failure_rate": 0.40,
        }

        self.assertEqual(_final_state({"SAM-B": samb_pass, "SAM-L": failed}), "PHASE2_GATE_D_PASS_SAMB")
        self.assertEqual(_final_state({"SAM-B": failed, "SAM-L": saml_pass}), "PHASE2_GATE_D_PASS_SAML")
        self.assertEqual(
            _final_state({"SAM-B": samb_pass, "SAM-L": saml_pass}),
            "PHASE2_GATE_D_PASS_BOTH_FREEZE_SAMB",
        )
        self.assertEqual(
            _final_state({"SAM-B": samb_pass, "SAM-L": saml_clear}),
            "PHASE2_GATE_D_PASS_BOTH_FREEZE_SAML",
        )
        self.assertEqual(
            _final_state({"SAM-B": failed, "SAM-L": failed}),
            "STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED",
        )

    def test_bootstrap_is_deterministic_and_positive_for_positive_margins(self) -> None:
        first = _bootstrap([0.05, 0.10, 0.15], iterations=50, seed=123)
        second = _bootstrap([0.05, 0.10, 0.15], iterations=50, seed=123)

        self.assertEqual(first, second)
        self.assertGreater(first["mean"], 0.0)
        self.assertGreater(first["ci_high"], 0.0)

    def test_dry_run_writes_summary_without_sam(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_run_glas_gate_d_confirmation"
        root.mkdir(parents=True, exist_ok=True)
        episodes_csv = root / "episodes_val.csv"
        candidate_csv = root / "candidate_pilot_candidates.csv"
        out_dir = root / "out"
        rows = [
            {
                "episode_id": f"glas_phase2_val_train_{idx}_ref1_train_0",
                "query_case_id": f"train_{idx}",
                "query_source_split": "train",
                "query_image_path": f"train_{idx}.bmp",
                "query_mask_path": f"train_{idx}_anno.bmp",
            }
            for idx in range(30)
        ]
        with episodes_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        candidate_csv.write_text("episode_id,positive_limit,kind,x,y\n", encoding="utf-8")

        exit_code = run_confirmation(
            project_root=PROJECT_ROOT,
            out_dir=out_dir,
            episodes_csv=episodes_csv,
            candidate_csv=candidate_csv,
            dry_run=True,
        )

        self.assertEqual(exit_code, 0)
        summary = json.loads((out_dir / "gate_d_confirmation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "DRY_RUN")
        self.assertEqual(summary["query_count"], 30)


if __name__ == "__main__":
    unittest.main()
