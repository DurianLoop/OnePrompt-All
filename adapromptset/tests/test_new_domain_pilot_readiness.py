from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_new_domain_pilot_readiness import (
    build_new_domain_pilot_readiness,
    write_new_domain_pilot_readiness,
)


class NewDomainPilotReadinessTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_readiness_marks_glas_blocked_but_ddr_infrastructure_ready(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "new_domain_readiness"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._write_json(
                root / "manifests" / "glas_manifest.summary.json",
                {
                    "case_count": 165,
                    "split_counts": {"train": 85, "testA": 60, "testB": 20},
                    "role_counts": {"train": 8, "excluded_overlap": 77, "test": 80},
                    "patient_count": 16,
                },
            )
            self._write_json(
                root / "manifests" / "episodes" / "glas" / "protocol_metadata.json",
                {
                    "role_counts": {"train": 7, "val": 1, "test": 80, "excluded_overlap": 77},
                    "episode_counts": {"train": 7, "val": 1, "test": 80},
                    "leakage_rules": ["Support rows always come from role=train."],
                },
            )
            self._write_json(
                root / "reports" / "candidate_pilot_glas" / "candidate_pilot_summary.json",
                {"status": "PASS", "episode_count": 1, "formal_gate_c": "PASS"},
            )
            self._write_json(
                root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json",
                {
                    "status": "PENDING_INSUFFICIENT_EPISODES",
                    "episode_count": 1,
                    "gate_min_episodes": 10,
                    "dice_oracle_gap": 0.0,
                    "gate_f_min_dice_gap": 0.01,
                },
            )
            self._write_json(
                root / "reports" / "ddr_protocol_audit" / "ddr_protocol_audit.json",
                {
                    "status": "PASS_PROTOCOL_DEFINED",
                    "stage1_union": {
                        "mask_rule": "binary_union_of_EX_HE_MA_SE",
                        "allowed_use": "infrastructure_only",
                    },
                    "stage2_four_intents": {"intents": ["EX", "HE", "MA", "SE"]},
                    "formal_policy_training_allowed": False,
                },
            )

            summary = build_new_domain_pilot_readiness(root)

            self.assertEqual(summary["status"], "READY_WITH_BLOCKED_FORMAL_DATASETS")
            glas = summary["datasets"]["glas"]
            self.assertEqual(glas["status"], "BLOCKED_INSUFFICIENT_EPISODES")
            self.assertEqual(glas["gate"]["episode_count"], 1)
            self.assertEqual(glas["gate"]["gate_min_episodes"], 10)
            self.assertFalse(glas["formal_policy_training_allowed"])
            self.assertIn("predeclare_glas_trainval_diagnostic_without_official_test", glas["allowed_next_actions"])
            self.assertIn("promote_glas_smoke_to_gate_f", glas["prohibited_next_actions"])

            ddr = summary["datasets"]["ddr"]
            self.assertEqual(ddr["status"], "READY_FOR_INFRASTRUCTURE_ONLY")
            self.assertEqual(ddr["stage1_mask_rule"], "binary_union_of_EX_HE_MA_SE")
            self.assertFalse(ddr["formal_policy_training_allowed"])
            self.assertIn("ddr_stage1_union_loader_smoke", ddr["allowed_next_actions"])

            self.assertEqual(summary["datasets"]["idrid"]["status"], "READ_ONLY_NON_TUNABLE")
            self.assertEqual(summary["datasets"]["cpm17"]["status"], "DEFERRED_MISSING_OR_INCOMPLETE")
            self.assertIn("monusac_policy_training", summary["global_prohibited_next_actions"])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_writer_emits_json_and_markdown(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "new_domain_readiness_writer"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._write_json(
                root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json",
                {"status": "PENDING_INSUFFICIENT_EPISODES", "episode_count": 1, "gate_min_episodes": 10},
            )
            outputs = write_new_domain_pilot_readiness(root)

            self.assertTrue(outputs["json"].exists())
            self.assertTrue(outputs["markdown"].exists())
            payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            text = outputs["markdown"].read_text(encoding="utf-8")
            self.assertEqual(payload["datasets"]["glas"]["status"], "BLOCKED_INSUFFICIENT_EPISODES")
            self.assertIn("GlaS", text)
            self.assertIn("BLOCKED_INSUFFICIENT_EPISODES", text)
            self.assertIn("DDR", text)
            self.assertIn("READ_ONLY_NON_TUNABLE", text)
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
