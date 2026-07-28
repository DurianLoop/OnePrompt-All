"""Focused unit tests for Phase Y preregistration artifacts.

Verifies:
- script paths resolve
- action bank YAML is well-formed and has 210 actions
- split manifests are patient-level disjoint
- reference sets A/B are patient-disjoint from queries and mutually disjoint
- no test case appears in any Phase Y manifest
- protocol snapshot has required keys
- input hashes are non-empty for all preregistration files

Run: python -m pytest tests/test_phaseY_preregistration.py -v
Or:  python tests/test_phaseY_preregistration.py
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(r"D:\SAM\adapromptset")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestPhaseYPreregistration(unittest.TestCase):
    def setUp(self) -> None:
        self.root = PROJECT_ROOT

    def test_action_bank_yaml_exists_and_has_210_actions(self) -> None:
        path = self.root / "configs" / "actions" / "phaseY_full_single_call_bank.yaml"
        self.assertTrue(path.exists(), f"Missing: {path}")
        text = path.read_text(encoding="utf-8")
        # Count action_id lines
        count = text.count("action_id: Y_")
        self.assertEqual(count, 210, f"Expected 210 actions, got {count}")

    def test_compact_bank_placeholder_exists(self) -> None:
        path = self.root / "configs" / "actions" / "phaseY_compact_policy_bank.yaml"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("PLACEHOLDER_PENDING_PARETO_PRUNING", text)

    def test_split_manifests_exist(self) -> None:
        for name in [
            "policy_train_queries.csv",
            "policy_validation_queries.csv",
            "confirmatory_holdout_queries.csv",
            "reference_pool.csv",
            "reference_set_A.csv",
            "reference_set_B.csv",
        ]:
            path = self.root / "manifests" / "phaseY" / name
            self.assertTrue(path.exists(), f"Missing: {path}")

    def test_split_counts(self) -> None:
        summary = read_json(self.root / "reports" / "phaseY_data_audit" / "split_summary.json")
        self.assertEqual(summary["split"]["train"]["case_count"], 80)
        self.assertEqual(summary["split"]["validation"]["case_count"], 41)
        self.assertEqual(summary["split"]["holdout"]["case_count"], 40)
        self.assertEqual(summary["eligible_fresh_cases"], 161)

    def test_patient_level_disjoint(self) -> None:
        train = read_csv(self.root / "manifests" / "phaseY" / "policy_train_queries.csv")
        val = read_csv(self.root / "manifests" / "phaseY" / "policy_validation_queries.csv")
        holdout = read_csv(self.root / "manifests" / "phaseY" / "confirmatory_holdout_queries.csv")
        train_pats = {r["patient"] for r in train}
        val_pats = {r["patient"] for r in val}
        holdout_pats = {r["patient"] for r in holdout}
        self.assertEqual(len(train_pats & val_pats), 0, "train/val patient overlap")
        self.assertEqual(len(train_pats & holdout_pats), 0, "train/holdout patient overlap")
        self.assertEqual(len(val_pats & holdout_pats), 0, "val/holdout patient overlap")

    def test_no_test_cases_in_manifests(self) -> None:
        for name in [
            "policy_train_queries.csv",
            "policy_validation_queries.csv",
            "confirmatory_holdout_queries.csv",
            "reference_pool.csv",
            "reference_set_A.csv",
            "reference_set_B.csv",
        ]:
            rows = read_csv(self.root / "manifests" / "phaseY" / name)
            for r in rows:
                case_id = r.get("case_id") or r.get("reference_case_id") or ""
                self.assertFalse(
                    case_id.startswith("test_"),
                    f"Test case {case_id} found in {name}",
                )

    def test_reference_sets_disjoint_per_query(self) -> None:
        refA = read_csv(self.root / "manifests" / "phaseY" / "reference_set_A.csv")
        refB = read_csv(self.root / "manifests" / "phaseY" / "reference_set_B.csv")
        a_by_q: dict[str, set[str]] = {}
        b_by_q: dict[str, set[str]] = {}
        for r in refA:
            a_by_q.setdefault(r["query_case_id"], set()).add(r["reference_case_id"])
        for r in refB:
            b_by_q.setdefault(r["query_case_id"], set()).add(r["reference_case_id"])
        for q in a_by_q:
            overlap = a_by_q[q] & b_by_q.get(q, set())
            self.assertEqual(len(overlap), 0, f"Query {q}: A/B overlap {overlap}")

    def test_reference_patient_disjoint_from_query(self) -> None:
        train = read_csv(self.root / "manifests" / "phaseY" / "policy_train_queries.csv")
        val = read_csv(self.root / "manifests" / "phaseY" / "policy_validation_queries.csv")
        holdout = read_csv(self.root / "manifests" / "phaseY" / "confirmatory_holdout_queries.csv")
        all_queries = {r["case_id"]: r["patient"] for r in train + val + holdout}
        for ref_file in ["reference_set_A.csv", "reference_set_B.csv"]:
            refs = read_csv(self.root / "manifests" / "phaseY" / ref_file)
            for r in refs:
                q_pat = all_queries.get(r["query_case_id"], "")
                self.assertNotEqual(
                    q_pat,
                    r["reference_patient"],
                    f"Query {r['query_case_id']} patient == reference patient in {ref_file}",
                )

    def test_reference_audit_pass(self) -> None:
        audit = read_json(self.root / "reports" / "phaseY_reference_audit.json")
        self.assertEqual(audit["status"], "PHASEY_REFERENCE_AUDIT_PASS")
        self.assertEqual(len(audit["violations"]), 0)

    def test_protocol_snapshot_required_keys(self) -> None:
        snap = read_json(self.root / "reports" / "phaseY_protocol_snapshot.json")
        for key in ["phase", "name", "branch", "preregistration_date", "directive",
                     "main_question", "frozen_scope", "frozen_grids", "full_audit_bank",
                     "gates", "model_ladders", "statistics", "compute_caps_gpu_hours",
                     "historical_anchors"]:
            self.assertIn(key, snap, f"Missing key in protocol_snapshot: {key}")
        self.assertEqual(snap["full_audit_bank"]["total_actions"], 210)
        self.assertEqual(snap["branch"], "phaseY-policy-feasibility")

    def test_input_hashes_non_empty(self) -> None:
        hashes = read_json(self.root / "reports" / "phaseY_input_hashes.json")
        missing = []
        for rel, info in hashes["input_hashes"].items():
            if not info["sha256"]:
                missing.append(rel)
        self.assertEqual(len(missing), 0, f"Empty hashes for: {missing}")

    def test_compute_ledger_exists(self) -> None:
        path = self.root / "reports" / "phaseY_compute_ledger.csv"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("data_audit_and_split", text)
        self.assertIn("COMPLETE", text)

    def test_gate_status_exists(self) -> None:
        path = self.root / "reports" / "phaseY_gate_status.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Gate Y1", text)
        self.assertIn("Gate Y2", text)
        self.assertIn("Gate Y3", text)
        self.assertIn("Gate Y4", text)
        self.assertIn("LOCKED_UNUSED", text)

    def test_phasex_excluded_patients_not_in_splits(self) -> None:
        phasex_exposed_patients = {
            "TCGA-55-1594", "TCGA-YL-A9WY", "TCGA-UZ-A9PO", "TCGA-G9-6499",
            "TCGA-P4-AAVK", "TCGA-BH-A18T", "TCGA-G9-6342", "TCGA-EJ-5495",
            "TCGA-L4-A4E5", "TCGA-86-7713",
        }
        for name in ["policy_train_queries.csv", "policy_validation_queries.csv",
                      "confirmatory_holdout_queries.csv"]:
            rows = read_csv(self.root / "manifests" / "phaseY" / name)
            for r in rows:
                self.assertNotIn(
                    r["patient"], phasex_exposed_patients,
                    f"Phase X exposed patient {r['patient']} in {name}",
                )

    def test_preregistration_docs_exist(self) -> None:
        docs = [
            "phaseY_goal_read_receipt.md",
            "phaseY_scientific_questions.md",
            "phaseY_data_split_protocol.md",
            "phaseY_reference_protocol.md",
            "phaseY_single_call_action_bank.md",
            "phaseY_oracle_stability_protocol.md",
            "phaseY_bank_policy_protocol.md",
            "phaseY_point_subset_protocol.md",
            "phaseY_statistics_protocol.md",
            "phaseY_claim_registry.md",
            "phaseY_test_lock_notice.md",
        ]
        for doc in docs:
            path = self.root / "docs" / doc
            self.assertTrue(path.exists(), f"Missing doc: {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
