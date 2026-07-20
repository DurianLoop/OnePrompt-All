import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.reporting.gate_reports import build_standard_gate_reports


class GateReportExportTests(unittest.TestCase):
    def _project_root(self, name: str) -> Path:
        base = Path.cwd() / ".test_tmp" / "gate_reports"
        base.mkdir(parents=True, exist_ok=True)
        root = base / name
        root.mkdir(exist_ok=True)
        return root

    def _write_json(self, root: Path, relative: str, payload: dict) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_text(self, root: Path, relative: str, text: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _seed_minimal_gate_artifacts(self, root: Path) -> None:
        self._write_json(
            root,
            "reports/dataset_audit/monusac.json",
            {"gate": {"name": "GATE-A/MoNuSAC", "passed": True, "issues": []}},
        )
        self._write_json(
            root,
            "reports/dataset_audit/monuseg.json",
            {"gate": {"name": "GATE-A/MoNuSeg", "passed": True, "issues": []}},
        )
        self._write_text(
            root,
            "reports/gate_b_status.md",
            "| Gate | Status | Issues |\n|---|---|---|\n| GATE-B/monusac | PASS | none |\n",
        )
        self._write_json(
            root,
            "reports/candidate_pilot/candidate_pilot_summary.json",
            {
                "status": "PASS",
                "formal_gate_c": "PASS",
                "mean_precision": 0.52,
                "mean_random_precision": 0.16,
            },
        )
        self._write_json(
            root,
            "reports/sam_smoke/sam_smoke_summary.json",
            {"status": "PASS", "formal_gate_d": "PASS", "component_dice": 0.9382},
        )
        self._write_json(
            root,
            "reports/action_bank/action_bank_summary.json",
            {
                "name": "prompt_set_bank_v2",
                "bank_hash": "abc123",
                "action_count": 13,
                "valid_action_count": 12,
            },
        )
        self._write_json(
            root,
            "reports/oracle_gap_pilot_v2/oracle_gap_pilot_summary.json",
            {
                "gate_f_status": "FAIL_ORACLE_GAP_SMALL",
                "status": "FAIL_ORACLE_GAP_SMALL",
                "episode_count": 10,
                "record_count": 120,
                "mean_oracle_dice": 0.3762,
                "mean_best_fixed_dice": 0.3712,
                "dice_oracle_gap": 0.005011,
                "aji_available": True,
                "mean_oracle_aji": 0.214,
                "mean_best_fixed_aji": 0.209,
                "aji_oracle_gap": 0.005,
                "gate_f_min_dice_gap": 0.01,
                "top_oracle_action_fraction": 0.5,
                "best_fixed_action": "B_top5_pos3neg",
            },
        )
        self._write_json(
            root,
            "reports/decision_checkpoint/phase5_decision.json",
            {
                "phase5_status": "PIVOT_ORACLE_ANALYSIS",
                "allowed_next_actions": ["fixed_rule_action_oracle_analysis", "predeclared_new_domain_pilot"],
                "prohibited_next_actions": ["monusac_policy_training", "post_hoc_monusac_action_bank_redesign"],
            },
        )

    def test_build_standard_reports_writes_gate_json_and_markdown(self) -> None:
        root = self._project_root("standard_reports")
        self._seed_minimal_gate_artifacts(root)

        written = build_standard_gate_reports(root)

        gate_a = root / "reports/gates/gate_a_data_integrity.json"
        gate_f = root / "reports/gates/gate_f_oracle_gap.json"
        gate_f_md = root / "reports/gates/gate_f_oracle_gap.md"
        self.assertIn(gate_a, written)
        self.assertIn(gate_f, written)
        self.assertIn(gate_f_md, written)

        payload = json.loads(gate_f.read_text(encoding="utf-8"))
        self.assertEqual(payload["gate"], "F")
        self.assertEqual(payload["status"], "FAIL_ORACLE_GAP_SMALL")
        self.assertEqual(payload["pass_fail"], "FAIL")
        self.assertEqual(payload["diagnostics"]["phase5_status"], "PIVOT_ORACLE_ANALYSIS")
        self.assertTrue(payload["diagnostics"]["aji_available"])
        self.assertAlmostEqual(payload["diagnostics"]["aji_oracle_gap"], 0.005)
        self.assertTrue(payload["input_hashes"])
        self.assertIn("scripts/run_oracle_gap_pilot.py", "\n".join(payload["commands"]))
        self.assertIn("monusac_policy_training", payload["prohibited_next_actions"])
        self.assertIn("fixed_rule_action_oracle_analysis", payload["allowed_next_actions"])

        text = gate_f_md.read_text(encoding="utf-8")
        self.assertIn("# Gate F: Oracle Gap", text)
        self.assertIn("FAIL_ORACLE_GAP_SMALL", text)
        self.assertIn("Prohibited Next Actions", text)

    def test_policy_gates_are_marked_prohibited_after_gate_f_pivot(self) -> None:
        root = self._project_root("policy_prohibited")
        self._seed_minimal_gate_artifacts(root)

        build_standard_gate_reports(root)

        gate_h = json.loads(
            (root / "reports/gates/gate_h_policy_learning.json").read_text(encoding="utf-8")
        )
        self.assertEqual(gate_h["status"], "PROHIBITED_BY_GATE_F_PIVOT")
        self.assertEqual(gate_h["pass_fail"], "NOT_RUN")
        self.assertIn("monusac_policy_training", gate_h["prohibited_next_actions"])


if __name__ == "__main__":
    unittest.main()
