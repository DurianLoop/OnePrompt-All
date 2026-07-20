from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_scientific_claims import export_scientific_claim_audit


class ScientificClaimAuditTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _seed_project(self, root: Path, *, unsafe_text: str = "") -> None:
        self._write_json(
            root / "reports" / "final_project_report.json",
            {
                "status": "PIVOT_ORACLE_ANALYSIS",
                "claim_boundary": "oracle_analysis_only",
                "gate_f": {
                    "status": "FAIL_ORACLE_GAP_SMALL",
                    "dice_oracle_gap": 0.0050113301616486194,
                    "min_required_gap": 0.01,
                },
                "fixed_oracle": {
                    "best_fixed_action_id": "B_top5_pos3neg",
                    "deterministic_promotion_status": "NOT_PROMOTED",
                },
                "representation": {
                    "status": "PIVOT_REPRESENTATION_ANALYSIS",
                    "claim": "oracle_analysis_only_insufficient_separation",
                },
                "external_pilots": {
                    "glas": {"status": "PENDING_INSUFFICIENT_EPISODES"}
                },
                "statistics": {"status": "PENDING_INSUFFICIENT_CLUSTERS"},
                "pdf": {"sha256": "ABCDEF" * 10 + "ABCD"},
                "prohibited_next_actions": [
                    "monusac_policy_training",
                    "posthoc_monusac_action_redesign",
                    "test_time_action_search",
                ],
            },
        )
        self._write_json(
            root / "reports" / "gates" / "gate_f_oracle_gap.json",
            {
                "status": "FAIL_ORACLE_GAP_SMALL",
                "pass_fail": "FAIL",
                "diagnostics": {
                    "dice_oracle_gap": 0.0050113301616486194,
                    "gate_f_min_dice_gap": 0.01,
                },
                "prohibited_next_actions": ["monusac_policy_training"],
            },
        )
        self._write_json(
            root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json",
            {
                "recommended_claim": "oracle_analysis_only",
                "best_fixed_action_id": "B_top5_pos3neg",
                "deterministic_promotion_status": "NOT_PROMOTED",
            },
        )
        self._write_json(
            root / "reports" / "representation_analysis" / "representation_summary.json",
            {
                "status": "PIVOT_REPRESENTATION_ANALYSIS",
                "representation_claim": "oracle_analysis_only_insufficient_separation",
                "prohibited_next_actions": [
                    "monusac_policy_training",
                    "posthoc_monusac_action_redesign",
                    "test_time_action_search",
                ],
            },
        )
        safe_claim_text = (
            "Current status is PIVOT_ORACLE_ANALYSIS. "
            "Gate F failed with FAIL_ORACLE_GAP_SMALL, so policy training on MoNuSAC "
            "is prohibited. The claim boundary is oracle_analysis_only. "
            "GlaS is PENDING_INSUFFICIENT_EPISODES and bootstrap is "
            "PENDING_INSUFFICIENT_CLUSTERS. "
            "Oracle rows are analysis upper bounds, not deployable method outputs. "
            "Background reference evidence is not explicit SAM negative point prompting. "
            "Multi-reference episodes are repeated measurements and not independent samples. "
            "No in-domain retraining or transfer claim is made. "
            "Failure results and limitations are disclosed. "
        )
        self._write_text(root / "README.md", safe_claim_text + unsafe_text)
        self._write_text(root / "reports" / "final_project_report.md", safe_claim_text)
        self._write_text(root / "paper" / "main.tex", safe_claim_text)
        self._write_text(root / "paper" / "sections" / "05_results.tex", safe_claim_text)

    def test_claim_audit_passes_for_oracle_analysis_boundary(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_scientific_claim_audit_pass"
        self._seed_project(root)

        outputs = export_scientific_claim_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["claim_boundary"], "oracle_analysis_only")
        self.assertEqual(payload["gate_f_status"], "FAIL_ORACLE_GAP_SMALL")
        self.assertIn("reports/final_project_report.json", payload["evidence_artifacts"])
        self.assertIn("paper/main.tex", payload["checked_files"])
        self.assertTrue(payload["required_evidence_checks"]["gate_f_pivot_documented"])
        self.assertTrue(payload["required_evidence_checks"]["policy_training_prohibited"])
        self.assertTrue(payload["required_evidence_checks"]["oracle_not_deployable"])
        self.assertTrue(payload["required_evidence_checks"]["background_reference_not_explicit_negative"])
        self.assertTrue(payload["required_evidence_checks"]["multi_reference_not_independent"])
        self.assertTrue(payload["required_evidence_checks"]["no_in_domain_transfer_claim"])
        self.assertTrue(payload["required_evidence_checks"]["failures_and_limitations_disclosed"])
        self.assertFalse(payload["violations"])

        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Status: `PASS`", markdown)
        self.assertIn("oracle_analysis_only", markdown)
        self.assertIn("Gate F pivot documented", markdown)
        self.assertIn("Oracle not deployable", markdown)
        self.assertIn("Multi reference not independent", markdown)

    def test_claim_audit_fails_on_prohibited_claims(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_scientific_claim_audit_fail"
        self._seed_project(
            root,
            unsafe_text=(
                " AdaPromptSet is state of the art and policy improves segmentation "
                "with model-agnostic generalization."
            ),
        )

        outputs = export_scientific_claim_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FAIL")
        matched = {(item["file"], item["token"]) for item in payload["violations"]}
        self.assertIn(("README.md", "state of the art"), matched)
        self.assertIn(("README.md", "policy improves"), matched)
        self.assertIn(("README.md", "model-agnostic"), matched)
        self.assertIn(("README.md", "generalization"), matched)


if __name__ == "__main__":
    unittest.main()
