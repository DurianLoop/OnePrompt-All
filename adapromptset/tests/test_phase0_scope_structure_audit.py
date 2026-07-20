from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_phase0_scope_structure import (  # noqa: E402
    analyze_phase0_scope_structure,
    write_phase0_scope_structure_audit,
)


class Phase0ScopeStructureAuditTests(unittest.TestCase):
    def _make_project(self, root: Path) -> None:
        for relative in [
            "configs/data",
            "configs/matcher",
            "configs/segmenter",
            "configs/actions",
            "configs/experiments",
            "src/adapromptset/actions",
            "src/adapromptset/candidates",
            "src/adapromptset/data",
            "src/adapromptset/matching",
            "src/adapromptset/policy",
            "src/adapromptset/reporting",
            "src/adapromptset/scripts",
            "src/adapromptset/segmenters",
            "src/adapromptset/statistics",
            "src/adapromptset/training",
            "scripts",
            "tests",
            "manifests",
            "reports",
            "paper",
            "docs",
        ]:
            (root / relative).mkdir(parents=True, exist_ok=True)
        for filename in ["environment.yml", "README.md", "checklist.md"]:
            (root / filename).write_text("placeholder\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "adapromptset-test"',
                    "[project.optional-dependencies]",
                    'dev = ["pytest"]',
                    "[project.scripts]",
                    'adapromptset-test = "adapromptset.scripts.fake:main"',
                    "[tool.setuptools.packages.find]",
                    'where = ["src"]',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "execution.md").write_text(
            "\n".join(
                [
                    "AdaPromptSet studies budget-constrained structured prompt-set selection.",
                    "GF-SAM remains a graph-rule baseline.",
                    "SynPo is a negative-prompt rule baseline.",
                    "AlignSAM uses sequential RL and must be separated from this one-shot setting.",
                    "Point Prompt Defender uses adversarial DQN.",
                    "Do not claim first to learn prompts for SAM.",
                    "RQ1: fixed prompt rules and oracle gap.",
                    "RQ2: policy recovery under the same budget.",
                    "RQ3: prompt allocation conditions.",
                    "H1: action heterogeneity.",
                    "H2: oracle gap.",
                    "H3: policy value.",
                    "H4: oracle recovery.",
                    "H5: cost adaptivity.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "AdaPromptSet_Codex_Response.md").write_text(
            "\n".join(
                [
                    "Current mainline uses offline one-shot set-level utility ranking.",
                    "Excluded: unrestricted coordinate generation.",
                    "Excluded: learned coordinate offsets.",
                    "Excluded: iterative SAM-response-conditioned prompting.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_analysis_accepts_existing_structure_and_records_scope_boundaries(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "phase0_scope_structure"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_project(root)

            summary = analyze_phase0_scope_structure(root)

            self.assertEqual(summary["status"], "PASS_SCOPE_STRUCTURE")
            self.assertEqual(summary["structure"]["status"], "PASS")
            self.assertEqual(summary["scope"]["status"], "PASS")
            self.assertFalse(summary["policy_training_allowed"])
            self.assertTrue(summary["research_question"]["does_not_claim_first_to_learn_prompts"])
            self.assertEqual(summary["rq_hypotheses"]["rq_count"], 3)
            self.assertEqual(summary["rq_hypotheses"]["hypothesis_count"], 5)
            for key in ["gf_sam", "synpo", "alignsam", "ppd"]:
                self.assertTrue(summary["similar_work_boundaries"][key]["present"])
            exclusions = summary["scope"]["non_mainline_exclusions"]
            self.assertTrue(exclusions["sequential_rl"]["present"])
            self.assertTrue(exclusions["continuous_coordinate_generation"]["present"])
            self.assertTrue(exclusions["online_sam_response_refinement"]["present"])
            self.assertEqual(summary["toolchain"]["status"], "PASS_WITH_LINT_DEFERRED")

            report_text = "\n".join(summary["reader_facing_notes"])
            self.assertIn("one-shot", report_text)
            self.assertIn("not deployable policy evidence", report_text)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_writer_emits_json_and_markdown(self) -> None:
        root = PROJECT_ROOT / ".test_tmp" / "phase0_scope_structure_writer"
        if root.exists():
            shutil.rmtree(root)
        try:
            self._make_project(root)

            summary = write_phase0_scope_structure_audit(root)

            self.assertEqual(summary["status"], "PASS_SCOPE_STRUCTURE")
            json_path = root / "reports" / "phase0_scope_structure" / "phase0_scope_structure.json"
            md_path = root / "reports" / "phase0_scope_structure" / "phase0_scope_structure.md"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")
            self.assertEqual(payload["scope"]["status"], "PASS")
            self.assertIn("Phase 0 Scope and Structure Audit", markdown)
            self.assertIn("GF-SAM", markdown)
            self.assertIn("Policy training remains prohibited", markdown)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_real_project_toolchain_has_lint_and_format_config(self) -> None:
        summary = analyze_phase0_scope_structure(PROJECT_ROOT)

        self.assertEqual(summary["toolchain"]["status"], "PASS")
        self.assertTrue(summary["toolchain"]["checks"]["lint_config_present"])


if __name__ == "__main__":
    unittest.main()
