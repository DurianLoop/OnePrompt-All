from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_paper_sections import export_paper_section_audit  # noqa: E402


class PaperSectionAuditTests(unittest.TestCase):
    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _seed_sections(self, root: Path, *, unsafe_results: str = "") -> None:
        sections = root / "paper" / "sections"
        tables = root / "paper" / "tables"
        self._write_text(
            sections / "00_abstract.tex",
            (
                r"\begin{abstract}"
                "Current evidence supports an oracle-analysis-only report after a Gate-F pivot. "
                "This is not an adaptive-policy result and contains no performance claim. "
                r"\end{abstract}"
            ),
        )
        self._write_text(
            sections / "01_intro.tex",
            (
                "AdaPromptSet studies budget-constrained structured prompt-set selection. "
                "GF-SAM, SynPo, AlignSAM, and Point Prompt Defender define the nearest boundaries, "
                "so this draft does not claim to be first to learn prompts."
            ),
        )
        self._write_text(
            sections / "02_related.tex",
            "GF-SAM, SynPo, AlignSAM, and Point Prompt Defender are separated in a related-work table.",
        )
        self._write_text(
            sections / "03_method.tex",
            (
                "The method freezes DINOv2-L and SAM-B and evaluates predeclared prompt-set actions. "
                "It is an offline action-bank analysis rather than end-to-end training. "
                "Figure 1 distinguishes frozen matcher, frozen segmenter, learned utility predictor, "
                "train/validation only utility labels, and the test-time safe path. "
                "Figure 2 shows the current v2 MVP action-bank subset as complete prompt-set proposals "
                "with SAM calls and prompt points; KMeans, density, and GF-SAM-style graph families are deferred."
            ),
        )
        self._write_text(
            sections / "04_experiments.tex",
            (
                "Statistical reporting treats query/case, and patient clusters where available, as the "
                "unit of analysis. Multi-reference episodes are repeated measurements and not independent samples."
            ),
        )
        self._write_text(
            sections / "05_results.tex",
            (
                "Results report Dice accuracy together with cost: mean SAM calls, prompt counts, runtime, "
                "and GPU-hour ledger status. Oracle rows are analysis upper bounds, not deployable method outputs. "
                f"{unsafe_results}"
            ),
        )
        self._write_text(
            sections / "06_limitations.tex",
            (
                "Limitations include the MVP action bank upper bound, sensitivity to reference quality, "
                "frozen matcher errors that cannot be recovered by the current selector, and the utility generation cost "
                "of running the action bank."
            ),
        )
        self._write_text(
            sections / "07_conclusion.tex",
            "The conclusion is oracle-analysis-only and does not report adaptive policy success.",
        )
        self._write_text(
            tables / "table_oracle_pivot.tex",
            (
                r"\caption{Gate-F pivot status from real pilot artifacts. "
                r"Oracle columns are analysis upper bounds, not deployable method outputs; "
                r"no privileged-input row is included.}"
            ),
        )
        self._write_text(
            tables / "table_action_summary.tex",
            (
                r"\caption{MoNuSAC action-oracle summary after the Gate-F pivot. "
                r"Oracle wins are post-hoc analysis labels, not deployable method selections; "
                r"no privileged-input row is included.}"
            ),
        )
        self._write_text(
            tables / "table_representation_pivot.tex",
            (
                r"\caption{Representation diagnostics for the Gate-F pivot. "
                r"The row uses fixed artifact diagnostics only; no privileged-input row is included.}"
            ),
        )

    def test_paper_section_audit_passes_when_sections_cover_boundaries(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_paper_section_audit_pass"
        self._seed_sections(root)

        outputs = export_paper_section_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["checks"]["abstract_claim_free"])
        self.assertTrue(payload["checks"]["related_work_boundaries_present"])
        self.assertTrue(payload["checks"]["method_no_end_to_end_differentiable"])
        self.assertTrue(payload["checks"]["method_figure_frozen_learned_train_only"])
        self.assertTrue(payload["checks"]["method_action_bank_figure_current_subset"])
        self.assertTrue(payload["checks"]["experiments_statistical_units_present"])
        self.assertTrue(payload["checks"]["results_cost_and_accuracy_present"])
        self.assertTrue(payload["checks"]["oracle_privileged_caption_boundaries"])
        self.assertTrue(payload["checks"]["limitations_action_bank_upper_bound"])
        self.assertTrue(payload["checks"]["limitations_reference_quality"])
        self.assertTrue(payload["checks"]["limitations_frozen_matcher_irrecoverable"])
        self.assertTrue(payload["checks"]["limitations_utility_generation_cost"])
        self.assertTrue(payload["checks"]["conclusion_oracle_analysis_only"])
        self.assertFalse(payload["violations"])

        markdown = outputs["markdown"].read_text(encoding="utf-8")
        self.assertIn("Status: `PASS`", markdown)
        self.assertIn("Limitations reference quality", markdown)
        self.assertIn("Results cost and accuracy", markdown)
        self.assertIn("Oracle and privileged caption boundaries", markdown)

    def test_paper_section_audit_fails_on_unsafe_claim_and_missing_limitations(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_paper_section_audit_fail"
        self._seed_sections(root, unsafe_results="The policy improves segmentation and is state of the art.")
        (root / "paper" / "sections" / "06_limitations.tex").write_text(
            "Limitations only mention that the pilot is small.",
            encoding="utf-8",
        )

        outputs = export_paper_section_audit(root)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "FAIL")
        tokens = {(item["check"], item.get("token")) for item in payload["violations"]}
        self.assertIn(("prohibited_claim_token", "policy improves"), tokens)
        self.assertIn(("prohibited_claim_token", "state of the art"), tokens)
        self.assertIn(("limitations_action_bank_upper_bound", None), tokens)
        self.assertIn(("limitations_reference_quality", None), tokens)


if __name__ == "__main__":
    unittest.main()
