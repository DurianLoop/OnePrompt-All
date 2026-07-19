from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.audit_glas_phase2_gate_c import analyze_gate_c


class GlasPhase2GateCAuditTests(unittest.TestCase):
    def test_passes_when_query_aggregated_thresholds_pass(self) -> None:
        rows = []
        for query in range(2):
            for ref in range(3):
                for limit, coverage in [(1, 0.30), (3, 0.55), (5, 0.70)]:
                    rows.append(
                        {
                            "episode_id": f"q{query}_r{ref}",
                            "query_case_id": f"q{query}",
                            "positive_limit": str(limit),
                            "point_precision": "0.80",
                            "random_precision": "0.40",
                            "component_coverage": str(coverage),
                            "negative_blank_fraction": "0.10",
                            "validation_issue_count": "0",
                        }
                    )

        summary = analyze_gate_c(rows, expected_query_count=2, references_per_query=3)

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["decision"], "CONTINUE_TO_GATE_D")
        self.assertTrue(summary["checks"]["precision_enrichment"])
        self.assertTrue(summary["checks"]["component_coverage"])

    def test_fails_to_redesign_when_coverage_is_low(self) -> None:
        rows = []
        for query in range(2):
            for ref in range(3):
                for limit, coverage in [(1, 0.10), (3, 0.20), (5, 0.25)]:
                    rows.append(
                        {
                            "episode_id": f"q{query}_r{ref}",
                            "query_case_id": f"q{query}",
                            "positive_limit": str(limit),
                            "point_precision": "0.90",
                            "random_precision": "0.30",
                            "component_coverage": str(coverage),
                            "negative_blank_fraction": "0.10",
                            "validation_issue_count": "0",
                        }
                    )

        summary = analyze_gate_c(rows, expected_query_count=2, references_per_query=3)

        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["decision"], "REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED")
        self.assertFalse(summary["checks"]["component_coverage"])
        self.assertIn("component coverage", summary["issues"][0])


if __name__ == "__main__":
    unittest.main()
