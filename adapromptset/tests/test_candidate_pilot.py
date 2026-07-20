from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.evaluation.gate_c import summarize_candidate_pilot


class CandidatePilotGateTests(unittest.TestCase):
    def test_gate_c_passes_when_precision_beats_random_and_coverage_is_monotonic(self) -> None:
        rows = [
            {"episode_id": "e0", "positive_limit": 1, "point_precision": 0.5, "component_coverage": 0.1, "random_precision": 0.1, "validation_issue_count": 0},
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.6, "component_coverage": 0.4, "random_precision": 0.1, "validation_issue_count": 0},
            {"episode_id": "e1", "positive_limit": 1, "point_precision": 0.4, "component_coverage": 0.2, "random_precision": 0.1, "validation_issue_count": 0},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.5, "component_coverage": 0.5, "random_precision": 0.1, "validation_issue_count": 0},
        ]

        summary = summarize_candidate_pilot(rows, precision_margin=0.05, min_episodes=2)

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["episode_count"], 2)

    def test_gate_c_fails_when_precision_does_not_beat_random(self) -> None:
        rows = [
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.10, "component_coverage": 0.4, "random_precision": 0.09, "validation_issue_count": 0},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.11, "component_coverage": 0.5, "random_precision": 0.10, "validation_issue_count": 0},
        ]

        summary = summarize_candidate_pilot(rows, precision_margin=0.05, min_episodes=2)

        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("precision margin" in issue for issue in summary["issues"]))

    def test_gate_c_fails_on_coordinate_validation_issues(self) -> None:
        rows = [
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.8, "component_coverage": 0.4, "random_precision": 0.1, "validation_issue_count": 1},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.8, "component_coverage": 0.5, "random_precision": 0.1, "validation_issue_count": 0},
        ]

        summary = summarize_candidate_pilot(rows, precision_margin=0.05, min_episodes=2)

        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("validation issues" in issue for issue in summary["issues"]))

    def test_gate_c_fails_when_negative_blank_fraction_is_high(self) -> None:
        rows = [
            {"episode_id": "e0", "positive_limit": 5, "point_precision": 0.8, "component_coverage": 0.4, "random_precision": 0.1, "validation_issue_count": 0, "negative_blank_fraction": 0.7},
            {"episode_id": "e1", "positive_limit": 5, "point_precision": 0.8, "component_coverage": 0.5, "random_precision": 0.1, "validation_issue_count": 0, "negative_blank_fraction": 0.6},
        ]

        summary = summarize_candidate_pilot(rows, precision_margin=0.05, min_episodes=2)

        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("blank" in issue for issue in summary["issues"]))


if __name__ == "__main__":
    unittest.main()
