from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.export_action_bank_figure import write_action_bank_figure  # noqa: E402


def write_bank_fixture(root: Path) -> None:
    bank = {
        "name": "prompt_set_bank_v2",
        "action_count": 5,
        "valid_action_count": 4,
        "invalid_action_count": 1,
        "bank_hash": "F" * 64,
        "source": {
            "candidate_csv": "reports/candidate_pilot/candidate_pilot_candidates.csv",
            "episode_id": "episode_001",
            "positive_limit": 5,
            "candidate_index_order": "positives_then_negatives",
        },
        "actions": [
            {
                "action_id": "A_top1_one_group",
                "family": "single_group_positive",
                "groups": [{"positive_indices": [0], "negative_indices": []}],
                "valid": True,
                "num_groups": 1,
                "num_positive": 1,
                "num_negative": 0,
                "sam_calls": 1,
                "prompt_points": 1,
            },
            {
                "action_id": "B_top5_pos3neg",
                "family": "single_group_positive_negative",
                "groups": [{"positive_indices": [0, 1, 2, 3, 4], "negative_indices": [5, 6, 7]}],
                "valid": True,
                "num_groups": 1,
                "num_positive": 5,
                "num_negative": 3,
                "sam_calls": 1,
                "prompt_points": 8,
            },
            {
                "action_id": "C_top3_independent",
                "family": "independent_positive_groups",
                "groups": [{"positive_indices": [0], "negative_indices": []}],
                "valid": True,
                "num_groups": 3,
                "num_positive": 3,
                "num_negative": 0,
                "sam_calls": 3,
                "prompt_points": 3,
            },
            {
                "action_id": "R_split5_2groups_localneg",
                "family": "rescue_group_specific_negatives",
                "groups": [{"positive_indices": [0, 1, 2], "negative_indices": [8]}],
                "valid": True,
                "num_groups": 2,
                "num_positive": 5,
                "num_negative": 2,
                "sam_calls": 2,
                "prompt_points": 7,
            },
            {
                "action_id": "A_top10_one_group",
                "family": "single_group_positive",
                "groups": [],
                "valid": False,
                "invalid_reason": "requires_10_positive_candidates",
                "num_groups": 0,
                "num_positive": 0,
                "num_negative": 0,
                "sam_calls": 0,
                "prompt_points": 0,
            },
        ],
    }
    path = root / "configs" / "actions" / "prompt_set_bank_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank), encoding="utf-8")


class ExportActionBankFigureTests(unittest.TestCase):
    def test_write_action_bank_figure_records_current_subset_and_deferred_families(self) -> None:
        root = PROJECT_ROOT / "tmp" / "test_export_action_bank_figure"
        write_bank_fixture(root)

        outputs = write_action_bank_figure(root)

        for path in outputs.values():
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0)
        tex = outputs["tex"].read_text(encoding="utf-8")
        self.assertIn("Top-K one group", tex)
        self.assertIn("Positive + negative", tex)
        self.assertIn("Independent groups", tex)
        self.assertIn("Rescue split/local negatives", tex)
        self.assertIn("Deferred in current MVP subset", tex)
        self.assertIn("Complete prompt-set proposals only", tex)
        self.assertIn("SAM calls", tex)
        self.assertNotIn("SIMULATED PLACEHOLDER", tex)

        payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS_ACTION_BANK_FIGURE")
        self.assertEqual(payload["claim_boundary"], "oracle_analysis_only_current_mvp_subset")
        self.assertEqual(payload["source_bank"]["name"], "prompt_set_bank_v2")
        self.assertEqual(payload["family_counts"]["single_group_positive"], 1)
        self.assertTrue(payload["checks"]["deferred_families_disclosed"])
        self.assertTrue(payload["checks"]["valid_counts_match_source"])
        self.assertIn("gf_sam_style_graph_groups", payload["deferred_families"])


if __name__ == "__main__":
    unittest.main()
