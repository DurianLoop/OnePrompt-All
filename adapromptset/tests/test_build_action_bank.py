from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_action_bank import build_action_bank_artifacts


class BuildActionBankTests(unittest.TestCase):
    def test_build_action_bank_artifacts_from_candidate_csv(self) -> None:
        work_dir = PROJECT_ROOT / ".test_tmp" / "action_bank_build"
        work_dir.mkdir(parents=True, exist_ok=True)
        candidates_csv = work_dir / "candidates.csv"
        candidates_csv.write_text(
            "\n".join(
                [
                    "episode_id,positive_limit,kind,grid_row,grid_col,x,y,score",
                    "e0,5,positive,0,0,1,1,5.0",
                    "e0,5,positive,0,1,2,1,4.0",
                    "e0,5,positive,0,2,3,1,3.0",
                    "e0,5,positive,0,3,4,1,2.0",
                    "e0,5,positive,0,4,5,1,1.0",
                    "e0,5,negative,1,0,1,9,0.5",
                    "e0,5,negative,1,1,2,9,0.4",
                    "e0,5,negative,1,2,3,9,0.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_dir = work_dir / "reports"
        config_dir = work_dir / "configs"

        result = build_action_bank_artifacts(
            candidates_csv=candidates_csv,
            episode_id="e0",
            positive_limit=5,
            out_dir=out_dir,
            config_dir=config_dir,
        )

        self.assertEqual(result, 0)
        summary = json.loads((out_dir / "action_bank_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["name"], "prompt_set_bank_v1")
        self.assertEqual(summary["action_count"], 10)
        self.assertEqual(summary["valid_action_count"], 9)
        self.assertEqual(summary["invalid_action_count"], 1)
        self.assertEqual(len(summary["bank_hash"]), 64)
        self.assertTrue((config_dir / "prompt_set_bank_v1.json").is_file())
        yaml_text = (config_dir / "prompt_set_bank_v1.yaml").read_text(encoding="utf-8")
        self.assertIn("name: prompt_set_bank_v1", yaml_text)
        self.assertIn("A_top5_one_group", yaml_text)

    def test_build_rescue_action_bank_artifacts_writes_v2_without_overwriting_v1_name(self) -> None:
        work_dir = PROJECT_ROOT / ".test_tmp" / "action_bank_build_v2"
        work_dir.mkdir(parents=True, exist_ok=True)
        candidates_csv = work_dir / "candidates.csv"
        candidates_csv.write_text(
            "\n".join(
                [
                    "episode_id,positive_limit,kind,grid_row,grid_col,x,y,score",
                    "e0,5,positive,0,0,1,1,5.0",
                    "e0,5,positive,0,1,2,1,4.0",
                    "e0,5,positive,0,2,3,1,3.0",
                    "e0,5,positive,0,3,4,1,2.0",
                    "e0,5,positive,0,4,5,1,1.0",
                    "e0,5,negative,1,0,1,9,0.5",
                    "e0,5,negative,1,1,2,9,0.4",
                    "e0,5,negative,1,2,3,9,0.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = build_action_bank_artifacts(
            candidates_csv=candidates_csv,
            episode_id="e0",
            positive_limit=5,
            out_dir=work_dir / "reports",
            config_dir=work_dir / "configs",
            bank_version="v2",
        )

        self.assertEqual(result, 0)
        summary = json.loads((work_dir / "reports" / "action_bank_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["name"], "prompt_set_bank_v2")
        self.assertGreater(summary["action_count"], 10)
        self.assertTrue((work_dir / "configs" / "prompt_set_bank_v2.json").is_file())
        self.assertIn("R_split5_2groups_localneg", (work_dir / "configs" / "prompt_set_bank_v2.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
