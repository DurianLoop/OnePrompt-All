from __future__ import annotations

# ruff: noqa: E402, I001

import csv
import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths, run_smoke, summarize_score_maps


class RealMatcherSmokeTests(unittest.TestCase):
    def test_resolve_episode_paths_uses_monusac_dataset_parent(self) -> None:
        row = {
            "dataset": "MoNuSAC",
            "support_image_path": "processed/train/images/train_001.png",
            "support_mask_path": "processed/train/masks/train_001.png",
            "query_image_path": "processed/train/images/train_002.png",
            "query_mask_path": "processed/train/masks/train_002.png",
        }

        resolved = resolve_episode_paths(row, PROJECT_ROOT.parent)

        self.assertEqual(
            resolved["support_image"],
            PROJECT_ROOT.parent / "code" / "datasets" / "MoNuSAC" / "processed" / "train" / "images" / "train_001.png",
        )

    def test_resolve_episode_paths_uses_glass_external_root(self) -> None:
        row = {
            "dataset": "GlaS",
            "support_image_path": "Warwick QU Dataset (Released 2016_07_08)/train_60.bmp",
            "support_mask_path": "Warwick QU Dataset (Released 2016_07_08)/train_60_anno.bmp",
            "query_image_path": "Warwick QU Dataset (Released 2016_07_08)/train_41.bmp",
            "query_mask_path": "Warwick QU Dataset (Released 2016_07_08)/train_41_anno.bmp",
        }

        resolved = resolve_episode_paths(row, PROJECT_ROOT.parent)

        self.assertEqual(
            resolved["query_mask"],
            PROJECT_ROOT.parent
            / "external"
            / "research_data"
            / "extracted"
            / "GlaS"
            / "Warwick QU Dataset (Released 2016_07_08)"
            / "train_41_anno.bmp",
        )

    def test_dry_run_report_keeps_formal_gate_pending(self) -> None:
        out_dir = PROJECT_ROOT / ".test_tmp" / "real_matcher_smoke_dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        episodes = out_dir / "episodes_val.csv"
        with episodes.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode_id",
                    "dataset",
                    "support_case_id",
                    "support_image_path",
                    "support_mask_path",
                    "query_case_id",
                    "query_image_path",
                    "query_mask_path",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "episode_id": "e0",
                    "dataset": "MoNuSAC",
                    "support_case_id": "s0",
                    "support_image_path": "processed/train/images/s0.png",
                    "support_mask_path": "processed/train/masks/s0.png",
                    "query_case_id": "q0",
                    "query_image_path": "processed/train/images/q0.png",
                    "query_mask_path": "processed/train/masks/q0.png",
                }
            )

        result = run_smoke(
            project_root=PROJECT_ROOT,
            episodes_csv=episodes,
            out_dir=out_dir,
            dry_run=True,
        )

        payload = json.loads((out_dir / "real_matcher_smoke.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["status"], "DRY_RUN")
        self.assertEqual(payload["formal_gate_c"], "PENDING_FULL_CANDIDATE_PILOT")
        self.assertNotIn("query_mask_path", json.dumps(payload["evidence_metadata"]))

    def test_summarize_score_maps_exports_global_features_for_each_map(self) -> None:
        maps = {
            "s_fg": torch.tensor([[3.0, 1.0], [0.0, -1.0]]),
            "s_bg": torch.tensor([[0.0, 2.0], [1.0, -1.0]]),
            "s_ctr": torch.tensor([[3.0, -1.0], [-1.0, 0.0]]),
        }

        summary = summarize_score_maps(maps)

        self.assertEqual(set(summary), {"s_fg", "s_bg", "s_ctr"})
        self.assertAlmostEqual(summary["s_fg"]["top_margin"], 2.0)
        self.assertIn("matching_entropy", summary["s_ctr"])
        self.assertIn("high_score_area_fraction", summary["s_bg"])


if __name__ == "__main__":
    unittest.main()
