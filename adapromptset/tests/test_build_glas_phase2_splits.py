from __future__ import annotations

# ruff: noqa: E402, I001

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.build_glas_phase2_splits import (
    GlasCase,
    build_phase2_assignments,
    build_reference_episodes,
)  # noqa: E402


def case(case_id: str, source_split: str, grade: str, patient: str = "p") -> GlasCase:
    return GlasCase(
        case_id=case_id,
        source_split=source_split,
        patient_id=patient,
        grade=grade,
        sirin_grade=grade,
        image_path=Path(__file__),
        mask_path=Path(__file__),
    )


class TestBuildGlasPhase2Splits(unittest.TestCase):
    def test_builds_stratified_55_30_official_train_split(self) -> None:
        image = Path(__file__)
        mask = Path(__file__)
        cases: list[GlasCase] = []
        for index in range(37):
            cases.append(GlasCase(f"train_b_{index}", "train", f"pb{index}", "benign", "benign", image, mask))
        for index in range(48):
            cases.append(
                GlasCase(f"train_m_{index}", "train", f"pm{index}", "malignant", "malignant", image, mask)
            )
        for index in range(60):
            cases.append(GlasCase(f"testA_{index}", "testA", f"ta{index}", "benign", "benign", image, mask))
        for index in range(20):
            cases.append(GlasCase(f"testB_{index}", "testB", f"tb{index}", "malignant", "malignant", image, mask))

        assignments, metadata = build_phase2_assignments(cases)

        self.assertEqual(metadata["phase2_role_counts"]["utility_train"], 55)
        self.assertEqual(metadata["phase2_role_counts"]["strict_val"], 30)
        self.assertEqual(metadata["phase2_role_counts"]["locked_testA"], 60)
        self.assertEqual(metadata["phase2_role_counts"]["locked_testB"], 20)
        self.assertEqual(metadata["grade_counts_by_role"]["strict_val"], {"benign": 13, "malignant": 17})
        self.assertTrue(all(row["phase2_role"] != "locked_testA" for row in assignments if row["source_split"] == "train"))

    def test_reference_episodes_use_training_pool_only(self) -> None:
        image = Path(__file__)
        mask = Path(__file__)
        cases = [
            GlasCase(f"train_{index}", "train", f"p{index}", "benign", "benign", image, mask)
            for index in range(85)
        ]
        cases.extend(
            GlasCase(f"testA_{index}", "testA", f"ta{index}", "benign", "benign", image, mask)
            for index in range(60)
        )
        cases.extend(
            GlasCase(f"testB_{index}", "testB", f"tb{index}", "benign", "benign", image, mask)
            for index in range(20)
        )
        assignments, _ = build_phase2_assignments(cases)
        episodes = build_reference_episodes(assignments, references_per_query=3)

        self.assertEqual(len(episodes["train"]), 55 * 3)
        self.assertEqual(len(episodes["val"]), 30 * 3)
        self.assertTrue(all(row["support_role"] == "utility_train" for rows in episodes.values() for row in rows))
        self.assertTrue(all(row["support_case_id"] != row["query_case_id"] for rows in episodes.values() for row in rows))


if __name__ == "__main__":
    unittest.main()
