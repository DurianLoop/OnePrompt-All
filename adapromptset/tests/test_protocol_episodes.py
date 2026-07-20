from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.data.protocol import build_episode_protocol, validate_gate_b


def row(case_id: str, role: str, patient: str) -> dict[str, str]:
    return {
        "dataset": "Toy",
        "case_id": case_id,
        "split": "train" if role in {"train", "val", "excluded_overlap"} else "test",
        "role": role,
        "patient_id": patient,
        "image_path": f"{case_id}.png",
        "mask_path": f"{case_id}_mask.png",
        "image_sha256": f"image-{case_id}",
        "mask_sha256": f"mask-{case_id}",
    }


class ProtocolEpisodeTests(unittest.TestCase):
    def test_protocol_is_deterministic_and_patient_disjoint(self) -> None:
        rows = [
            row("a0", "train", "pa"),
            row("a1", "train", "pa"),
            row("b0", "train", "pb"),
            row("b1", "train", "pb"),
            row("c0", "train", "pc"),
            row("c1", "train", "pc"),
            row("d0", "train", "pd"),
            row("d1", "train", "pd"),
            row("x0", "excluded_overlap", "px"),
            row("t0", "test", "pt"),
        ]

        first = build_episode_protocol(rows, val_patient_count=1, references_per_query=2, seed=17)
        second = build_episode_protocol(rows, val_patient_count=1, references_per_query=2, seed=17)

        self.assertEqual(first, second)
        self.assertIn("episodes_test", first)
        for key in ("episodes_train", "episodes_val", "episodes_test"):
            for episode in first[key]:
                self.assertEqual(episode["support_role"], "train")
                self.assertNotEqual(episode["support_patient"], episode["query_patient"])
                self.assertNotEqual(episode["support_case_id"], episode["query_case_id"])

    def test_gate_b_rejects_test_support_leakage(self) -> None:
        protocol = {
            "case_assignments": [row("a0", "train", "pa"), row("t0", "test", "pt")],
            "episodes_train": [],
            "episodes_val": [],
            "episodes_test": [
                {
                    "episode_id": "bad",
                    "support_role": "test",
                    "support_case_id": "t0",
                    "support_patient": "pt",
                    "query_case_id": "a0",
                    "query_patient": "pa",
                }
            ],
        }

        result = validate_gate_b(protocol)

        self.assertFalse(result.passed)
        self.assertTrue(any("support_role" in issue for issue in result.issues))

    def test_protocol_fails_when_not_enough_patient_disjoint_supports(self) -> None:
        rows = [
            row("a0", "train", "pa"),
            row("a1", "train", "pa"),
            row("t0", "test", "pt"),
        ]

        with self.assertRaisesRegex(ValueError, "patient-disjoint train supports"):
            build_episode_protocol(rows, val_patient_count=0, references_per_query=1, seed=3)


if __name__ == "__main__":
    unittest.main()
