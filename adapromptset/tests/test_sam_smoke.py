from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_sam_smoke import (
    classify_gate_d,
    connected_component_at_point,
    mask_centroid,
    run_smoke,
)


class SamSmokeTests(unittest.TestCase):
    def test_mask_centroid_uses_foreground_center(self) -> None:
        mask = [
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ]

        self.assertEqual(mask_centroid(mask), (2, 2))

    def test_dry_run_writes_pending_gate_d(self) -> None:
        out_dir = PROJECT_ROOT / ".test_tmp" / "sam_smoke_dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_smoke(project_root=PROJECT_ROOT, out_dir=out_dir, dry_run=True)

        payload = json.loads((out_dir / "sam_smoke_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["status"], "DRY_RUN")
        self.assertEqual(payload["formal_gate_d"], "PENDING_EXECUTION")

    def test_low_privileged_dice_keeps_formal_gate_pending(self) -> None:
        status = classify_gate_d(
            privileged_dice=0.008,
            repeat_same=True,
            call_count=4,
            prompt_group_count=4,
        )

        self.assertEqual(status, "PENDING_INSTANCE_AWARE_GT_CENTER_CHECK")

    def test_connected_component_at_point_isolates_component(self) -> None:
        mask = [
            [False, True, True, False],
            [False, False, True, False],
            [True, False, False, True],
        ]

        component = connected_component_at_point(mask, (1, 0))

        self.assertEqual(
            component,
            [
                [False, True, True, False],
                [False, False, True, False],
                [False, False, False, False],
            ],
        )

    def test_instance_aware_privileged_dice_allows_formal_gate_pass(self) -> None:
        status = classify_gate_d(
            privileged_dice=0.008,
            privileged_component_dice=0.93,
            repeat_same=True,
            call_count=4,
            prompt_group_count=4,
        )

        self.assertEqual(status, "PASS")


if __name__ == "__main__":
    unittest.main()
