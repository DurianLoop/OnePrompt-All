import json
import uuid
import unittest
from pathlib import Path

from adapromptset.scripts.audit_optimizer_guard import run_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(prefix: str) -> Path:
    path = PROJECT_ROOT / "tmp" / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class OptimizerGuardCliTests(unittest.TestCase):
    def test_run_audit_writes_pass_report(self) -> None:
        out_dir = _tmp_dir("optimizer_guard_pass")
        exit_code = run_audit(out_dir=out_dir, include_frozen_violation=False)
        payload = json.loads((out_dir / "optimizer_guard.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["policy_training"], "not_started")
        self.assertEqual(payload["guard_report"]["checked_roles"], ["matcher", "segmenter"])

    def test_run_audit_writes_fail_report_for_violation(self) -> None:
        out_dir = _tmp_dir("optimizer_guard_fail")
        exit_code = run_audit(out_dir=out_dir, include_frozen_violation=True)
        payload = json.loads((out_dir / "optimizer_guard.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "FAIL")
        self.assertIn("matcher", payload["error"])


if __name__ == "__main__":
    unittest.main()
