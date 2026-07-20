import json
import uuid
import unittest
from pathlib import Path

from adapromptset.scripts.audit_statistical_units import run_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(prefix: str) -> Path:
    path = PROJECT_ROOT / "tmp" / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class StatisticalUnitAuditCliTests(unittest.TestCase):
    def test_run_audit_writes_json_and_markdown(self) -> None:
        root = _tmp_dir("stat_units")
        manifest = root / "episodes_val.csv"
        manifest.write_text(
            "\n".join(
                [
                    "episode_id,dataset,role,support_case_id,support_patient,query_case_id,query_patient",
                    "e1,Toy,val,r1,pr1,q1,pq1",
                    "e2,Toy,val,r2,pr2,q1,pq1",
                    "e3,Toy,val,r3,pr3,q2,pq2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_dir = root / "reports"

        exit_code = run_audit(manifest_paths=[manifest], out_dir=out_dir)
        payload = json.loads((out_dir / "statistical_units.json").read_text(encoding="utf-8"))
        report = (out_dir / "statistical_units.md").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["manifests"][0]["query_count"], 2)
        self.assertFalse(payload["manifests"][0]["multi_reference_episodes_are_independent"])
        self.assertIn("query-level", report)
        self.assertIn("not independent samples", report)


if __name__ == "__main__":
    unittest.main()
