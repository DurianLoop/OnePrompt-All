import json
import unittest
import uuid
from pathlib import Path

from adapromptset.scripts.run_bootstrap import run_bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(prefix: str) -> Path:
    path = PROJECT_ROOT / "tmp" / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class BootstrapCliTests(unittest.TestCase):
    def test_run_bootstrap_writes_pass_report_for_multiple_patient_clusters(self) -> None:
        root = _tmp_dir("bootstrap_pass")
        regret_csv = root / "episode_regret.csv"
        manifest_csv = root / "episodes_val.csv"
        regret_csv.write_text(
            "\n".join(
                [
                    "episode_id,oracle_dice,best_fixed_dice",
                    "e1,0.60,0.50",
                    "e2,0.80,0.70",
                    "e3,0.80,0.90",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_csv.write_text(
            "\n".join(
                [
                    "episode_id,query_case_id,query_patient,support_case_id",
                    "e1,q1,p1,r1",
                    "e2,q1,p1,r2",
                    "e3,q2,p2,r3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_dir = root / "reports"

        exit_code = run_bootstrap(
            episode_regret_csv=regret_csv,
            episode_manifest_csv=manifest_csv,
            out_dir=out_dir,
            iterations=100,
            seed=3,
        )

        payload = json.loads((out_dir / "bootstrap_summary.json").read_text(encoding="utf-8"))
        report = (out_dir / "bootstrap_summary.md").read_text(encoding="utf-8")
        query_csv = (out_dir / "query_level_comparison.csv").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["bootstrap"]["unit"], "patient_cluster")
        self.assertEqual(payload["bootstrap"]["query_count"], 2)
        self.assertEqual(payload["paired_significance"]["method"], "exact_sign_flip")
        self.assertIsNotNone(payload["paired_significance"]["p_value_two_sided"])
        self.assertIn("patient-cluster bootstrap", report)
        self.assertIn("Paired permutation p-value", report)
        self.assertIn("q1,p1,2", query_csv)

    def test_run_bootstrap_does_not_pass_with_one_patient_cluster(self) -> None:
        root = _tmp_dir("bootstrap_pending")
        regret_csv = root / "episode_regret.csv"
        manifest_csv = root / "episodes_val.csv"
        regret_csv.write_text(
            "\n".join(
                [
                    "episode_id,oracle_dice,best_fixed_dice",
                    "e1,0.60,0.50",
                    "e2,0.80,0.70",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_csv.write_text(
            "\n".join(
                [
                    "episode_id,query_case_id,query_patient,support_case_id",
                    "e1,q1,p1,r1",
                    "e2,q2,p1,r2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        exit_code = run_bootstrap(
            episode_regret_csv=regret_csv,
            episode_manifest_csv=manifest_csv,
            out_dir=root / "reports",
            iterations=50,
            seed=0,
        )

        payload = json.loads((root / "reports" / "bootstrap_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "PENDING_INSUFFICIENT_CLUSTERS")
        self.assertIn("paired_significance", payload)


if __name__ == "__main__":
    unittest.main()
