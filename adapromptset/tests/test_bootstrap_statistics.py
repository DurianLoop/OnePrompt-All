import unittest

from adapromptset.statistics.bootstrap import (
    build_query_level_comparison,
    paired_cluster_bootstrap,
    paired_permutation_test,
    win_tie_loss,
)


class BootstrapStatisticsTests(unittest.TestCase):
    def test_query_level_comparison_aggregates_references_before_bootstrap(self) -> None:
        episode_rows = [
            {"episode_id": "e1", "best_fixed_dice": "0.50", "oracle_dice": "0.60"},
            {"episode_id": "e2", "best_fixed_dice": "0.70", "oracle_dice": "0.80"},
            {"episode_id": "e3", "best_fixed_dice": "0.90", "oracle_dice": "0.80"},
        ]
        manifest_rows = [
            {"episode_id": "e1", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r1"},
            {"episode_id": "e2", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r2"},
            {"episode_id": "e3", "query_case_id": "q2", "query_patient": "p2", "support_case_id": "r3"},
        ]

        query_rows = build_query_level_comparison(
            episode_rows,
            manifest_rows,
            baseline_metric="best_fixed_dice",
            comparator_metric="oracle_dice",
        )

        self.assertEqual(len(query_rows), 2)
        self.assertEqual(query_rows[0]["query_case_id"], "q1")
        self.assertEqual(query_rows[0]["query_patient"], "p1")
        self.assertEqual(query_rows[0]["reference_episode_count"], 2)
        self.assertAlmostEqual(query_rows[0]["best_fixed_dice_mean"], 0.60)
        self.assertAlmostEqual(query_rows[0]["oracle_dice_mean"], 0.70)
        self.assertAlmostEqual(query_rows[0]["paired_delta"], 0.10)

    def test_paired_cluster_bootstrap_resamples_patient_clusters(self) -> None:
        query_rows = [
            {"query_case_id": "q1", "query_patient": "p1", "paired_delta": 0.10},
            {"query_case_id": "q2", "query_patient": "p1", "paired_delta": -0.10},
            {"query_case_id": "q3", "query_patient": "p2", "paired_delta": 0.20},
        ]

        summary = paired_cluster_bootstrap(query_rows, iterations=200, seed=7)

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["unit"], "patient_cluster")
        self.assertEqual(summary["cluster_count"], 2)
        self.assertEqual(summary["query_count"], 3)
        self.assertAlmostEqual(summary["observed_mean_delta"], (0.10 - 0.10 + 0.20) / 3)
        self.assertLessEqual(summary["ci_low"], summary["observed_mean_delta"])
        self.assertGreaterEqual(summary["ci_high"], summary["observed_mean_delta"])

    def test_paired_cluster_bootstrap_requires_at_least_two_clusters(self) -> None:
        query_rows = [
            {"query_case_id": "q1", "query_patient": "p1", "paired_delta": 0.10},
            {"query_case_id": "q2", "query_patient": "p1", "paired_delta": -0.10},
        ]

        summary = paired_cluster_bootstrap(query_rows, iterations=50, seed=0)

        self.assertEqual(summary["status"], "PENDING_INSUFFICIENT_CLUSTERS")
        self.assertEqual(summary["cluster_count"], 1)
        self.assertIsNone(summary["ci_low"])
        self.assertIsNone(summary["ci_high"])

    def test_win_tie_loss_uses_query_level_delta(self) -> None:
        rows = [
            {"paired_delta": 0.010},
            {"paired_delta": -0.010},
            {"paired_delta": 0.004},
        ]

        self.assertEqual(win_tie_loss(rows, tie_threshold=0.005), {"win": 1, "tie": 1, "loss": 1})

    def test_paired_permutation_test_uses_two_sided_sign_flips(self) -> None:
        rows = [
            {"paired_delta": 0.10},
            {"paired_delta": 0.20},
            {"paired_delta": 0.30},
        ]

        summary = paired_permutation_test(rows)

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["method"], "exact_sign_flip")
        self.assertEqual(summary["query_count"], 3)
        self.assertAlmostEqual(summary["observed_mean_delta"], 0.20)
        self.assertAlmostEqual(summary["p_value_two_sided"], 0.25)

    def test_paired_permutation_test_requires_nonzero_deltas(self) -> None:
        rows = [
            {"paired_delta": 0.0},
            {"paired_delta": 0.0},
        ]

        summary = paired_permutation_test(rows)

        self.assertEqual(summary["status"], "PENDING_NO_NONZERO_DELTAS")
        self.assertIsNone(summary["p_value_two_sided"])


if __name__ == "__main__":
    unittest.main()
