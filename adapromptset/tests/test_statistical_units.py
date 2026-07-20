import unittest

from adapromptset.statistics.units import (
    aggregate_reference_metrics,
    summarize_episode_manifest_units,
)


class StatisticalUnitsTests(unittest.TestCase):
    def test_aggregate_reference_metrics_groups_by_query_and_action(self) -> None:
        rows = [
            {"dataset": "Toy", "role": "val", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r1", "action_id": "A", "dice": "0.20"},
            {"dataset": "Toy", "role": "val", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r2", "action_id": "A", "dice": "0.40"},
            {"dataset": "Toy", "role": "val", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r1", "action_id": "B", "dice": "0.50"},
            {"dataset": "Toy", "role": "val", "query_case_id": "q2", "query_patient": "p1", "support_case_id": "r3", "action_id": "A", "dice": "0.90"},
        ]

        aggregated = aggregate_reference_metrics(rows, metric_names=["dice"])

        self.assertEqual(len(aggregated), 3)
        first = aggregated[0]
        self.assertEqual(first["query_case_id"], "q1")
        self.assertEqual(first["action_id"], "A")
        self.assertEqual(first["reference_episode_count"], 2)
        self.assertEqual(first["independent_unit"], "query")
        self.assertAlmostEqual(first["dice_mean"], 0.30)
        self.assertEqual(first["support_case_ids"], "r1;r2")

    def test_manifest_summary_counts_queries_and_patient_clusters(self) -> None:
        episodes = [
            {"dataset": "Toy", "role": "test", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r1"},
            {"dataset": "Toy", "role": "test", "query_case_id": "q1", "query_patient": "p1", "support_case_id": "r2"},
            {"dataset": "Toy", "role": "test", "query_case_id": "q2", "query_patient": "p1", "support_case_id": "r3"},
            {"dataset": "Toy", "role": "test", "query_case_id": "q3", "query_patient": "p2", "support_case_id": "r4"},
        ]

        summary = summarize_episode_manifest_units(episodes)

        self.assertEqual(summary["episode_count"], 4)
        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["query_patient_count"], 2)
        self.assertEqual(summary["recommended_independent_unit"], "patient_cluster")
        self.assertEqual(summary["max_references_per_query"], 2)
        self.assertEqual(summary["reference_episode_count_by_query"]["q1"], 2)
        self.assertEqual(summary["multi_reference_episodes_are_independent"], False)


if __name__ == "__main__":
    unittest.main()
