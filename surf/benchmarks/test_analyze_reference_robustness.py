import pytest

from analyze_reference_robustness import summarize_reference_robustness


def test_summarize_reference_robustness_aggregates_within_query_variation() -> None:
    rows = [
        {"query_case_id": "q1", "dice": "0.4", "selected_action": "a"},
        {"query_case_id": "q1", "dice": "0.6", "selected_action": "b"},
        {"query_case_id": "q2", "dice": "0.5", "selected_action": "a"},
        {"query_case_id": "q2", "dice": "0.5", "selected_action": "a"},
    ]

    per_query, summary = summarize_reference_robustness(rows)

    assert len(per_query) == 2
    assert summary["query_count"] == 2
    assert summary["reference_episode_count"] == 4
    assert summary["mean_reference_range"] == pytest.approx(0.1)
    assert summary["mean_unique_actions"] == 1.5
