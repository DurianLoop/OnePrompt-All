import pytest

from analyze_policy_switches import analyze_switches


def test_analyze_switches_pairs_legacy_case_name_and_episode_id() -> None:
    fixed = [
        {"case_name": "e1", "query_case_id": "q1", "dice": "0.5", "method": "anchor"},
        {"case_name": "e2", "query_case_id": "q2", "dice": "0.6", "method": "anchor"},
    ]
    learned = [
        {"episode_id": "e1", "query_case_id": "q1", "dice": "0.7", "selected_action": "alt", "switched": "1"},
        {"episode_id": "e2", "query_case_id": "q2", "dice": "0.6", "selected_action": "anchor", "switched": "0"},
    ]

    rows, action_rows, summary = analyze_switches(fixed, learned)

    assert len(rows) == 2
    assert summary["switch_count"] == 1
    assert summary["switch_mean_delta_dice"] == pytest.approx(0.2)
    assert summary["switch_wins"] == 1
    assert action_rows[0]["action"] == "alt"
