import pytest

from fspl.reporting import summarize_rule_rows


def test_summarize_rule_rows_reports_counts_and_mean_dice() -> None:
    rows = [
        {"rule_name": "top5_n3", "utility": 0.4},
        {"rule_name": "top5_n3", "utility": 0.6},
        {"rule_name": "spread5_n3", "utility": 0.5},
        {"rule_name": "spread5_n3", "utility": 0.7},
    ]

    summary = summarize_rule_rows(rows, expected_episodes=2)

    assert summary == {
        "spread5_n3": {"case_count": 2, "mean_dice": 0.6},
        "top5_n3": {"case_count": 2, "mean_dice": 0.5},
    }


def test_summarize_rule_rows_rejects_incomplete_rules() -> None:
    rows = [
        {"rule_name": "top5_n3", "utility": 0.4},
        {"rule_name": "spread5_n3", "utility": 0.5},
        {"rule_name": "spread5_n3", "utility": 0.7},
    ]

    with pytest.raises(ValueError, match="expected 2 episodes"):
        summarize_rule_rows(rows, expected_episodes=2)
