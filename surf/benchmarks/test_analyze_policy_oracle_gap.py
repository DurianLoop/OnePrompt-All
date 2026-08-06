import pytest

from analyze_policy_oracle_gap import decompose_policy_regret


def test_decompose_policy_regret_splits_prefilter_and_selector_errors() -> None:
    candidates = [
        {"query_case_id": "q", "method": "A", "dice": 0.8, "k": 20, "lambda_bg": 0.25},
        {"query_case_id": "q", "method": "B", "dice": 0.6, "k": 40, "lambda_bg": 0.75},
        {"query_case_id": "q", "method": "C", "dice": 0.4, "k": 75, "lambda_bg": 1.5},
    ]
    shortlist = [
        {"query_case_id": "q", "action": "B", "dice": 0.6, "k": 40, "lambda_bg": 0.75},
        {"query_case_id": "q", "action": "C", "dice": 0.4, "k": 75, "lambda_bg": 1.5},
    ]
    selected = [{"query_case_id": "q", "action": "C", "dice": 0.4, "k": 75, "lambda_bg": 1.5}]
    fixed = [{"query_case_id": "q", "method": "B", "dice": 0.6, "k": 40, "lambda_bg": 0.75}]

    rows, summary = decompose_policy_regret(candidates, selected, fixed, shortlist)

    assert rows[0]["full_oracle_dice"] == pytest.approx(0.8)
    assert rows[0]["shortlist_oracle_dice"] == pytest.approx(0.6)
    assert rows[0]["prefilter_regret"] == pytest.approx(0.2)
    assert rows[0]["selector_regret"] == pytest.approx(0.2)
    assert summary["mean_policy_regret"] == pytest.approx(0.4)
    assert summary["shortlist_oracle_recall"] == pytest.approx(0.0)
