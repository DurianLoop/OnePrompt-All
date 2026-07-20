import pytest

from analyze_prompt_policy_results import (
    aggregate_paired_by_query,
    exact_two_sided_sign_test,
    paired_policy_summary,
    pair_policy_rows,
)


def test_pair_policy_rows_aligns_cases_and_computes_delta() -> None:
    fixed = [
        {"query_case_id": "a", "dice": "0.4", "action": "fixed"},
        {"query_case_id": "b", "dice": "0.6", "action": "fixed"},
    ]
    learned = [
        {"query_case_id": "b", "dice": "0.5", "action": "B", "selection_source": "policy"},
        {"query_case_id": "a", "dice": "0.7", "action": "A", "selection_source": "policy"},
    ]

    paired = pair_policy_rows(fixed, learned)

    assert [row["query_case_id"] for row in paired] == ["a", "b"]
    assert paired[0]["delta_dice"] == pytest.approx(0.3)
    assert paired[1]["delta_dice"] == pytest.approx(-0.1)


def test_paired_policy_summary_reports_wins_and_reproducible_interval() -> None:
    paired = [
        {"fixed_dice": 0.4, "learned_dice": 0.6, "delta_dice": 0.2},
        {"fixed_dice": 0.5, "learned_dice": 0.5, "delta_dice": 0.0},
        {"fixed_dice": 0.7, "learned_dice": 0.6, "delta_dice": -0.1},
    ]

    first = paired_policy_summary(paired, bootstrap_samples=1000, seed=7)
    second = paired_policy_summary(paired, bootstrap_samples=1000, seed=7)

    assert first == second
    assert first["case_count"] == 3
    assert first["wins"] == 1
    assert first["ties"] == 1
    assert first["losses"] == 1
    assert first["mean_delta_dice"] == pytest.approx(1 / 30)
    assert first["median_delta_dice"] == pytest.approx(0.0)
    assert first["leave_one_out_mean_delta_min"] == pytest.approx(-0.05)
    assert first["leave_one_out_mean_delta_max"] == pytest.approx(0.1)
    assert first["ci95_low"] <= first["mean_delta_dice"] <= first["ci95_high"]


def test_pair_policy_rows_uses_unique_query_id_across_legacy_and_episode_outputs() -> None:
    fixed = [{"query_case_id": "q", "case_name": "support_to_q", "dice": "0.4"}]
    learned = [{"query_case_id": "q", "dice": "0.5"}]

    paired = pair_policy_rows(fixed, learned)

    assert paired[0]["query_case_id"] == "q"
    assert paired[0]["delta_dice"] == pytest.approx(0.1)


def test_aggregate_paired_by_query_averages_references_before_statistics() -> None:
    paired = [
        {"episode_id": "r1_to_q1", "query_case_id": "q1", "fixed_dice": 0.4, "learned_dice": 0.6},
        {"episode_id": "r2_to_q1", "query_case_id": "q1", "fixed_dice": 0.6, "learned_dice": 0.5},
        {"episode_id": "r1_to_q2", "query_case_id": "q2", "fixed_dice": 0.8, "learned_dice": 0.9},
    ]

    aggregated = aggregate_paired_by_query(paired)

    assert [row["query_case_id"] for row in aggregated] == ["q1", "q2"]
    assert aggregated[0]["reference_count"] == 2
    assert aggregated[0]["fixed_dice"] == pytest.approx(0.5)
    assert aggregated[0]["learned_dice"] == pytest.approx(0.55)
    assert aggregated[0]["delta_dice"] == pytest.approx(0.05)


def test_exact_sign_test_ignores_ties_and_is_two_sided() -> None:
    result = exact_two_sided_sign_test(wins=9, losses=1)

    assert result["non_tied_count"] == 10
    assert result["p_value"] == pytest.approx(0.021484375)
