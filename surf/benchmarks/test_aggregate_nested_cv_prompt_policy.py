import pytest

from aggregate_nested_cv_prompt_policy import validate_outer_test_coverage


def test_validate_outer_test_coverage_accepts_one_fold_per_query_and_five_references() -> None:
    rows = [
        {"fold": "0", "query_case_id": "a", "episode_id": f"a_{index}"}
        for index in range(5)
    ] + [
        {"fold": "1", "query_case_id": "b", "episode_id": f"b_{index}"}
        for index in range(5)
    ]

    summary = validate_outer_test_coverage(rows, expected_reference_count=5)

    assert summary == {"query_count": 2, "episode_count": 10, "fold_count": 2}


def test_validate_outer_test_coverage_rejects_query_in_multiple_folds() -> None:
    rows = [
        {"fold": "0", "query_case_id": "a", "episode_id": "a_0"},
        {"fold": "1", "query_case_id": "a", "episode_id": "a_1"},
    ]

    with pytest.raises(ValueError, match="multiple outer folds"):
        validate_outer_test_coverage(rows, expected_reference_count=1)
