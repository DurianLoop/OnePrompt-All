from types import SimpleNamespace

from train_negative_budget_policy import action_space_from_rows, feature_importance_rows, join_negative_candidates


def test_join_negative_candidates_reuses_matcher_features_and_overwrites_sam_response() -> None:
    base = [{
        "episode_id": "episode_a",
        "query_case_id": "query_a",
        "score_mean": 0.2,
        "sam_score": 0.9,
        "pred_area_pct": 50.0,
    }]
    negative = [{
        "episode_id": "episode_a",
        "query_case_id": "query_a",
        "negative_count": "5",
        "dice": "0.7",
        "iou": "0.6",
        "object_recall": "0.8",
        "merge_error": "0.1",
        "sam_score": "0.75",
        "pred_area_pct": "42.0",
    }]

    joined = join_negative_candidates(negative, base)

    assert joined[0]["action"] == "neg5"
    assert joined[0]["method"] == "neg5"
    assert joined[0]["score_mean"] == 0.2
    assert joined[0]["sam_score"] == 0.75
    assert joined[0]["pred_area_pct"] == 42.0


def test_join_negative_candidates_uses_requested_budget_for_action_and_actual_count_for_feature() -> None:
    base = [{"episode_id": "episode_a", "query_case_id": "query_a"}]
    negative = [{
        "episode_id": "episode_a",
        "query_case_id": "query_a",
        "requested_negative_count": "5",
        "negative_count": "2",
        "dice": "0.7",
        "iou": "0.6",
        "object_recall": "0.8",
        "merge_error": "0.1",
        "sam_score": "0.75",
        "pred_area_pct": "42.0",
    }]

    joined = join_negative_candidates(negative, base)

    assert joined[0]["action"] == "neg5"
    assert joined[0]["requested_negative_count"] == 5.0
    assert joined[0]["negative_count"] == 2.0


def test_feature_importance_rows_are_sorted_and_named() -> None:
    estimator = SimpleNamespace(feature_importances_=[0.1, 0.8, 0.1])

    rows = feature_importance_rows(estimator, ["a", "b", "c"], stage="post")

    assert rows[0] == {"stage": "post", "feature": "b", "importance": 0.8}


def test_action_space_from_rows_reports_only_evaluated_budgets_in_numeric_order() -> None:
    rows = [{"action": "neg5"}, {"action": "neg0"}, {"action": "neg5"}]

    assert action_space_from_rows(rows) == ["neg0", "neg5"]
