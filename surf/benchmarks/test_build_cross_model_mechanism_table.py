import pytest

from build_cross_model_mechanism_table import summarize_paired_methods


def test_summarize_paired_methods_uses_same_query_cases() -> None:
    rows = [
        {"query_case_id": "a", "method": "fg", "dice": "0.2", "object_recall": "0.5", "merge_error": "0.2", "pred_area_pct": "70"},
        {"query_case_id": "a", "method": "bg", "dice": "0.4", "object_recall": "0.7", "merge_error": "0.1", "pred_area_pct": "60"},
        {"query_case_id": "b", "method": "fg", "dice": "0.3", "object_recall": "0.6", "merge_error": "0.3", "pred_area_pct": "80"},
        {"query_case_id": "b", "method": "bg", "dice": "0.4", "object_recall": "0.8", "merge_error": "0.2", "pred_area_pct": "65"},
    ]

    summary = summarize_paired_methods(rows, "fg", "bg", bootstrap_samples=100, seed=7)

    assert summary["case_count"] == 2
    assert summary["mean_fg_dice"] == pytest.approx(0.25)
    assert summary["mean_bg_dice"] == pytest.approx(0.40)
    assert summary["mean_delta_dice"] == pytest.approx(0.15)
    assert summary["wins"] == 2
    assert summary["losses"] == 0
