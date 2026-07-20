import pytest

from analyze_monusac_patient_level import aggregate_query_rows_by_patient


def test_aggregate_query_rows_by_patient_averages_patches_before_inference() -> None:
    query_rows = [
        {"query_case_id": "a0", "fixed_dice": 0.4, "learned_dice": 0.6},
        {"query_case_id": "a1", "fixed_dice": 0.6, "learned_dice": 0.7},
        {"query_case_id": "b0", "fixed_dice": 0.8, "learned_dice": 0.7},
    ]
    patient_by_case = {"a0": "pa", "a1": "pa", "b0": "pb"}

    rows = aggregate_query_rows_by_patient(query_rows, patient_by_case)

    assert rows[0]["patient"] == "pa"
    assert rows[0]["patch_count"] == 2
    assert rows[0]["fixed_dice"] == pytest.approx(0.5)
    assert rows[0]["learned_dice"] == pytest.approx(0.65)
    assert rows[0]["delta_dice"] == pytest.approx(0.15)
