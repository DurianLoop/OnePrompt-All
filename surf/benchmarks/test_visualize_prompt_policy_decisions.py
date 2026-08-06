import numpy as np

from visualize_prompt_policy_decisions import pair_by_query, segmentation_error_rgb


def test_segmentation_error_rgb_uses_distinct_tp_fp_fn_colors() -> None:
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    pred = np.array([[1, 1], [0, 0]], dtype=bool)

    rgb = segmentation_error_rgb(gt, pred)

    assert rgb[0, 0].tolist() == [35, 190, 90]
    assert rgb[0, 1].tolist() == [235, 65, 55]
    assert rgb[1, 0].tolist() == [55, 125, 235]
    assert rgb[1, 1].tolist() == [245, 245, 245]


def test_pair_by_query_aligns_rows_and_computes_delta() -> None:
    fixed = [{"query_case_id": "q", "dice": "0.4"}]
    learned = [{"query_case_id": "q", "dice": "0.6"}]

    paired = pair_by_query(fixed, learned)

    assert paired[0]["query_case_id"] == "q"
    assert paired[0]["delta_dice"] == 0.2
