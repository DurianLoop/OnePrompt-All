import math

import pytest

from all_nuclei_utility import (
    AllNucleiWeights,
    add_all_nuclei_utility,
    relative_area_error,
)


def test_relative_area_error_is_zero_when_predicted_and_gt_areas_match() -> None:
    assert relative_area_error(fg_precision=0.8, fg_recall=0.8) == pytest.approx(0.0)


def test_relative_area_error_is_symmetric_for_double_or_half_area() -> None:
    over = relative_area_error(fg_precision=0.4, fg_recall=0.8)
    under = relative_area_error(fg_precision=0.8, fg_recall=0.4)

    assert over == pytest.approx(math.log(2.0))
    assert under == pytest.approx(math.log(2.0))


def test_all_nuclei_utility_combines_overlap_recall_area_and_merge_terms() -> None:
    rows = [{
        "dice": "0.60",
        "object_recall": "0.80",
        "fg_precision": "0.40",
        "fg_recall": "0.80",
        "merge_error": "0.10",
    }]
    weights = AllNucleiWeights(object_recall=0.2, area_error=0.1, merge_error=0.3)

    enriched = add_all_nuclei_utility(rows, weights)

    expected = 0.60 + 0.2 * 0.80 - 0.1 * math.log(2.0) - 0.3 * 0.10
    assert enriched[0]["area_error"] == pytest.approx(math.log(2.0))
    assert enriched[0]["all_nuclei_utility"] == pytest.approx(expected)


def test_all_nuclei_weights_reject_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AllNucleiWeights(object_recall=-0.1, area_error=0.1, merge_error=0.1)
