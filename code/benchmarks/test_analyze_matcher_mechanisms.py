import numpy as np
import pytest

from analyze_matcher_mechanisms import (
    paired_method_comparison,
    patch_target_labels,
    point_target_metrics,
    rank_correlation,
)


def test_point_target_metrics_reports_precision_and_component_coverage() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    points = [(1.0, 1.0), (6.0, 6.0), (0.0, 7.0)]

    metrics = point_target_metrics(mask, points)

    assert metrics["point_count"] == 3
    assert metrics["point_precision"] == pytest.approx(2 / 3)
    assert metrics["hit_component_count"] == 2
    assert metrics["gt_component_count"] == 2
    assert metrics["component_coverage"] == pytest.approx(1.0)


def test_patch_target_labels_marks_any_foreground_in_each_patch() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[3, 3] = True

    labels = patch_target_labels(mask, score_shape=(2, 2))

    assert labels.tolist() == [[True, False], [False, True]]


def test_rank_correlation_detects_monotonic_relationship() -> None:
    assert rank_correlation([10, 20, 30], [3, 2, 1]) == pytest.approx(-1.0)


def test_paired_method_comparison_uses_episode_aligned_deltas() -> None:
    rows = [
        {"episode_id": "a", "method": "base", "dice": 0.4},
        {"episode_id": "a", "method": "new", "dice": 0.6},
        {"episode_id": "b", "method": "base", "dice": 0.7},
        {"episode_id": "b", "method": "new", "dice": 0.6},
    ]

    result = paired_method_comparison(rows, "base", "new", ["dice"], bootstrap_samples=500, seed=3)

    assert result["episode_count"] == 2
    assert result["metrics"]["dice"]["baseline_mean"] == pytest.approx(0.55)
    assert result["metrics"]["dice"]["treatment_mean"] == pytest.approx(0.6)
    assert result["metrics"]["dice"]["mean_delta"] == pytest.approx(0.05)
