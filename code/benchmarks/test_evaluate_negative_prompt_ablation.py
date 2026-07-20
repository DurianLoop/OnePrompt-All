import io

import numpy as np

from evaluate_negative_prompt_ablation import (
    artifact_stem,
    completed_negative_budget_keys,
    pending_negative_counts,
    read_csv,
    select_backend_loader,
    select_negative_points,
    summarize,
)


def test_select_negative_points_uses_lowest_contrast_patches_away_from_positives() -> None:
    score = np.asarray([
        [-5.0, -4.0, 2.0],
        [-3.0, 1.0, 3.0],
        [0.0, 4.0, 5.0],
    ])
    positives = np.asarray([[25.0, 25.0]], dtype=np.float32)

    points = select_negative_points(
        score,
        original_size=(30, 30),
        count=2,
        positive_coords=positives,
        min_distance=6.0,
    )

    assert points.shape == (2, 2)
    assert points[0].tolist() == [5.0, 5.0]
    assert points[1].tolist() == [15.0, 5.0]


def test_select_negative_points_returns_empty_array_for_zero_budget() -> None:
    points = select_negative_points(
        np.zeros((2, 2)),
        original_size=(20, 20),
        count=0,
        positive_coords=np.empty((0, 2)),
    )

    assert points.shape == (0, 2)


def test_artifact_stem_keeps_reference_episodes_distinct() -> None:
    first = artifact_stem("support_a_to_query", "query", 5)
    second = artifact_stem("support_b_to_query", "query", 5)

    assert first != second
    assert first.endswith("__neg5")


def test_pending_negative_counts_skips_only_completed_episode_settings() -> None:
    completed = {("support_a_to_query", 0), ("support_a_to_query", 5)}

    assert pending_negative_counts("support_a_to_query", [0, 5, 10], completed) == [10]
    assert pending_negative_counts("support_b_to_query", [0, 5, 10], completed) == [0, 5, 10]


def test_completed_keys_use_requested_budget_not_actual_point_count() -> None:
    rows = [{
        "episode_id": "support_a_to_query",
        "requested_negative_count": "5",
        "negative_count": "2",
    }]

    assert completed_negative_budget_keys(rows) == {("support_a_to_query", 5)}


def test_summary_groups_requested_budget_and_reports_actual_count() -> None:
    rows = [
        {
            "requested_negative_count": "5",
            "negative_count": "2",
            "dice": "0.6",
            "iou": "0.4",
            "object_recall": "0.8",
            "merge_error": "0.1",
            "pred_area_pct": "50",
        },
        {
            "requested_negative_count": "5",
            "negative_count": "5",
            "dice": "0.8",
            "iou": "0.6",
            "object_recall": "1.0",
            "merge_error": "0.0",
            "pred_area_pct": "40",
        },
    ]

    summary = summarize(rows)

    assert summary == [{
        "requested_negative_count": 5,
        "case_count": 2,
        "mean_actual_negative_count": 3.5,
        "mean_dice": 0.7,
        "mean_iou": 0.5,
        "mean_object_recall": 0.9,
        "mean_merge_error": 0.05,
        "mean_pred_area_pct": 45.0,
    }]


def test_read_csv_accepts_windows_utf8_bom() -> None:
    class BomCsv:
        def open(self, *, newline: str, encoding: str):
            assert newline == ""
            return io.TextIOWrapper(
                io.BytesIO(b"\xef\xbb\xbfpilot_id,query_case_id\nepisode_1,query_1\n"),
                encoding=encoding,
                newline=newline,
            )

    assert read_csv(BomCsv()) == [{"pilot_id": "episode_1", "query_case_id": "query_1"}]


def test_select_backend_loader_routes_supported_segmenters() -> None:
    classic = object()
    medical = object()

    assert select_backend_loader("sam", {"sam": classic, "sammed2d": medical}) is classic
    assert select_backend_loader("sammed2d", {"sam": classic, "sammed2d": medical}) is medical


def test_select_backend_loader_rejects_unknown_segmenter() -> None:
    try:
        select_backend_loader("unknown", {"sam": object()})
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Unknown backends must fail explicitly")
