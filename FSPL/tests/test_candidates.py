import numpy as np

from fspl.candidates import build_candidate_pools, contrast_map


def test_contrast_map_rewards_foreground_and_penalizes_background() -> None:
    foreground = np.array([[4.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    background = np.array([[0.0, 5.0], [1.0, 2.0]], dtype=np.float32)

    score = contrast_map(foreground, background, lambda_bg=1.0)

    assert score[0, 0] == score.max()
    assert score[0, 1] == score.min()


def test_candidate_pools_select_distinct_positive_and_negative_coordinates() -> None:
    foreground = np.array([[4.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    background = np.array([[0.0, 5.0], [1.0, 2.0]], dtype=np.float32)

    pools = build_candidate_pools(
        foreground,
        background,
        patch_size=14,
        positive_limit=2,
        negative_limit=2,
    )

    assert pools.positives[0].coordinate == (7, 7)
    assert pools.negatives[0].coordinate == (21, 7)
    coordinates = [candidate.coordinate for candidate in (*pools.positives, *pools.negatives)]
    assert len(coordinates) == len(set(coordinates))


def test_candidate_pools_handle_flat_score_maps_without_nan() -> None:
    flat = np.ones((2, 2), dtype=np.float32)

    pools = build_candidate_pools(flat, flat, patch_size=8, positive_limit=2, negative_limit=2)

    assert len(pools.positives) == 2
    assert len(pools.negatives) == 2
    assert all(np.isfinite(candidate.score) for candidate in (*pools.positives, *pools.negatives))


def test_spread_layout_skips_adjacent_high_scoring_positive_patch() -> None:
    foreground = np.array(
        [[10.0, 9.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        dtype=np.float32,
    )
    background = np.zeros_like(foreground)

    topk = build_candidate_pools(foreground, background, patch_size=10, positive_limit=2, negative_limit=0)
    spread = build_candidate_pools(
        foreground,
        background,
        patch_size=10,
        positive_limit=2,
        negative_limit=0,
        positive_layout="spread",
        min_positive_distance_patches=2.0,
    )

    assert [candidate.coordinate for candidate in topk.positives] == [(5, 5), (15, 5)]
    assert [candidate.coordinate for candidate in spread.positives] == [(5, 5), (35, 35)]
