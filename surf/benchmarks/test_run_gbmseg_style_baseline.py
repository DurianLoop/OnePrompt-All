import numpy as np

from run_gbmseg_style_baseline import cycle_consistent_matches, patch_indices_from_mask


def test_cycle_consistent_matches_keeps_queries_that_map_back_to_reference_region() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.asarray([[0.9, 0.1], [0.1, 0.9], [-0.9, 0.1]], dtype=np.float32)

    kept, rejected = cycle_consistent_matches(reference, query, np.asarray([0, 1]))

    assert kept.tolist() == [0, 1]
    assert rejected.tolist() == []


def test_patch_indices_from_mask_separates_full_foreground_and_background_patches() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True

    foreground, background = patch_indices_from_mask(mask, grid_shape=(2, 2), foreground_threshold=0.99, background_threshold=0.01)

    assert foreground.tolist() == [0]
    assert background.tolist() == [1, 2, 3]
