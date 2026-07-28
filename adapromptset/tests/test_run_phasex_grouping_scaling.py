"""Tests for Phase X Stage S3: grouping scaling."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_grouping_scaling import (
    G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD,
    G_GRAPH_DISTANCE_THRESHOLD,
    G_SPATIAL_GRID_CELLS_PER_AXIS,
    IMAGE_SIZE,
    _connected_component_groups,
    _gate_x4_grouping_status,
    _kmeans_assign,
    _kmeans_groups,
    _spatial_grid_groups,
    split_groups,
)


class SplitGroupsTests(unittest.TestCase):
    def test_one_group_returns_single_group_with_all_indices(self) -> None:
        coords = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
        groups = split_groups("G_ONE_GROUP", 5, coords)
        self.assertEqual(groups, [(0, 1, 2, 3, 4)])

    def test_one_point_per_group_returns_k_groups_of_one(self) -> None:
        coords = [(10, 10), (20, 20), (30, 30)]
        groups = split_groups("G_ONE_POINT_PER_GROUP", 3, coords)
        self.assertEqual(groups, [(0,), (1,), (2,)])

    def test_fixed_chunk_5_splits_into_chunks_of_5(self) -> None:
        coords = [(i * 10, i * 10) for i in range(12)]
        groups = split_groups("G_FIXED_CHUNK_5", 12, coords)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0], (0, 1, 2, 3, 4))
        self.assertEqual(groups[1], (5, 6, 7, 8, 9))
        self.assertEqual(groups[2], (10, 11))

    def test_fixed_chunk_10_with_k_smaller_than_chunk_returns_single_group(self) -> None:
        coords = [(10, 10), (20, 20), (30, 30)]
        groups = split_groups("G_FIXED_CHUNK_10", 3, coords)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], (0, 1, 2))

    def test_fixed_chunk_20_with_k_equal_to_chunk_returns_single_group(self) -> None:
        coords = [(i * 5, i * 5) for i in range(20)]
        groups = split_groups("G_FIXED_CHUNK_20", 20, coords)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 20)

    def test_unknown_grouping_raises(self) -> None:
        with self.assertRaises(ValueError):
            split_groups("G_NONSENSE", 5, [(0, 0)])


class KMeansTests(unittest.TestCase):
    def test_kmeans_assign_picks_nearest_centroid(self) -> None:
        points = [(0.0, 0.0), (10.0, 10.0), (1.0, 1.0), (9.0, 9.0)]
        centroids = [(0.0, 0.0), (10.0, 10.0)]
        assignments = _kmeans_assign(points, centroids)
        self.assertEqual(assignments, [0, 1, 0, 1])

    def test_kmeans_groups_returns_requested_cluster_count_when_k_le_n(self) -> None:
        # Two well-separated clusters of 3 points each.
        coords = [(0, 0), (1, 1), (2, 2), (100, 100), (101, 101), (102, 102)]
        groups = _kmeans_groups(coords, 2)
        self.assertEqual(len(groups), 2)
        # All points in cluster 1 should be near origin, cluster 2 near (100,100).
        # Group order depends on deterministic init; check that members of each
        # cluster end up together.
        sorted_groups = sorted(
            [tuple(sorted(g)) for g in groups],
            key=lambda g: min(coords[i][0] for i in g),
        )
        self.assertEqual(sorted_groups[0], (0, 1, 2))
        self.assertEqual(sorted_groups[1], (3, 4, 5))

    def test_kmeans_groups_degrades_when_k_exceeds_n(self) -> None:
        coords = [(0, 0), (10, 10)]
        # k=5 > n=2 -> should return at most 2 groups.
        groups = _kmeans_groups(coords, 5)
        self.assertLessEqual(len(groups), 2)
        # All indices must be covered.
        all_indices = {i for g in groups for i in g}
        self.assertEqual(all_indices, {0, 1})

    def test_kmeans_groups_empty_coords_returns_empty(self) -> None:
        self.assertEqual(_kmeans_groups([], 3), [])

    def test_kmeans_groups_k_equals_one_returns_single_group(self) -> None:
        coords = [(0, 0), (10, 10), (20, 20)]
        groups = _kmeans_groups(coords, 1)
        self.assertEqual(groups, [(0, 1, 2)])


class ConnectedComponentGroupsTests(unittest.TestCase):
    def test_all_points_within_threshold_form_one_cluster(self) -> None:
        coords = [(0, 0), (10, 10), (20, 20)]
        # All within G_GRAPH_DISTANCE_THRESHOLD (80).
        groups = _connected_component_groups(coords, 80.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [0, 1, 2])

    def test_points_split_into_two_clusters(self) -> None:
        coords = [(0, 0), (5, 5), (200, 200), (210, 210)]
        groups = _connected_component_groups(coords, 80.0)
        self.assertEqual(len(groups), 2)
        # Sort groups by minimum index for stable comparison.
        sorted_groups = sorted(
            [tuple(sorted(g)) for g in groups], key=lambda g: min(g)
        )
        self.assertEqual(sorted_groups[0], (0, 1))
        self.assertEqual(sorted_groups[1], (2, 3))

    def test_candidate_component_threshold_is_tighter_than_graph(self) -> None:
        # G_CANDIDATE_COMPONENT should use a smaller threshold than G_GRAPH.
        self.assertLess(
            G_CANDIDATE_COMPONENT_DISTANCE_THRESHOLD,
            G_GRAPH_DISTANCE_THRESHOLD,
        )

    def test_empty_coords_returns_empty(self) -> None:
        self.assertEqual(_connected_component_groups([], 50.0), [])

    def test_single_point_returns_single_cluster(self) -> None:
        self.assertEqual(_connected_component_groups([(10, 10)], 50.0), [(0,)])


class SpatialGridGroupsTests(unittest.TestCase):
    def test_points_in_four_distinct_cells_form_four_groups(self) -> None:
        # With cells_per_axis=2 on a 518x518 image, cell_size = 259.
        # Place one point in each quadrant.
        coords = [
            (10, 10),     # top-left
            (300, 10),    # top-right
            (10, 300),    # bottom-left
            (300, 300),   # bottom-right
        ]
        groups = _spatial_grid_groups(coords, 2, 518)
        self.assertEqual(len(groups), 4)
        # Each group should have exactly one point.
        for g in groups:
            self.assertEqual(len(g), 1)

    def test_points_in_same_cell_form_one_group(self) -> None:
        coords = [(10, 10), (20, 20), (30, 30)]  # all in top-left cell
        groups = _spatial_grid_groups(coords, 2, 518)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [0, 1, 2])

    def test_empty_coords_returns_empty(self) -> None:
        self.assertEqual(_spatial_grid_groups([], 2, 518), [])

    def test_spatial_grid_cells_per_axis_is_two(self) -> None:
        # Frozen at preregistration time.
        self.assertEqual(G_SPATIAL_GRID_CELLS_PER_AXIS, 2)


class SplitGroupsIntegratedTests(unittest.TestCase):
    """Test split_groups for the remaining grouping families via the dispatcher."""

    def test_split_groups_kmeans_3(self) -> None:
        coords = [(i * 30, i * 30) for i in range(10)]
        groups = split_groups("G_KMEANS_3", 10, coords)
        self.assertLessEqual(len(groups), 3)
        all_indices = {i for g in groups for i in g}
        self.assertEqual(all_indices, set(range(10)))

    def test_split_groups_spatial_grid(self) -> None:
        coords = [(10, 10), (300, 300), (10, 300), (300, 10)]
        groups = split_groups("G_SPATIAL_GRID", 4, coords)
        self.assertEqual(len(groups), 4)

    def test_split_groups_graph(self) -> None:
        coords = [(0, 0), (5, 5), (200, 200), (210, 210)]
        groups = split_groups("G_GRAPH", 4, coords)
        self.assertEqual(len(groups), 2)

    def test_split_groups_candidate_component(self) -> None:
        # Tighter threshold means closer points needed to cluster.
        coords = [(0, 0), (10, 10), (200, 200), (210, 210)]
        groups = split_groups("G_CANDIDATE_COMPONENT", 4, coords)
        # With threshold 30, each pair (0,0)-(10,10) and (200,200)-(210,210)
        # is connected.
        self.assertEqual(len(groups), 2)


class GateX4GroupingStatusTests(unittest.TestCase):
    def _curve_row(
        self,
        k: int,
        grouping: str,
        mean_dice: float,
    ) -> dict:
        return {
            "k_positive": k,
            "grouping": grouping,
            "episode_count": 10,
            "mean_dice": mean_dice,
            "mean_iou": 0.3,
            "mean_aji": 0.3,
            "mean_component_recall": 0.9,
            "mean_sam_calls": 1,
            "mean_runtime_ms": 25.0,
            "mean_group_size": 20.0,
            "mean_retained_masks": 1,
        }

    def test_best_beats_one_group(self) -> None:
        curve = [
            self._curve_row(20, "G_ONE_GROUP", 0.500),
            self._curve_row(20, "G_FIXED_CHUNK_5", 0.520),
            self._curve_row(20, "G_KMEANS_5", 0.515),
        ]
        status, _ = _gate_x4_grouping_status(curve)
        self.assertEqual(status, "GROUPING_BEST_BEATS_ONE_GROUP")

    def test_one_group_dominates(self) -> None:
        curve = [
            self._curve_row(20, "G_ONE_GROUP", 0.500),
            self._curve_row(20, "G_FIXED_CHUNK_5", 0.505),  # within 0.01
            self._curve_row(20, "G_KMEANS_5", 0.498),
        ]
        status, _ = _gate_x4_grouping_status(curve)
        self.assertEqual(status, "GROUPING_ONE_GROUP_DOMINATES")

    def test_insufficient_data_missing_k20_falls_back_to_k10(self) -> None:
        curve = [
            self._curve_row(10, "G_ONE_GROUP", 0.500),
            self._curve_row(10, "G_KMEANS_5", 0.520),
        ]
        status, _ = _gate_x4_grouping_status(curve)
        # K=20 missing -> falls back to K=10 -> best beats one group.
        self.assertEqual(status, "GROUPING_BEST_BEATS_ONE_GROUP")

    def test_insufficient_data_no_k20_or_k10(self) -> None:
        curve = [
            self._curve_row(30, "G_ONE_GROUP", 0.500),
        ]
        status, _ = _gate_x4_grouping_status(curve)
        self.assertEqual(status, "GROUPING_INSUFFICIENT_DATA")

    def test_insufficient_data_missing_one_group(self) -> None:
        curve = [
            self._curve_row(20, "G_FIXED_CHUNK_5", 0.520),
            self._curve_row(20, "G_KMEANS_5", 0.515),
        ]
        status, _ = _gate_x4_grouping_status(curve)
        self.assertEqual(status, "GROUPING_INSUFFICIENT_DATA")

    def test_insufficient_data_no_valid_dice(self) -> None:
        curve = [
            {"k_positive": 20, "grouping": "G_ONE_GROUP", "mean_dice": None},
            {"k_positive": 20, "grouping": "G_KMEANS_5", "mean_dice": None},
        ]
        status, _ = _gate_x4_grouping_status(curve)
        self.assertEqual(status, "GROUPING_INSUFFICIENT_DATA")


class ImageSizeConstantTests(unittest.TestCase):
    def test_image_size_matches_sam_input(self) -> None:
        # Must match run_sam_smoke.IMAGE_SIZE (518).
        self.assertEqual(IMAGE_SIZE, 518)


if __name__ == "__main__":
    unittest.main()
