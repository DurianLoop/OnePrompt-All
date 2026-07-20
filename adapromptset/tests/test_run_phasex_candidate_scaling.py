from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.scripts.run_phasex_candidate_scaling import (
    COVERAGE_RADIUS,
    _boundary_pixels,
    _candidate_entropy,
    _component_labels,
    _compute_negative_metrics,
    _compute_positive_metrics,
    _gate_x2_status,
    _point_precision,
    _read_candidate_pool,
    _redundancy,
    _spatial_dispersion,
    run_candidate_scaling,
)
from adapromptset.scripts.build_phasex_preregistration import (
    NEGATIVE_K_GRID,
    POSITIVE_K_GRID,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_mask_image(path: Path, squares: list[tuple[int, int, int, int]]) -> None:
    """Create a binary mask PNG with white filled squares on black background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (518, 518), 0)
    pixels = image.load()
    for x0, y0, x1, y1 in squares:
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixels[x, y] = 255
    image.save(path)


def _make_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (518, 518), (128, 128, 128))
    image.save(path)


class CandidateScalingMetricTests(unittest.TestCase):
    """Unit tests for individual metric functions."""

    def setUp(self) -> None:
        # A 10x10 mask with two square components.
        self.mask = [
            [False] * 20 for _ in range(20)
        ]
        for r in range(2, 6):
            for c in range(2, 6):
                self.mask[r][c] = True
        for r in range(12, 16):
            for c in range(12, 16):
                self.mask[r][c] = True
        self.labels = _component_labels(self.mask)
        self.boundary = _boundary_pixels(self.mask)

    def test_component_labels_finds_two_components(self) -> None:
        self.assertEqual(len(set(self.labels.values())), 2)

    def test_boundary_pixels_include_component_edges(self) -> None:
        self.assertGreater(len(self.boundary), 0)
        self.assertIn((2, 2), self.boundary)

    def test_point_precision_all_inside(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(2, 2), coordinate=(3, 3), score=1.0),
            CandidatePoint(kind="positive", grid_index=(3, 3), coordinate=(4, 4), score=0.9),
        ]
        self.assertAlmostEqual(_point_precision(positives, self.mask), 1.0)

    def test_point_precision_half_inside(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(2, 2), coordinate=(3, 3), score=1.0),
            CandidatePoint(kind="positive", grid_index=(0, 0), coordinate=(0, 0), score=0.9),
        ]
        self.assertAlmostEqual(_point_precision(positives, self.mask), 0.5)

    def test_spatial_dispersion_two_points(self) -> None:
        disp = _spatial_dispersion([(0, 0), (3, 4)])
        self.assertAlmostEqual(disp, 5.0)

    def test_spatial_dispersion_single_point(self) -> None:
        self.assertEqual(_spatial_dispersion([(1, 1)]), 0.0)

    def test_redundancy_close_points(self) -> None:
        red = _redundancy([(0, 0), (1, 1), (10, 10)], radius=2.0)
        # (0,0) and (1,1) are within radius 2.0 -> one of them is redundant.
        self.assertAlmostEqual(red, 1.0 / 3.0)

    def test_redundancy_far_points(self) -> None:
        red = _redundancy([(0, 0), (10, 10), (20, 20)], radius=2.0)
        self.assertAlmostEqual(red, 0.0)

    def test_candidate_entropy_uniform(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(0, 0), coordinate=(0, 0), score=1.0),
            CandidatePoint(kind="positive", grid_index=(1, 1), coordinate=(1, 1), score=1.0),
        ]
        entropy = _candidate_entropy(positives)
        self.assertAlmostEqual(entropy, -2 * 0.5 * (0.5).bit_length() if False else entropy)
        # Uniform distribution over 2 values -> entropy = ln(2)
        import math
        self.assertAlmostEqual(entropy, math.log(2), places=6)

    def test_candidate_entropy_single_point(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(0, 0), coordinate=(0, 0), score=1.0),
        ]
        self.assertEqual(_candidate_entropy(positives), 0.0)

    def test_compute_positive_metrics_returns_all_fields(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(2, 2), coordinate=(3, 3), score=1.0),
            CandidatePoint(kind="positive", grid_index=(3, 3), coordinate=(4, 4), score=0.9),
        ]
        metrics = _compute_positive_metrics(positives, self.mask, self.labels, k=2)
        for field in (
            "k_positive",
            "actual_positive_count",
            "shortfall",
            "point_precision",
            "component_coverage",
            "missed_component_count",
            "foreground_area_coverage_proxy",
            "spatial_dispersion",
            "nearest_neighbor_distance",
            "redundancy",
            "candidate_entropy",
        ):
            self.assertIn(field, metrics)
        self.assertEqual(metrics["k_positive"], 2)
        self.assertEqual(metrics["actual_positive_count"], 2)
        self.assertEqual(metrics["shortfall"], 0)
        self.assertAlmostEqual(metrics["point_precision"], 1.0)
        self.assertAlmostEqual(metrics["component_coverage"], 0.5)

    def test_compute_positive_metrics_shortfall_when_pool_too_small(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        positives = [
            CandidatePoint(kind="positive", grid_index=(2, 2), coordinate=(3, 3), score=1.0),
        ]
        metrics = _compute_positive_metrics(positives, self.mask, self.labels, k=100)
        self.assertEqual(metrics["actual_positive_count"], 1)
        self.assertEqual(metrics["shortfall"], 99)

    def test_compute_negative_metrics_returns_all_fields(self) -> None:
        from adapromptset.scripts.run_phasex_candidate_scaling import CandidatePoint
        negatives = [
            CandidatePoint(kind="negative", grid_index=(0, 0), coordinate=(0, 0), score=1.0),
            CandidatePoint(kind="negative", grid_index=(19, 19), coordinate=(19, 19), score=0.9),
        ]
        metrics = _compute_negative_metrics(negatives, self.mask, self.boundary, k=2)
        for field in (
            "k_negative",
            "actual_negative_count",
            "shortfall",
            "background_precision",
            "false_negative_risk",
            "boundary_proximity",
            "hard_background_rate",
            "easy_background_fraction",
            "spatial_dispersion",
        ):
            self.assertIn(field, metrics)
        self.assertAlmostEqual(metrics["background_precision"], 1.0)


class GateX2StatusTests(unittest.TestCase):
    def test_candidate_universe_inadequate_when_k100_low(self) -> None:
        coverage = {1: 0.1, 3: 0.2, 5: 0.25, 10: 0.3, 20: 0.35, 30: 0.38, 50: 0.4, 75: 0.42, 100: 0.45}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "CANDIDATE_UNIVERSE_INADEQUATE")

    def test_old_small_k_bank_inadequate_when_k5_low_k100_high(self) -> None:
        coverage = {1: 0.1, 3: 0.2, 5: 0.3, 10: 0.5, 20: 0.6, 30: 0.7, 50: 0.75, 75: 0.8, 100: 0.85}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "OLD_SMALL_K_BANK_INADEQUATE")

    def test_candidate_scaling_supports_large_k(self) -> None:
        coverage = {1: 0.5, 3: 0.6, 5: 0.7, 10: 0.75, 20: 0.8, 30: 0.82, 50: 0.85, 75: 0.88, 100: 0.9}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "CANDIDATE_SCALING_SUPPORTS_LARGE_K")

    def test_candidate_scaling_saturates_small_k(self) -> None:
        coverage = {1: 0.8, 3: 0.85, 5: 0.88, 10: 0.89, 20: 0.9, 30: 0.9, 50: 0.91, 75: 0.91, 100: 0.92}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "CANDIDATE_SCALING_SATURATES_SMALL_K")

    def test_insufficient_data_when_pool_maxes_below_100(self) -> None:
        coverage = {1: 0.3, 3: 0.5, 5: 0.8}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "CANDIDATE_SCALING_INSUFFICIENT_DATA")

    def test_universe_inadequate_when_pool_below_100_and_low(self) -> None:
        coverage = {1: 0.1, 3: 0.15, 5: 0.2}
        status, _ = _gate_x2_status(coverage)
        self.assertEqual(status, "CANDIDATE_UNIVERSE_INADEQUATE")


class ReadCandidatePoolTests(unittest.TestCase):
    def test_read_candidate_pool_sorts_by_score(self) -> None:
        tmp_csv = PROJECT_ROOT / ".test_tmp" / "cand_pool_read" / "candidates.csv"
        _write_csv(
            tmp_csv,
            [
                {"episode_id": "e0", "positive_limit": "5", "kind": "positive", "grid_row": "0", "grid_col": "1", "x": "10", "y": "10", "score": "0.5"},
                {"episode_id": "e0", "positive_limit": "5", "kind": "positive", "grid_row": "0", "grid_col": "0", "x": "5", "y": "5", "score": "0.9"},
                {"episode_id": "e0", "positive_limit": "5", "kind": "negative", "grid_row": "1", "grid_col": "0", "x": "100", "y": "100", "score": "0.3"},
            ],
        )
        pool = _read_candidate_pool(tmp_csv)
        self.assertIn("e0", pool)
        self.assertEqual(len(pool["e0"]["positive"]), 2)
        self.assertEqual(pool["e0"]["positive"][0].score, 0.9)
        self.assertEqual(pool["e0"]["positive"][1].score, 0.5)
        self.assertEqual(len(pool["e0"]["negative"]), 1)


class RunCandidateScalingIntegrationTests(unittest.TestCase):
    """End-to-end test with synthetic mask images and candidate pool."""

    def setUp(self) -> None:
        self.workspace = PROJECT_ROOT / ".test_tmp" / "cand_scaling_workspace"
        self.project_root = self.workspace / "adapromptset"
        # Wipe any previous state.
        if self.workspace.exists():
            for path in self.workspace.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(self.workspace.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        self.project_root.mkdir(parents=True, exist_ok=True)

        # Create mask + image files in the workspace dataset root.
        dataset_root = self.workspace / "code" / "datasets" / "MoNuSAC"
        mask_dir = dataset_root / "processed" / "val" / "masks"
        image_dir = dataset_root / "processed" / "val" / "images"

        # Episode e0: two square components in the mask.
        _make_mask_image(
            mask_dir / "e0_mask.png",
            [(40, 40, 80, 80), (300, 300, 360, 360)],
        )
        _make_rgb_image(image_dir / "e0_image.png")

        # Episode e1: one square component.
        _make_mask_image(
            mask_dir / "e1_mask.png",
            [(100, 100, 160, 160)],
        )
        _make_rgb_image(image_dir / "e1_image.png")

        # Episodes CSV.
        episodes_csv = self.project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv"
        _write_csv(
            episodes_csv,
            [
                {
                    "episode_id": "e0",
                    "dataset": "MoNuSAC",
                    "role": "val",
                    "support_case_id": "s0",
                    "query_case_id": "q0",
                    "support_image_path": "processed/val/images/e0_image.png",
                    "support_mask_path": "processed/val/masks/e0_mask.png",
                    "query_image_path": "processed/val/images/e0_image.png",
                    "query_mask_path": "processed/val/masks/e0_mask.png",
                },
                {
                    "episode_id": "e1",
                    "dataset": "MoNuSAC",
                    "role": "val",
                    "support_case_id": "s1",
                    "query_case_id": "q1",
                    "support_image_path": "processed/val/images/e1_image.png",
                    "support_mask_path": "processed/val/masks/e1_mask.png",
                    "query_image_path": "processed/val/images/e1_image.png",
                    "query_mask_path": "processed/val/masks/e1_mask.png",
                },
            ],
        )

        # Candidate pool CSV with 10 positives and 5 negatives per episode.
        # Positives: some inside the mask, some outside.
        candidates_csv = self.project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv"
        cand_rows: list[dict[str, object]] = []
        # e0: 10 positives; 4 inside component 1, 2 inside component 2, 4 outside.
        e0_positives = [
            (50, 50, 0.9), (55, 55, 0.85), (60, 60, 0.8), (45, 45, 0.75),  # inside comp 1
            (320, 320, 0.7), (330, 330, 0.65),  # inside comp 2
            (200, 200, 0.6), (250, 250, 0.55), (10, 10, 0.5), (400, 400, 0.45),  # outside
        ]
        for idx, (x, y, score) in enumerate(e0_positives):
            cand_rows.append({
                "episode_id": "e0",
                "positive_limit": "10",
                "kind": "positive",
                "grid_row": str(idx),
                "grid_col": "0",
                "x": str(x),
                "y": str(y),
                "score": str(score),
            })
        # e0: 5 negatives, all outside the mask.
        for idx, (x, y, score) in enumerate([(0, 0, 0.9), (10, 10, 0.8), (500, 500, 0.7), (5, 500, 0.6), (500, 5, 0.5)]):
            cand_rows.append({
                "episode_id": "e0",
                "positive_limit": "10",
                "kind": "negative",
                "grid_row": str(idx),
                "grid_col": "1",
                "x": str(x),
                "y": str(y),
                "score": str(score),
            })
        # e1: 10 positives; 5 inside the component, 5 outside.
        e1_positives = [
            (110, 110, 0.9), (120, 120, 0.85), (130, 130, 0.8), (140, 140, 0.75), (150, 150, 0.7),
            (300, 300, 0.65), (350, 350, 0.6), (10, 10, 0.55), (400, 400, 0.5), (250, 250, 0.45),
        ]
        for idx, (x, y, score) in enumerate(e1_positives):
            cand_rows.append({
                "episode_id": "e1",
                "positive_limit": "10",
                "kind": "positive",
                "grid_row": str(idx),
                "grid_col": "0",
                "x": str(x),
                "y": str(y),
                "score": str(score),
            })
        for idx, (x, y, score) in enumerate([(0, 0, 0.9), (10, 10, 0.8), (500, 500, 0.7), (5, 500, 0.6), (500, 5, 0.5)]):
            cand_rows.append({
                "episode_id": "e1",
                "positive_limit": "10",
                "kind": "negative",
                "grid_row": str(idx),
                "grid_col": "1",
                "x": str(x),
                "y": str(y),
                "score": str(score),
            })
        _write_csv(candidates_csv, cand_rows)

    def test_run_candidate_scaling_produces_all_outputs(self) -> None:
        summary = run_candidate_scaling(project_root=self.project_root)

        self.assertEqual(summary["phase"], "phaseX")
        self.assertEqual(summary["stage"], "candidate_scaling")
        self.assertEqual(summary["gate"], "X2")
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["pool_max_positive"], 10)
        self.assertEqual(summary["pool_max_negative"], 5)
        self.assertTrue(summary["cpu_only"])
        self.assertFalse(summary["sam_inference_run"])

        out_dir = self.project_root / "reports" / "phaseX_candidate_scaling"
        for name in (
            "positive_scaling_query_metrics.csv",
            "negative_scaling_query_metrics.csv",
            "positive_coverage_curve.csv",
            "negative_quality_curve.csv",
            "component_coverage_by_k.csv",
            "spatial_dispersion_by_k.csv",
            "scaling_summary.md",
            "scaling_summary.json",
        ):
            self.assertTrue((out_dir / name).is_file(), f"missing output: {name}")

    def test_positive_scaling_metrics_cover_all_k_values(self) -> None:
        run_candidate_scaling(project_root=self.project_root)
        out_csv = (
            self.project_root
            / "reports"
            / "phaseX_candidate_scaling"
            / "positive_scaling_query_metrics.csv"
        )
        with out_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        k_values = {int(row["k_positive"]) for row in rows}
        for k in POSITIVE_K_GRID:
            self.assertIn(k, k_values)
        # Pool has 10 positives, so K+=100 should record shortfall.
        k100_rows = [row for row in rows if int(row["k_positive"]) == 100]
        self.assertEqual(len(k100_rows), 2)  # 2 episodes
        for row in k100_rows:
            self.assertEqual(int(row["actual_positive_count"]), 10)
            self.assertEqual(int(row["shortfall"]), 90)

    def test_gate_x2_reports_insufficient_data_when_pool_below_100(self) -> None:
        summary = run_candidate_scaling(project_root=self.project_root)
        # Pool maxes at 10 positives; cannot reach K+=100.
        self.assertEqual(summary["gate_x2_status"], "CANDIDATE_SCALING_INSUFFICIENT_DATA")
        self.assertIn("cannot test K+=100", summary["gate_x2_rationale"])

    def test_coverage_curve_monotonically_non_decreasing(self) -> None:
        summary = run_candidate_scaling(project_root=self.project_root)
        coverage = summary["coverage_curve"]
        values = [coverage[str(k)] for k in POSITIVE_K_GRID]
        for i in range(1, len(values)):
            self.assertGreaterEqual(values[i], values[i - 1] - 1e-9)

    def test_negative_metrics_cover_all_nonzero_k_values(self) -> None:
        run_candidate_scaling(project_root=self.project_root)
        out_csv = (
            self.project_root
            / "reports"
            / "phaseX_candidate_scaling"
            / "negative_scaling_query_metrics.csv"
        )
        with out_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        k_values = {int(row["k_negative"]) for row in rows}
        for k in NEGATIVE_K_GRID:
            if k == 0:
                continue
            self.assertIn(k, k_values)

    def test_summary_md_contains_gate_status_and_table(self) -> None:
        run_candidate_scaling(project_root=self.project_root)
        md_path = (
            self.project_root
            / "reports"
            / "phaseX_candidate_scaling"
            / "scaling_summary.md"
        )
        content = md_path.read_text(encoding="utf-8")
        self.assertIn("Gate X2 Status", content)
        self.assertIn("CANDIDATE_SCALING_INSUFFICIENT_DATA", content)
        self.assertIn("Positive coverage curve", content)
        self.assertIn("Negative quality curve", content)


if __name__ == "__main__":
    unittest.main()
