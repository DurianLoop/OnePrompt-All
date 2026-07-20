from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.geometry.resize_pad import build_resize_pad_transform


class ResizePadTransformTests(unittest.TestCase):
    def test_wide_image_round_trips_through_letterbox_padding(self) -> None:
        transform = build_resize_pad_transform(original_size=(400, 200), target_size=(224, 224))

        self.assertAlmostEqual(transform.scale_x, 0.56)
        self.assertAlmostEqual(transform.scale_y, 0.56)
        self.assertEqual(transform.resized_size, (224, 112))
        self.assertEqual(transform.pad_left, 0.0)
        self.assertEqual(transform.pad_top, 56.0)

        original = (100.0, 50.0)
        padded = transform.original_to_padded_point(original)
        self.assertAlmostEqual(padded[0], 56.0)
        self.assertAlmostEqual(padded[1], 84.0)
        round_trip = transform.padded_to_original_point(padded)
        self.assertAlmostEqual(round_trip[0], original[0])
        self.assertAlmostEqual(round_trip[1], original[1])

    def test_tall_image_rounds_integer_points_after_inverse_mapping(self) -> None:
        transform = build_resize_pad_transform(original_size=(100, 300), target_size=(224, 224))

        self.assertEqual(transform.resized_size, (75, 224))
        self.assertEqual(transform.pad_left, 74.5)
        self.assertEqual(transform.pad_top, 0.0)

        padded = transform.original_to_padded_point((99.0, 299.0))
        self.assertEqual(transform.padded_to_original_pixel(padded), (99, 299))

    def test_batch_point_mapping_preserves_order(self) -> None:
        transform = build_resize_pad_transform(original_size=(10, 10), target_size=(20, 30))

        points = [(0.0, 0.0), (5.0, 5.0), (9.0, 9.0)]
        mapped = transform.original_to_padded_points(points)

        self.assertEqual(mapped, [(0.0, 5.0), (10.0, 15.0), (18.0, 23.0)])
        self.assertEqual([transform.padded_to_original_pixel(point) for point in mapped], [(0, 0), (5, 5), (9, 9)])

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_resize_pad_transform(original_size=(0, 10), target_size=(224, 224))
        with self.assertRaises(ValueError):
            build_resize_pad_transform(original_size=(10, 10), target_size=(-1, 224))


if __name__ == "__main__":
    unittest.main()
