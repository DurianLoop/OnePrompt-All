from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Point = tuple[float, float]
Pixel = tuple[int, int]
Size = tuple[int, int]


def _validate_size(name: str, size: Size) -> None:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive width and height")


@dataclass(frozen=True)
class ResizePadTransform:
    original_size: Size
    target_size: Size
    resized_size: Size
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float

    def original_to_padded_point(self, point: Point) -> Point:
        x, y = point
        return (x * self.scale_x + self.pad_left, y * self.scale_y + self.pad_top)

    def padded_to_original_point(self, point: Point) -> Point:
        x, y = point
        return ((x - self.pad_left) / self.scale_x, (y - self.pad_top) / self.scale_y)

    def original_to_padded_points(self, points: Iterable[Point]) -> list[Point]:
        return [self.original_to_padded_point(point) for point in points]

    def padded_to_original_pixel(self, point: Point) -> Pixel:
        x, y = self.padded_to_original_point(point)
        width, height = self.original_size
        px = min(width - 1, max(0, int(round(x))))
        py = min(height - 1, max(0, int(round(y))))
        return px, py


def build_resize_pad_transform(original_size: Size, target_size: Size) -> ResizePadTransform:
    _validate_size("original_size", original_size)
    _validate_size("target_size", target_size)
    original_width, original_height = original_size
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_left = (target_width - resized_width) / 2.0
    pad_top = (target_height - resized_height) / 2.0
    return ResizePadTransform(
        original_size=original_size,
        target_size=target_size,
        resized_size=(resized_width, resized_height),
        scale_x=scale,
        scale_y=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
