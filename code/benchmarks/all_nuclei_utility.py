from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any, Iterable


@dataclass(frozen=True)
class AllNucleiWeights:
    object_recall: float = 0.2
    area_error: float = 0.1
    merge_error: float = 0.3

    def __post_init__(self) -> None:
        if min(self.object_recall, self.area_error, self.merge_error) < 0.0:
            raise ValueError("All all-nuclei utility coefficients must be non-negative.")


def relative_area_error(fg_precision: float, fg_recall: float, epsilon: float = 1e-8) -> float:
    """Return symmetric log area-ratio error using |P|/|G| = recall/precision."""
    precision = max(float(fg_precision), epsilon)
    recall = max(float(fg_recall), epsilon)
    return abs(log(recall / precision))


def add_all_nuclei_utility(
    rows: Iterable[dict[str, Any]],
    weights: AllNucleiWeights,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        area_error = relative_area_error(
            fg_precision=float(row["fg_precision"]),
            fg_recall=float(row["fg_recall"]),
        )
        utility = (
            float(row["dice"])
            + weights.object_recall * float(row["object_recall"])
            - weights.area_error * area_error
            - weights.merge_error * float(row["merge_error"])
        )
        enriched.append({
            **row,
            "area_error": area_error,
            "all_nuclei_utility": utility,
        })
    return enriched
