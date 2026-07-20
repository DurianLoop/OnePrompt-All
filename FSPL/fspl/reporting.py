from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def summarize_rule_rows(
    rows: Sequence[Mapping[str, float | int | str]], *, expected_episodes: int
) -> dict[str, dict[str, float | int]]:
    """Aggregate rule-wise Dice after enforcing a balanced episode protocol."""
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rule_name"])].append(float(row["utility"]))
    if not grouped:
        raise ValueError("at least one action row is required")
    summary: dict[str, dict[str, float | int]] = {}
    for rule_name in sorted(grouped):
        values = grouped[rule_name]
        if len(values) != expected_episodes:
            raise ValueError(
                f"rule {rule_name!r} has {len(values)} episodes; expected {expected_episodes} episodes"
            )
        summary[rule_name] = {
            "case_count": len(values),
            "mean_dice": round(sum(values) / len(values), 12),
        }
    return summary
