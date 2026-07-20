from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def paired_rule_effect(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    baseline_rule: str,
    variant_rule: str,
    seed: int = 20260719,
    resamples: int = 10_000,
) -> dict[str, float | int | list[float]]:
    """Summarize paired variant-minus-baseline Dice effects by episode.

    Every included ``case_id`` must have exactly one row for each requested
    rule.  This prevents an unpaired or duplicated action table from being
    summarized as evidence for a rule difference.
    """

    if resamples < 1:
        raise ValueError("resamples must be positive")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rule_name = str(row["rule_name"])
        if rule_name in (baseline_rule, variant_rule):
            grouped[str(row["case_id"])][rule_name].append(float(row["utility"]))
    if not grouped:
        raise ValueError("no rows for the requested rules")

    deltas: list[float] = []
    for case_id in sorted(grouped):
        actions = grouped[case_id]
        baseline = actions[baseline_rule]
        variant = actions[variant_rule]
        if len(baseline) != 1 or len(variant) != 1:
            raise ValueError(f"case {case_id!r} must contain exactly one row for each requested rule")
        deltas.append(variant[0] - baseline[0])

    values = np.asarray(deltas, dtype=np.float64)
    mean_delta = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    generator = np.random.default_rng(seed)
    bootstrap_means = values[generator.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "episode_count": len(values),
        "mean_delta": mean_delta,
        "standard_error": standard_error,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "wins": int(np.count_nonzero(values > 0.0)),
        "ties": int(np.count_nonzero(values == 0.0)),
        "losses": int(np.count_nonzero(values < 0.0)),
    }
