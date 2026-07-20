from __future__ import annotations

import random
from typing import Mapping, Sequence


def select_qualitative_case_ids(case_ids: Sequence[str], *, count: int, seed: int) -> tuple[str, ...]:
    """Choose reproducible figure cases before any method result is viewed."""
    ordered = sorted(set(case_ids))
    if count < 1 or count > len(ordered):
        raise ValueError("count must be between one and the number of unique case IDs")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    return tuple(sorted(shuffled[:count]))


def build_panel_manifest(
    case_ids: Sequence[str], reference_case_ids: Sequence[str], selections: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    """Define all qualitative panels without using outcomes to select a case.

    A panel contains each predeclared query with every fixed reference.  The
    policy rule comes from a frozen selection CSV, while the matched Top-5
    baseline remains visible for every reference.  Utilities and Dice scores
    are intentionally excluded so the manifest cannot be used to choose a
    favorable visual result.
    """

    ordered_cases = sorted(set(case_ids))
    ordered_references = sorted(set(reference_case_ids))
    if not ordered_cases or not ordered_references:
        raise ValueError("at least one qualitative case and reference are required")
    selected_rules: dict[tuple[str, str], str] = {}
    for row in selections:
        episode = str(row["case_id"])
        if "::" not in episode:
            raise ValueError(f"selection case_id must be reference::query: {episode!r}")
        reference_case_id, query_case_id = episode.split("::", maxsplit=1)
        key = (reference_case_id, query_case_id)
        if key in selected_rules:
            raise ValueError(f"duplicate frozen policy selection for {episode!r}")
        selected_rules[key] = str(row["rule_name"])

    panels = []
    for query_case_id in ordered_cases:
        reference_panels = []
        for reference_case_id in ordered_references:
            key = (reference_case_id, query_case_id)
            if key not in selected_rules:
                raise ValueError(f"missing frozen policy selection for {reference_case_id}::{query_case_id}")
            reference_panels.append(
                {
                    "reference_case_id": reference_case_id,
                    "baseline_rule": "top5_n3",
                    "selected_policy_rule": selected_rules[key],
                }
            )
        panels.append({"query_case_id": query_case_id, "reference_panels": reference_panels})
    return {
        "selection_rule": "predeclared query cases with every fixed reference; no utility-based filtering",
        "case_ids": ordered_cases,
        "reference_case_ids": ordered_references,
        "panels": panels,
    }
