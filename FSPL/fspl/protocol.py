from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .dataset import CaseRecord


@dataclass(frozen=True)
class FixedRuleConfig:
    name: str
    positive_limit: int
    negative_limit: int
    lambda_bg: float = 1.0
    positive_layout: str = "topk"
    min_positive_distance_patches: float = 3.0

    @property
    def prompt_count(self) -> int:
        return self.positive_limit + self.negative_limit


@dataclass(frozen=True)
class FixedBaselineProtocol:
    seed: int
    reference: CaseRecord
    validation_cases: tuple[CaseRecord, ...]
    test_cases: tuple[CaseRecord, ...]


@dataclass(frozen=True)
class RuleSelection:
    selected: FixedRuleConfig
    mean_dice_by_rule: dict[str, float]


def build_fixed_baseline_protocol(
    train_cases: Sequence[CaseRecord],
    test_cases: Sequence[CaseRecord],
    *,
    seed: int,
    validation_count: int,
) -> FixedBaselineProtocol:
    ordered_train = sorted(train_cases, key=lambda case: case.case_id)
    ordered_test = tuple(sorted(test_cases, key=lambda case: case.case_id))
    if validation_count < 1:
        raise ValueError("validation_count must be positive")
    if len(ordered_train) < validation_count + 1:
        raise ValueError("train cases must contain one reference plus validation cases")
    if any(case.split != "train" for case in ordered_train):
        raise ValueError("reference and validation cases must come from the train split")
    if any(case.split != "test" for case in ordered_test):
        raise ValueError("test cases must come from the test split")

    shuffled = list(ordered_train)
    random.Random(seed).shuffle(shuffled)
    return FixedBaselineProtocol(
        seed=seed,
        reference=shuffled[0],
        validation_cases=tuple(sorted(shuffled[1 : validation_count + 1], key=lambda case: case.case_id)),
        test_cases=ordered_test,
    )


def select_primary_rule(
    dice_by_rule: Mapping[str, Sequence[float]], rules: Sequence[FixedRuleConfig]
) -> RuleSelection:
    if not rules:
        raise ValueError("at least one fixed rule is required")
    mean_dice_by_rule: dict[str, float] = {}
    for rule in rules:
        values = dice_by_rule.get(rule.name)
        if not values:
            raise ValueError(f"missing validation Dice values for rule {rule.name}")
        mean_dice_by_rule[rule.name] = round(sum(values) / len(values), 12)
    selected = min(
        rules,
        key=lambda rule: (-mean_dice_by_rule[rule.name], rule.prompt_count, rule.name),
    )
    return RuleSelection(selected=selected, mean_dice_by_rule=mean_dice_by_rule)
