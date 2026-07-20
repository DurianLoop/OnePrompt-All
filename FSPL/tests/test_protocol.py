from pathlib import Path

from fspl.dataset import CaseRecord
from fspl.protocol import (
    FixedRuleConfig,
    build_fixed_baseline_protocol,
    select_primary_rule,
)


def _case(case_id: str, split: str = "train") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        image_path=Path(f"{case_id}.tif"),
        mask_path=Path(f"{case_id}.png"),
        split=split,
    )


def test_protocol_is_deterministic_and_keeps_test_cases_untouched() -> None:
    train = [_case(f"train_{index:02d}") for index in range(10)]
    test = [_case(f"test_{index:02d}", split="test") for index in range(3)]

    first = build_fixed_baseline_protocol(train, test, seed=9, validation_count=3)
    second = build_fixed_baseline_protocol(train, test, seed=9, validation_count=3)

    assert first == second
    assert first.reference not in first.validation_cases
    assert {case.case_id for case in first.validation_cases}.isdisjoint(
        {case.case_id for case in first.test_cases}
    )
    assert tuple(case.case_id for case in first.test_cases) == ("test_00", "test_01", "test_02")


def test_primary_rule_uses_validation_dice_then_prompt_cost_tie_break() -> None:
    sparse = FixedRuleConfig("top3_n0", positive_limit=3, negative_limit=0)
    dense = FixedRuleConfig("top5_n3", positive_limit=5, negative_limit=3)
    result = select_primary_rule(
        {sparse.name: [0.8, 0.9], dense.name: [0.8, 0.9]},
        [dense, sparse],
    )

    assert result.selected == sparse
    assert result.mean_dice_by_rule == {"top3_n0": 0.85, "top5_n3": 0.85}
