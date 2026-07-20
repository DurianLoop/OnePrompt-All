from fspl.protocol import FixedRuleConfig
from fspl.stage4 import build_spatial_radius_rules, rule_candidate_kwargs


def test_stage4_passes_layout_configuration_to_candidate_builder() -> None:
    rule = FixedRuleConfig(
        "spread5_n3",
        positive_limit=5,
        negative_limit=3,
        positive_layout="spread",
        min_positive_distance_patches=3.0,
    )

    assert rule_candidate_kwargs(rule) == {
        "positive_limit": 5,
        "negative_limit": 3,
        "lambda_bg": 1.0,
        "positive_layout": "spread",
        "min_positive_distance_patches": 3.0,
    }


def test_spatial_radius_rule_factory_keeps_a_matched_prompt_budget() -> None:
    rules = build_spatial_radius_rules((2.0, 3.0, 4.0))

    assert [rule.name for rule in rules] == ["top5_n3", "spread5_n3_d2", "spread5_n3_d3", "spread5_n3_d4"]
    assert [(rule.positive_limit, rule.negative_limit) for rule in rules] == [(5, 3)] * 4
    assert [rule.min_positive_distance_patches for rule in rules[1:]] == [2.0, 3.0, 4.0]
