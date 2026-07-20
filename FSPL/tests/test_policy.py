import numpy as np

from fspl.candidates import build_candidate_pools
from fspl.policy import build_action_features, choose_actions, fit_frozen_utility_policy, fit_utility_policy
from fspl.protocol import FixedRuleConfig


def test_action_features_are_finite_and_contain_action_metadata() -> None:
    foreground = np.array([[0.1, 0.4], [0.5, 0.2]], dtype=np.float32)
    background = np.array([[0.2, 0.1], [0.3, 0.6]], dtype=np.float32)
    rule = FixedRuleConfig("top1_n1", positive_limit=1, negative_limit=1)
    pools = build_candidate_pools(foreground, background, patch_size=14, positive_limit=1, negative_limit=1)

    features = build_action_features(foreground, background, pools, rule)

    assert features["action_positive_count"] == 1.0
    assert features["action_negative_count"] == 1.0
    assert all(np.isfinite(value) for value in features.values())


def test_action_features_expose_spread_layout_geometry() -> None:
    foreground = np.array([[10.0, 9.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 8.0]], dtype=np.float32)
    background = np.zeros_like(foreground)
    rule = FixedRuleConfig(
        "spread5_n3",
        positive_limit=2,
        negative_limit=0,
        positive_layout="spread",
        min_positive_distance_patches=2.0,
    )
    pools = build_candidate_pools(
        foreground,
        background,
        patch_size=14,
        positive_limit=2,
        negative_limit=0,
        positive_layout=rule.positive_layout,
        min_positive_distance_patches=rule.min_positive_distance_patches,
    )

    features = build_action_features(foreground, background, pools, rule)

    assert features["positive_layout_is_spread"] == 1.0
    assert features["positive_mean_nearest_distance_patches"] > 2.0


def test_choose_actions_prefers_score_then_lower_prompt_cost() -> None:
    actions = [
        {"case_id": "case_a", "rule_name": "top5_n3", "predicted_utility": 0.8, "prompt_count": 8},
        {"case_id": "case_a", "rule_name": "top3_n0", "predicted_utility": 0.8, "prompt_count": 3},
    ]

    selected = choose_actions(actions)

    assert selected == {"case_a": "top3_n0"}


def test_utility_policy_exposes_nonlinear_ranker_for_action_interactions() -> None:
    train = [
        {"case_id": f"train_{case}", "rule_name": rule, "signal": float(case), "action": float(action), "utility": float((case % 2) == action), "prompt_count": action + 1}
        for case in range(6)
        for action, rule in enumerate(("a", "b"))
    ]
    validation = [
        {"case_id": f"validation_{case}", "rule_name": rule, "signal": float(case), "action": float(action), "utility": float((case % 2) == action), "prompt_count": action + 1}
        for case in range(6, 8)
        for action, rule in enumerate(("a", "b"))
    ]

    fit = fit_utility_policy(train, validation, feature_names=("signal", "action"))

    assert fit.regressor_name == "ExtraTreesRegressor"


def test_frozen_policy_keeps_selected_leaf_size() -> None:
    rows = [
        {"signal": float(index), "utility": float(index % 2)}
        for index in range(8)
    ]

    estimator = fit_frozen_utility_policy(rows, feature_names=("signal",), min_samples_leaf=2)

    assert estimator.min_samples_leaf == 2
