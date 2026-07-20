from pathlib import Path

from apply_saved_negative_budget_policy import parse_negative_policy_configuration


def test_parse_negative_policy_configuration_resolves_models_and_selection() -> None:
    model_dir = Path("policy")
    config = parse_negative_policy_configuration(
        {
            "method": "anchored_two_stage_negative_budget_policy",
            "anchor_selected_on_validation": "neg0",
            "candidate_budget": 2,
            "pre_features": ["delta_negative_count"],
            "post_features": ["delta_negative_count", "delta_sam_score"],
            "selection": {"gain_threshold": 0.01},
        },
        model_dir,
    )

    assert config["anchor_action"] == "neg0"
    assert config["candidate_budget"] == 2
    assert config["gain_threshold"] == 0.01
    assert config["pre_model_path"] == model_dir / "negative_pre_sam.joblib"
    assert config["post_model_path"] == model_dir / "negative_post_sam.joblib"
