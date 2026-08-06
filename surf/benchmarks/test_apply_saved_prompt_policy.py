import json
from pathlib import Path

from apply_saved_prompt_policy import load_policy_configuration


def test_load_policy_configuration_keeps_frozen_action_grid_and_validation_gate() -> None:
    model_dir = Path(__file__).with_name("_saved_policy_config_fixture")
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps({
        "fallback_action": "contrast_topk_K40_lam0.75",
        "two_stage_candidate_budget": 3,
        "anchored_pre_sam_model_path": "pre.joblib",
        "anchored_post_sam_model_path": "post.joblib",
        "anchored_gain_threshold_selected_on_validation": 0.01,
        "anchored_uncertainty_weight_selected_on_validation": 0.0,
        "anchored_pre_min_samples_leaf_selected_on_validation": 2,
        "anchored_post_min_samples_leaf_selected_on_validation": 4,
        "anchor_pre_features": ["delta_k"],
        "anchor_post_features": ["delta_k", "delta_sam_score"],
        "allowed_rounds": ["round2"],
        "allowed_k": [5, 10, 20, 40, 75],
        "allowed_lambda": [0.25, 0.75, 1.5],
    }), encoding="utf-8")

    config = load_policy_configuration(model_dir)

    assert config["fallback_action"] == "contrast_topk_K40_lam0.75"
    assert config["gain_threshold"] == 0.01
    assert config["allowed_rounds"] == {"round2"}
    assert config["allowed_k"] == {5, 10, 20, 40, 75}
    assert config["allowed_lambda"] == {0.25, 0.75, 1.5}
    assert config["pre_model_path"] == model_dir / "pre.joblib"
