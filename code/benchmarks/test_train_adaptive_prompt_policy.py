import numpy as np
import pytest
import csv
from pathlib import Path
from PIL import Image

from train_adaptive_prompt_policy import (
    POST_SAM_FEATURES,
    add_within_case_policy_targets,
    add_anchor_relative_features,
    apply_two_stage_policy,
    apply_anchored_two_stage_policy,
    build_case_features,
    build_candidate_features,
    build_oracle_actions,
    candidate_feature_keys,
    predict_nearest_centroid,
    select_policy_candidates,
    score_map_statistics,
    select_ridge_candidates,
    select_top_n_candidates,
    select_top_n_with_anchor,
    select_anchor_gated_candidates,
    summarize_selected_candidates,
    predict_ridge_reward,
    point_geometry_statistics,
    mask_matcher_consistency_statistics,
    train_ridge_reward,
    train_extra_trees_reward,
    predict_extra_trees_reward,
    predict_extra_trees_distribution,
    train_nearest_centroid,
    filter_candidate_rounds,
    filter_candidate_grid,
    validation_best_action,
    write_csv,
)


def test_write_csv_preserves_fields_that_only_appear_in_later_rows() -> None:
    fixture_root = Path(__file__).with_name("_csv_writer_fixture")
    output_path = fixture_root / "heterogeneous_rows.csv"

    write_csv(output_path, [
        {"episode_id": "a", "action": "anchor"},
        {"episode_id": "b", "action": "learned", "policy_best_action": "K40"},
    ])

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["policy_best_action"] == ""
    assert rows[1]["policy_best_action"] == "K40"


def candidate(case_id: str, method: str, dice: float, k: int, lam: float) -> dict[str, str]:
    return {
        "query_case_id": case_id,
        "method": method,
        "dice": str(dice),
        "k": str(k),
        "lambda_bg": str(lam),
        "layout": "topk",
        "clustering": "none",
        "points_path": "missing.csv",
    }


def test_build_oracle_actions_picks_one_highest_dice_action_per_case() -> None:
    rows = [
        candidate("case_a", "contrast_topk_K20_lam0.75", 0.42, 20, 0.75),
        candidate("case_a", "contrast_topk_K40_lam0.75", 0.68, 40, 0.75),
        candidate("case_b", "contrast_topk_K20_lam0.75", 0.71, 20, 0.75),
        candidate("case_b", "contrast_topk_K40_lam0.75", 0.53, 40, 0.75),
    ]

    oracle = build_oracle_actions(rows)

    assert {row["query_case_id"] for row in oracle} == {"case_a", "case_b"}
    assert {row["query_case_id"]: row["oracle_method"] for row in oracle} == {
        "case_a": "contrast_topk_K40_lam0.75",
        "case_b": "contrast_topk_K20_lam0.75",
    }
    assert candidate_feature_keys() == ["k", "lambda_bg", "layout", "clustering"]


def test_score_map_statistics_describes_score_concentration_without_gt_or_sam() -> None:
    score_map = np.asarray([[0.0, 0.0], [1.0, 3.0]], dtype=np.float32)

    stats = score_map_statistics(score_map)

    assert stats["score_mean"] == 1.0
    assert stats["score_std"] > 1.0
    assert stats["top_quarter_mean"] == 3.0
    assert 0.0 <= stats["normalized_entropy"] <= 1.0


def test_nearest_centroid_predicts_action_from_matcher_only_features() -> None:
    rows = [
        {"action": "K20", "score_mean": 0.1, "score_std": 0.2},
        {"action": "K20", "score_mean": 0.2, "score_std": 0.1},
        {"action": "K75", "score_mean": 2.0, "score_std": 1.8},
        {"action": "K75", "score_mean": 1.8, "score_std": 2.1},
    ]
    model = train_nearest_centroid(rows, feature_names=["score_mean", "score_std"], label_key="action")

    prediction = predict_nearest_centroid(model, {"score_mean": 1.9, "score_std": 1.9})

    assert prediction == "K75"


def test_build_case_features_reads_pre_sam_score_map() -> None:
    fixture_root = Path(__file__).with_name("_policy_feature_fixture")
    case_root = fixture_root / "cases" / "case_a"
    score_dir = case_root / "t1_matching"
    points_dir = case_root / "points"
    score_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(exist_ok=True)
    score_path = score_dir / "round2_contrast_topk_K1_lam0.75_score.npy"
    np.save(score_path, np.asarray([[1.0, 3.0]], dtype=np.float32))
    rows = [{
        "query_case_id": "case_a",
        "round_name": "round2",
        "k": "1",
        "lambda_bg": "0.75",
        "points_path": str(points_dir / "round2_contrast_topk_K1_lam0.75_points.csv"),
    }]

    features = build_case_features(rows, workspace_root=fixture_root)

    assert len(features) == 1
    assert features[0]["query_case_id"] == "case_a"
    assert features[0]["score_mean"] == 2.0
    assert features[0]["top_quarter_mean"] == 3.0


def test_select_policy_candidates_uses_predicted_action_with_fixed_fallback() -> None:
    rows = [
        candidate("case_a", "K20", 0.2, 20, 0.75),
        candidate("case_a", "K40", 0.7, 40, 0.75),
        candidate("case_b", "K20", 0.6, 20, 0.75),
    ]

    selected = select_policy_candidates(rows, {"case_a": "K40", "case_b": "missing"}, fallback_action="K20")

    assert [(row["query_case_id"], row["selected_action"], row["selection_source"]) for row in selected] == [
        ("case_a", "K40", "policy"),
        ("case_b", "K20", "fallback"),
    ]


def test_summarize_selected_candidates_reports_one_mask_per_case_metrics() -> None:
    summary = summarize_selected_candidates([
        {"query_case_id": "case_a", "dice": "0.4", "iou": "0.3", "object_recall": "0.5", "merge_error": "0.1"},
        {"query_case_id": "case_b", "dice": "0.6", "iou": "0.5", "object_recall": "0.7", "merge_error": "0.3"},
    ])

    assert summary == {
        "case_count": 2,
        "candidates_per_case": 1,
        "mean_dice": 0.5,
        "mean_iou": 0.4,
        "mean_object_recall": 0.6,
        "mean_merge_error": 0.2,
    }


def test_module_entrypoint_is_after_oracle_helper_definition() -> None:
    source = Path(__file__).with_name("train_adaptive_prompt_policy.py").read_text(encoding="utf-8")

    assert source.index('if __name__ == "__main__":') > source.index("def build_oracle_actions")


def test_ridge_reward_ranks_higher_matcher_feature_above_lower_one() -> None:
    rows = [
        {"score_max": 0.0, "k": 20.0, "dice": 0.1},
        {"score_max": 1.0, "k": 20.0, "dice": 0.5},
        {"score_max": 2.0, "k": 20.0, "dice": 0.9},
    ]
    model = train_ridge_reward(rows, feature_names=["score_max", "k"], target_key="dice", alpha=0.01)

    low = predict_ridge_reward(model, {"score_max": 0.2, "k": 20.0})
    high = predict_ridge_reward(model, {"score_max": 1.8, "k": 20.0})

    assert high > low


def test_build_candidate_features_keeps_action_and_uses_score_map() -> None:
    fixture_root = Path(__file__).with_name("_policy_candidate_fixture")
    case_root = fixture_root / "cases" / "case_b"
    score_dir = case_root / "t1_matching"
    points_dir = case_root / "points"
    score_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(exist_ok=True)
    np.save(score_dir / "round2_contrast_topk_K20_lam0.75_score.npy", np.asarray([[2.0, 4.0]], dtype=np.float32))
    points_path = points_dir / "round2_contrast_topk_K20_lam0.75_points.csv"
    points_path.write_text("x,y,score,s_fg,s_bg\n0,0,0.2,0.3,0.1\n3,4,0.8,0.9,0.2\n", encoding="utf-8")
    Image.fromarray(np.asarray([[255, 0], [255, 0]], dtype=np.uint8)).save(fixture_root / "mask.png")
    rows = [{
        "query_case_id": "case_b", "method": "contrast_topk_K20_lam0.75", "round_name": "round2",
        "case_name": "support_x_to_case_b",
        "k": "20", "lambda_bg": "0.75", "layout": "topk", "clustering": "none", "dice": "0.6",
        "iou": "0.5", "object_recall": "0.7", "merge_error": "0.1",
        "sam_score": "0.83", "pred_area_pct": "12.5", "pred_components": "17", "prompt_count": "20",
        "mask_path": "mask.png", "overlay_path": "overlay.png",
        "points_path": str(points_path),
    }]

    features = build_candidate_features(rows, workspace_root=fixture_root)

    assert features[0]["action"] == "contrast_topk_K20_lam0.75"
    assert features[0]["episode_id"] == "support_x_to_case_b"
    assert features[0]["score_mean"] == 3.0
    assert features[0]["layout_topk"] == 1.0
    assert features[0]["dice"] == 0.6
    assert features[0]["iou"] == 0.5
    assert features[0]["point_pairwise_distance_mean"] == pytest.approx(0.005)
    assert features[0]["sam_score"] == pytest.approx(0.83)
    assert features[0]["pred_area_pct"] == pytest.approx(12.5)
    assert features[0]["pred_components"] == pytest.approx(17.0)
    assert features[0]["prompt_count"] == pytest.approx(20.0)
    assert features[0]["mask_path"] == "mask.png"
    assert features[0]["overlay_path"] == "overlay.png"
    assert features[0]["points_path"] == str(points_path)
    assert features[0]["mask_artifact_available"] == 1.0
    assert {"dice", "iou", "object_recall", "merge_error"}.isdisjoint(POST_SAM_FEATURES)

    rows[0]["mask_path"] = "missing_mask.png"
    missing_features = build_candidate_features(rows, workspace_root=fixture_root)
    assert missing_features[0]["mask_artifact_available"] == 0.0
    assert missing_features[0]["mask_score_contrast"] == 0.0


def test_select_ridge_candidates_keeps_highest_predicted_reward_per_case() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "A", "predicted_reward": 0.2, "dice": 0.1},
        {"query_case_id": "case_a", "action": "B", "predicted_reward": 0.8, "dice": 0.7},
        {"query_case_id": "case_b", "action": "A", "predicted_reward": 0.9, "dice": 0.5},
        {"query_case_id": "case_b", "action": "B", "predicted_reward": 0.1, "dice": 0.6},
    ]

    selected = select_ridge_candidates(rows)

    assert [(row["query_case_id"], row["selected_action"]) for row in selected] == [("case_a", "B"), ("case_b", "A")]


def test_point_geometry_statistics_describes_prompt_spread_and_match_quality() -> None:
    stats = point_geometry_statistics([
        {"x": "0", "y": "0", "score": "0.2", "s_fg": "0.3", "s_bg": "0.1"},
        {"x": "3", "y": "4", "score": "0.8", "s_fg": "0.9", "s_bg": "0.2"},
    ], image_width=10, image_height=10)

    assert stats["point_score_mean"] == pytest.approx(0.5)
    assert stats["point_fg_bg_gap_mean"] == pytest.approx(0.45)
    assert stats["point_pairwise_distance_mean"] == pytest.approx(0.5)


def test_mask_matcher_consistency_statistics_measures_evidence_alignment() -> None:
    mask = np.asarray([[1, 0], [1, 0]], dtype=bool)
    score_map = np.asarray([[4.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    points = [{"x": "0", "y": "0"}, {"x": "1", "y": "0"}]

    stats = mask_matcher_consistency_statistics(mask, score_map, points)

    assert stats["mask_score_mean"] == pytest.approx(3.0)
    assert stats["outside_score_mean"] == pytest.approx(0.0)
    assert stats["mask_score_contrast"] == pytest.approx(3.0)
    assert stats["top_score_mask_coverage"] == pytest.approx(1.0)
    assert stats["prompt_in_mask_ratio"] == pytest.approx(0.5)
    assert stats["score_mass_in_mask"] > 0.9


def test_extra_trees_reward_predicts_higher_value_for_high_feature_region() -> None:
    rows = [
        {"score_max": 0.0, "k": 20.0, "dice": 0.1},
        {"score_max": 0.2, "k": 20.0, "dice": 0.2},
        {"score_max": 1.8, "k": 20.0, "dice": 0.8},
        {"score_max": 2.0, "k": 20.0, "dice": 0.9},
    ]
    model = train_extra_trees_reward(rows, feature_names=["score_max", "k"], target_key="dice", min_samples_leaf=1)

    assert predict_extra_trees_reward(model, {"score_max": 1.9, "k": 20.0}) > predict_extra_trees_reward(model, {"score_max": 0.1, "k": 20.0})

    mean, std = predict_extra_trees_distribution(model, {"score_max": 1.9, "k": 20.0})
    assert mean == pytest.approx(predict_extra_trees_reward(model, {"score_max": 1.9, "k": 20.0}))
    assert std >= 0.0


def test_select_top_n_candidates_keeps_budget_per_case() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "A", "predicted_reward": 0.2},
        {"query_case_id": "case_a", "action": "B", "predicted_reward": 0.8},
        {"query_case_id": "case_a", "action": "C", "predicted_reward": 0.5},
        {"query_case_id": "case_a", "action": "D", "predicted_reward": 0.1},
    ]

    selected = select_top_n_candidates(rows, budget=3)

    assert [row["action"] for row in selected] == ["B", "C", "A"]
    assert [row["policy_rank"] for row in selected] == [1, 2, 3]


def test_summarize_selected_candidates_reports_evaluated_candidate_budget() -> None:
    summary = summarize_selected_candidates([
        {"query_case_id": "case_a", "dice": "0.6", "iou": "0.5", "object_recall": "0.7", "merge_error": "0.1"},
    ], candidates_per_case=3)

    assert summary["case_count"] == 1
    assert summary["candidates_per_case"] == 3


def test_two_stage_policy_shortlists_with_pre_sam_then_selects_with_post_sam() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "A", "pre": 3.0, "post": 0.0, "dice": 0.5},
        {"query_case_id": "case_a", "action": "B", "pre": 2.0, "post": 3.0, "dice": 0.9},
        {"query_case_id": "case_a", "action": "C", "pre": 1.0, "post": 1.0, "dice": 0.1},
    ]
    pre_model = train_extra_trees_reward(rows, ["pre"], "pre", min_samples_leaf=1)
    post_model = train_extra_trees_reward(rows, ["post"], "dice", min_samples_leaf=1)

    shortlist, selected = apply_two_stage_policy(rows, pre_model, post_model, budget=2)

    assert [row["action"] for row in shortlist] == ["A", "B"]
    assert selected[0]["action"] == "B"
    assert selected[0]["selection_source"] == "two_stage_post_sam"


def test_within_case_policy_targets_remove_case_difficulty_and_mark_top_actions() -> None:
    rows = [
        {"query_case_id": "easy", "action": "A", "dice": 0.8},
        {"query_case_id": "easy", "action": "B", "dice": 0.9},
        {"query_case_id": "hard", "action": "A", "dice": 0.1},
        {"query_case_id": "hard", "action": "B", "dice": 0.2},
    ]

    enriched = add_within_case_policy_targets(rows, top_n=1)
    indexed = {(row["query_case_id"], row["action"]): row for row in enriched}

    assert indexed[("easy", "A")]["case_centered_dice"] == pytest.approx(-0.05)
    assert indexed[("hard", "A")]["case_centered_dice"] == pytest.approx(-0.05)
    assert indexed[("easy", "B")]["case_normalized_dice"] == pytest.approx(1.0)
    assert indexed[("hard", "A")]["case_normalized_dice"] == pytest.approx(0.0)
    assert indexed[("easy", "B")]["case_top_action"] == 1.0
    assert indexed[("hard", "A")]["case_top_action"] == 0.0


def test_anchor_relative_features_learn_gain_and_feature_difference() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "anchor", "dice": 0.4, "sam_score": 0.7},
        {"query_case_id": "case_a", "action": "candidate", "dice": 0.6, "sam_score": 0.9},
    ]

    enriched = add_anchor_relative_features(rows, anchor_action="anchor", feature_names=["sam_score"])
    candidate_row = next(row for row in enriched if row["action"] == "candidate")

    assert candidate_row["dice_gain_over_anchor"] == pytest.approx(0.2)
    assert candidate_row["delta_sam_score"] == pytest.approx(0.2)


def test_anchor_shortlist_always_keeps_anchor_with_best_learned_alternatives() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "anchor", "predicted_reward": 0.0},
        {"query_case_id": "case_a", "action": "A", "predicted_reward": 0.8},
        {"query_case_id": "case_a", "action": "B", "predicted_reward": 0.6},
        {"query_case_id": "case_a", "action": "C", "predicted_reward": 0.4},
    ]

    shortlist = select_top_n_with_anchor(rows, budget=3, anchor_action="anchor")

    assert [row["action"] for row in shortlist] == ["A", "B", "anchor"]


def test_anchor_gate_falls_back_when_predicted_gain_is_too_small() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "anchor", "predicted_reward": 0.0},
        {"query_case_id": "case_a", "action": "A", "predicted_reward": 0.01},
        {"query_case_id": "case_b", "action": "anchor", "predicted_reward": 0.0},
        {"query_case_id": "case_b", "action": "B", "predicted_reward": 0.2},
    ]

    selected = select_anchor_gated_candidates(rows, anchor_action="anchor", threshold=0.05)

    assert [(row["query_case_id"], row["action"], row["selection_source"]) for row in selected] == [
        ("case_a", "anchor", "anchor_fallback"),
        ("case_b", "B", "anchor_gated_policy"),
    ]
    assert selected[0]["policy_best_action"] == "A"
    assert selected[0]["policy_predicted_gain"] == pytest.approx(0.01)
    assert selected[0]["gain_threshold"] == pytest.approx(0.05)
    assert selected[0]["switched"] == 0
    assert selected[1]["switched"] == 1


def test_anchor_gate_can_penalize_high_uncertainty_alternative() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "anchor", "predicted_reward": 0.0, "predicted_reward_std": 0.0},
        {"query_case_id": "case_a", "action": "risky", "predicted_reward": 0.2, "predicted_reward_std": 0.3},
        {"query_case_id": "case_a", "action": "stable", "predicted_reward": 0.1, "predicted_reward_std": 0.01},
    ]

    selected = select_anchor_gated_candidates(
        rows,
        anchor_action="anchor",
        threshold=0.05,
        uncertainty_weight=1.0,
    )

    assert selected[0]["action"] == "stable"
    assert selected[0]["policy_risk_adjusted_gain"] == pytest.approx(0.09)
    assert selected[0]["uncertainty_weight"] == pytest.approx(1.0)


def test_anchored_two_stage_policy_keeps_baseline_and_selects_positive_gain() -> None:
    rows = [
        {"query_case_id": "case_a", "action": "anchor", "delta_pre": 0.0, "delta_post": 0.0, "dice_gain_over_anchor": 0.0},
        {"query_case_id": "case_a", "action": "B", "delta_pre": 3.0, "delta_post": 3.0, "dice_gain_over_anchor": 0.2},
        {"query_case_id": "case_a", "action": "C", "delta_pre": 1.0, "delta_post": 1.0, "dice_gain_over_anchor": -0.1},
    ]
    pre_model = train_extra_trees_reward(rows, ["delta_pre"], "dice_gain_over_anchor", min_samples_leaf=1)
    post_model = train_extra_trees_reward(rows, ["delta_post"], "dice_gain_over_anchor", min_samples_leaf=1)

    shortlist, selected = apply_anchored_two_stage_policy(
        rows, pre_model, post_model, budget=2, anchor_action="anchor", threshold=0.0,
    )

    assert [row["action"] for row in shortlist] == ["B", "anchor"]
    assert selected[0]["action"] == "B"


def test_filter_candidate_rounds_removes_unstable_layout_round() -> None:
    rows = [
        {"round_name": "round1", "method": "K20"},
        {"round_name": "round2", "method": "K40"},
        {"round_name": "round3", "method": "local_maxima"},
    ]

    assert filter_candidate_rounds(rows, {"round1", "round2"}) == rows[:2]


def test_validation_best_action_uses_mean_validation_dice() -> None:
    rows = [
        {"query_case_id": "a", "method": "K20", "dice": "0.8"},
        {"query_case_id": "b", "method": "K20", "dice": "0.2"},
        {"query_case_id": "a", "method": "K40", "dice": "0.6"},
        {"query_case_id": "b", "method": "K40", "dice": "0.6"},
    ]

    assert validation_best_action(rows) == "K40"


def test_within_case_targets_keep_multiple_references_as_separate_episodes() -> None:
    rows = [
        {"episode_id": "ref1_to_q", "query_case_id": "q", "action": "A", "dice": 0.1},
        {"episode_id": "ref1_to_q", "query_case_id": "q", "action": "B", "dice": 0.2},
        {"episode_id": "ref2_to_q", "query_case_id": "q", "action": "A", "dice": 0.9},
        {"episode_id": "ref2_to_q", "query_case_id": "q", "action": "B", "dice": 0.8},
    ]

    enriched = add_within_case_policy_targets(rows, top_n=1)
    winners = {(row["episode_id"], row["action"]) for row in enriched if row["case_top_action"] == 1.0}

    assert winners == {("ref1_to_q", "B"), ("ref2_to_q", "A")}


def test_filter_candidate_grid_keeps_shared_round_k_lambda_action_space() -> None:
    rows = [
        {"round_name": "round2", "k": "5", "lambda_bg": "0.25"},
        {"round_name": "round2", "k": "10", "lambda_bg": "0.75"},
        {"round_name": "round2", "k": "3", "lambda_bg": "0.75"},
        {"round_name": "round1", "k": "5", "lambda_bg": "0.0"},
    ]

    selected = filter_candidate_grid(
        rows,
        allowed_rounds={"round2"},
        allowed_k={5, 10},
        allowed_lambda={0.25, 0.75},
    )

    assert selected == rows[:2]
