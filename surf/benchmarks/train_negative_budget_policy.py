from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from train_adaptive_prompt_policy import (
    RIDGE_FEATURES,
    add_anchor_relative_features,
    apply_anchored_two_stage_policy,
    build_candidate_features,
    filter_candidate_grid,
    read_csv,
    score_anchored_two_stage_candidates,
    select_anchor_gated_candidates,
    select_policy_candidates,
    summarize_selected_candidates,
    train_extra_trees_reward,
    validation_best_action,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
NEG_PRE_FEATURES = RIDGE_FEATURES + ["negative_count"]
NEG_POST_FEATURES = NEG_PRE_FEATURES + ["sam_score", "pred_area_pct"]
NEG_ANCHOR_PRE_FEATURES = [f"delta_{name}" for name in NEG_PRE_FEATURES]
NEG_ANCHOR_POST_FEATURES = [f"delta_{name}" for name in NEG_POST_FEATURES]


def action_space_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(row["action"]) for row in rows},
        key=lambda action: int(action.removeprefix("neg")),
    )


def feature_importance_rows(
    estimator: Any,
    feature_names: list[str],
    stage: str,
) -> list[dict[str, Any]]:
    rows = [
        {"stage": stage, "feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, estimator.feature_importances_)
    ]
    return sorted(rows, key=lambda row: (-float(row["importance"]), str(row["feature"])))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def join_negative_candidates(
    negative_rows: list[dict[str, str]],
    base_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_by_episode = {str(row["episode_id"]): row for row in base_features}
    joined: list[dict[str, Any]] = []
    for row in negative_rows:
        episode_id = row["episode_id"]
        if episode_id not in base_by_episode:
            raise KeyError(f"Missing base matcher features for {episode_id!r}.")
        requested_negative_count = int(row.get("requested_negative_count", row["negative_count"]))
        negative_count = int(row["negative_count"])
        joined.append({
            **base_by_episode[episode_id],
            "episode_id": episode_id,
            "query_case_id": row["query_case_id"],
            "action": f"neg{requested_negative_count}",
            "method": f"neg{requested_negative_count}",
            "requested_negative_count": float(requested_negative_count),
            "negative_count": float(negative_count),
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "object_recall": float(row["object_recall"]),
            "merge_error": float(row["merge_error"]),
            "sam_score": float(row["sam_score"]),
            "pred_area_pct": float(row["pred_area_pct"]),
        })
    return joined


def base_features(candidate_path: Path, fixed_action: str) -> list[dict[str, Any]]:
    candidates = [row for row in read_csv(candidate_path) if row["method"] == fixed_action]
    return build_candidate_features(candidates, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn only the negative-point budget while freezing matcher, K, lambda, positives, and SAM.")
    parser.add_argument("--train-negative", required=True)
    parser.add_argument("--val-negative", required=True)
    parser.add_argument("--test-negative", required=True)
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--test-candidates", required=True)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        negative_path = resolve(getattr(args, f"{split}_negative"))
        candidate_path = resolve(getattr(args, f"{split}_candidates"))
        splits[split] = join_negative_candidates(read_csv(negative_path), base_features(candidate_path, args.fixed_action))

    anchor = validation_best_action(splits["val"])
    anchor_train = add_anchor_relative_features(splits["train"], anchor, NEG_POST_FEATURES)
    anchor_val = add_anchor_relative_features(splits["val"], anchor, NEG_POST_FEATURES)
    leaf_options = (1, 2, 4, 8)
    thresholds = (-0.02, 0.0, 0.005, 0.01, 0.02, 0.05)
    pre_models = {
        leaf: train_extra_trees_reward(anchor_train, NEG_ANCHOR_PRE_FEATURES, "dice_gain_over_anchor", leaf)
        for leaf in leaf_options
    }
    post_models = {
        leaf: train_extra_trees_reward(anchor_train, NEG_ANCHOR_POST_FEATURES, "dice_gain_over_anchor", leaf)
        for leaf in leaf_options
    }
    validation_rows: list[dict[str, Any]] = []
    for pre_leaf, pre_model in pre_models.items():
        for post_leaf, post_model in post_models.items():
            _, scored = score_anchored_two_stage_candidates(
                anchor_val,
                pre_sam_model=pre_model,
                post_sam_model=post_model,
                budget=2,
                anchor_action=anchor,
            )
            for threshold in thresholds:
                selected = select_anchor_gated_candidates(scored, anchor, threshold)
                validation_rows.append({
                    "pre_min_samples_leaf": pre_leaf,
                    "post_min_samples_leaf": post_leaf,
                    "gain_threshold": threshold,
                    **summarize_selected_candidates(selected, candidates_per_case=2),
                })
    choice = max(validation_rows, key=lambda row: (
        float(row["mean_dice"]),
        float(row["gain_threshold"]),
        -int(row["pre_min_samples_leaf"]),
        -int(row["post_min_samples_leaf"]),
    ))
    pre_model = pre_models[int(choice["pre_min_samples_leaf"])]
    post_model = post_models[int(choice["post_min_samples_leaf"])]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "validation_selection.csv", validation_rows)
    joblib.dump(pre_model["estimator"], out_dir / "negative_pre_sam.joblib")
    joblib.dump(post_model["estimator"], out_dir / "negative_post_sam.joblib")
    feature_importances = (
        feature_importance_rows(pre_model["estimator"], NEG_ANCHOR_PRE_FEATURES, "pre_sam")
        + feature_importance_rows(post_model["estimator"], NEG_ANCHOR_POST_FEATURES, "post_sam")
    )
    write_csv(out_dir / "feature_importance.csv", feature_importances)
    summaries: list[dict[str, Any]] = []
    for split, rows in splits.items():
        relative = add_anchor_relative_features(rows, anchor, NEG_POST_FEATURES)
        shortlist, selected = apply_anchored_two_stage_policy(
            relative,
            pre_sam_model=pre_model,
            post_sam_model=post_model,
            budget=2,
            anchor_action=anchor,
            threshold=float(choice["gain_threshold"]),
        )
        episode_ids = {str(row["episode_id"]) for row in rows}
        fixed = select_policy_candidates(rows, {episode_id: anchor for episode_id in episode_ids}, anchor)
        write_csv(out_dir / f"{split}_fixed_predictions.csv", fixed)
        write_csv(out_dir / f"{split}_shortlist.csv", shortlist)
        write_csv(out_dir / f"{split}_learned_predictions.csv", selected)
        summaries.extend([
            {"split": split, "method": "validation_best_fixed_negative_budget", "anchor_action": anchor, **summarize_selected_candidates(fixed)},
            {"split": split, "method": "learned_negative_budget_top2", "anchor_action": anchor, **summarize_selected_candidates(selected, candidates_per_case=2)},
        ])
    write_csv(out_dir / "summary.csv", summaries)
    (out_dir / "model.json").write_text(json.dumps({
        "method": "anchored_two_stage_negative_budget_policy",
        "frozen_variables": {"matcher": "DINOv2-L fg-bg", "positive_count": 40, "lambda_bg": 0.75, "layout": "topk", "segmenter": "SAM ViT-B"},
        "action_space": action_space_from_rows(splits["train"]),
        "anchor_selected_on_validation": anchor,
        "candidate_budget": 2,
        "pre_features": NEG_PRE_FEATURES,
        "post_features": NEG_POST_FEATURES,
        "feature_importance": feature_importances,
        "selection": choice,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"anchor": anchor, "selection": choice, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
