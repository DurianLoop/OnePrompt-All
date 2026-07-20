from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

from train_adaptive_prompt_policy import (
    POST_SAM_FEATURES,
    ROOT,
    add_anchor_relative_features,
    build_candidate_features,
    policy_group_id,
    read_csv,
    score_extra_trees_distribution_rows,
    select_anchor_gated_candidates,
    summarize_selected_candidates,
    train_extra_trees_reward,
    write_csv,
)


RAW_CONSISTENCY_FEATURE_NAMES = [
    "mask_score_mean",
    "outside_score_mean",
    "mask_score_contrast",
    "top_score_mask_coverage",
    "score_mass_in_mask",
    "prompt_in_mask_ratio",
    "pred_components_per_prompt",
    "pred_area_per_prompt",
    "mask_artifact_available",
]
CONSISTENCY_FEATURE_NAMES = [f"delta_{name}" for name in RAW_CONSISTENCY_FEATURE_NAMES]


def base_selector_features() -> list[str]:
    return [
        f"delta_{name}"
        for name in POST_SAM_FEATURES
        if name not in RAW_CONSISTENCY_FEATURE_NAMES
    ]


def consistency_selector_features() -> list[str]:
    return base_selector_features() + CONSISTENCY_FEATURE_NAMES


def normalize_shortlist_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        action = row.get("action") or row.get("method")
        if not action:
            raise ValueError("Every shortlist row requires action or method.")
        normalized.append({**row, "action": action, "method": action})
    return normalized


def validate_shortlist(rows: list[dict[str, str]], anchor_action: str, expected_budget: int) -> None:
    counts = Counter(policy_group_id(row) for row in rows)
    if not counts:
        raise ValueError("Shortlist cannot be empty.")
    invalid = {episode: count for episode, count in counts.items() if count != expected_budget}
    if invalid:
        raise ValueError(f"Expected {expected_budget} candidates per episode, found {invalid}.")
    anchors = Counter(policy_group_id(row) for row in rows if row["method"] == anchor_action)
    missing = sorted(set(counts) - set(anchors))
    if missing:
        raise ValueError(f"Missing anchor action in episodes: {missing[:5]}")


def fixed_anchor_rows(rows: list[dict[str, Any]], anchor_action: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "selected_action": anchor_action,
            "selection_source": "fixed_anchor",
            "switched": 0,
        }
        for row in rows
        if row["action"] == anchor_action
    ]


def fit_selector(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    feature_names: list[str],
    anchor_action: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    choices: list[dict[str, Any]] = []
    models: dict[int, dict[str, Any]] = {}
    for min_leaf in (2, 4, 8, 16):
        model = train_extra_trees_reward(
            train_rows,
            feature_names,
            "dice_gain_over_anchor",
            min_samples_leaf=min_leaf,
        )
        models[min_leaf] = model
        scored = score_extra_trees_distribution_rows(model, val_rows)
        for threshold in (-0.02, 0.0, 0.01, 0.02, 0.05, 0.1):
            selected = select_anchor_gated_candidates(
                scored,
                anchor_action=anchor_action,
                threshold=threshold,
                uncertainty_weight=0.0,
            )
            choices.append({
                "min_samples_leaf": min_leaf,
                "gain_threshold": threshold,
                **summarize_selected_candidates(selected, candidates_per_case=3),
            })
    choice = max(
        choices,
        key=lambda row: (
            float(row["mean_dice"]),
            float(row["gain_threshold"]),
            -int(row["min_samples_leaf"]),
        ),
    )
    return models[int(choice["min_samples_leaf"])], choice, choices


def apply_selector(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    anchor_action: str,
    threshold: float,
) -> list[dict[str, Any]]:
    scored = score_extra_trees_distribution_rows(model, rows)
    return select_anchor_gated_candidates(
        scored,
        anchor_action=anchor_action,
        threshold=threshold,
        uncertainty_weight=0.0,
    )


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a top-3 post-SAM selector with a controlled matcher-mask consistency ablation."
    )
    parser.add_argument("--train-shortlist", required=True)
    parser.add_argument("--val-shortlist", required=True)
    parser.add_argument("--test-shortlist", required=True)
    parser.add_argument("--anchor-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--candidate-budget", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_by_split = {
        split: normalize_shortlist_rows(read_csv(resolve(path)))
        for split, path in {
            "train": args.train_shortlist,
            "val": args.val_shortlist,
            "test": args.test_shortlist,
        }.items()
    }
    for rows in raw_by_split.values():
        validate_shortlist(rows, args.anchor_action, args.candidate_budget)

    features_by_split = {
        split: add_anchor_relative_features(
            build_candidate_features(rows, ROOT),
            args.anchor_action,
            POST_SAM_FEATURES,
        )
        for split, rows in raw_by_split.items()
    }
    unavailable = {
        split: sum(float(row["mask_artifact_available"]) < 0.5 for row in rows)
        for split, rows in features_by_split.items()
    }
    if any(unavailable.values()):
        raise ValueError(f"All shortlisted masks must be materialized before training: {unavailable}")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {
        "anchor_action": args.anchor_action,
        "candidate_budget": args.candidate_budget,
        "training_unit": "reference-query episode",
        "test_inference_unit": "query after averaging references",
        "test_used_for_selection": False,
        "feature_sets": {},
    }

    for feature_set, feature_names in {
        "base_post_sam": base_selector_features(),
        "plus_matcher_mask_consistency": consistency_selector_features(),
    }.items():
        model, choice, choices = fit_selector(
            features_by_split["train"],
            features_by_split["val"],
            feature_names,
            args.anchor_action,
        )
        write_csv(out_dir / f"{feature_set}_validation_selection.csv", choices)
        joblib.dump(model, out_dir / f"{feature_set}_selector.joblib")
        model_metadata["feature_sets"][feature_set] = {
            "feature_names": feature_names,
            "min_samples_leaf_selected_on_validation": int(choice["min_samples_leaf"]),
            "gain_threshold_selected_on_validation": float(choice["gain_threshold"]),
            "validation_mean_dice": float(choice["mean_dice"]),
        }
        for split, rows in features_by_split.items():
            selected = apply_selector(
                model,
                rows,
                args.anchor_action,
                float(choice["gain_threshold"]),
            )
            write_csv(out_dir / f"{split}_{feature_set}_predictions.csv", selected)
            summary_rows.append({
                "split": split,
                "method": feature_set,
                **summarize_selected_candidates(selected, candidates_per_case=args.candidate_budget),
            })

    for split, rows in features_by_split.items():
        fixed = fixed_anchor_rows(rows, args.anchor_action)
        write_csv(out_dir / f"{split}_fixed_predictions.csv", fixed)
        summary_rows.append({
            "split": split,
            "method": "fixed_anchor",
            **summarize_selected_candidates(fixed, candidates_per_case=1),
        })

    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "model.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(model_metadata, indent=2))


if __name__ == "__main__":
    main()
