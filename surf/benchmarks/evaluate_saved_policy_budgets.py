from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from apply_saved_prompt_policy import load_policy_configuration
from train_adaptive_prompt_policy import (
    add_anchor_relative_features,
    apply_anchored_two_stage_policy,
    build_candidate_features,
    filter_candidate_grid,
    read_csv,
    select_policy_candidates,
    summarize_selected_candidates,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_validation_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one validation budget is required.")
    return max(rows, key=lambda row: (float(row["mean_dice"]), -int(row["candidate_budget"])))


def prepare_rows(candidate_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = filter_candidate_grid(
        read_csv(candidate_path),
        config["allowed_rounds"],
        config["allowed_k"],
        config["allowed_lambda"],
    )
    fallback = config["fallback_action"]
    episode_ids = {str(row.get("episode_id") or row.get("case_name") or row["query_case_id"]) for row in candidates}
    fixed = select_policy_candidates(candidates, {episode_id: fallback for episode_id in episode_ids}, fallback)
    features = build_candidate_features(candidates, ROOT)
    relative = add_anchor_relative_features(features, fallback, config["source_model_json"]["post_sam_features"])
    return fixed, relative


def main() -> None:
    parser = argparse.ArgumentParser(description="Select shortlist budget on validation and evaluate the frozen policy once on test.")
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--test-candidates", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--budgets", default="1,3,5,7,10,15")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model_dir = resolve(args.model_dir)
    config = load_policy_configuration(model_dir)
    pre_model = {
        "feature_names": config["pre_features"],
        "min_samples_leaf": config["pre_min_samples_leaf"],
        "estimator": joblib.load(config["pre_model_path"]),
    }
    post_model = {
        "feature_names": config["post_features"],
        "min_samples_leaf": config["post_min_samples_leaf"],
        "estimator": joblib.load(config["post_model_path"]),
    }
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    val_fixed, val_rows = prepare_rows(resolve(args.val_candidates), config)
    validation: list[dict[str, Any]] = []
    validation_predictions: dict[int, list[dict[str, Any]]] = {}
    for budget in budgets:
        _, selected = apply_anchored_two_stage_policy(
            val_rows, pre_model, post_model, budget,
            config["fallback_action"], config["gain_threshold"], config["uncertainty_weight"],
        )
        validation_predictions[budget] = selected
        validation.append({"candidate_budget": budget, **summarize_selected_candidates(selected, budget)})
    choice = choose_validation_budget(validation)
    selected_budget = int(choice["candidate_budget"])
    test_fixed, test_rows = prepare_rows(resolve(args.test_candidates), config)
    test_shortlist, test_selected = apply_anchored_two_stage_policy(
        test_rows, pre_model, post_model, selected_budget,
        config["fallback_action"], config["gain_threshold"], config["uncertainty_weight"],
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "validation_budget_selection.csv", validation)
    write_csv(out_dir / "validation_fixed_predictions.csv", val_fixed)
    write_csv(out_dir / "validation_selected_predictions.csv", validation_predictions[selected_budget])
    write_csv(out_dir / "test_fixed_predictions.csv", test_fixed)
    write_csv(out_dir / "test_shortlist.csv", test_shortlist)
    write_csv(out_dir / "test_selected_predictions.csv", test_selected)
    summary = {
        "selection_source": "validation mean Dice; ties prefer smaller budget",
        "selected_budget": selected_budget,
        "validation_choice": choice,
        "test_fixed": summarize_selected_candidates(test_fixed),
        "test_selected": summarize_selected_candidates(test_selected, selected_budget),
        "training_performed": False,
        "model_dir": str(model_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
