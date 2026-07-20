from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

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


def load_policy_configuration(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    raw = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    return {
        "fallback_action": str(raw["fallback_action"]),
        "candidate_budget": int(raw["two_stage_candidate_budget"]),
        "gain_threshold": float(raw["anchored_gain_threshold_selected_on_validation"]),
        "uncertainty_weight": float(raw.get("anchored_uncertainty_weight_selected_on_validation", 0.0)),
        "pre_min_samples_leaf": int(raw["anchored_pre_min_samples_leaf_selected_on_validation"]),
        "post_min_samples_leaf": int(raw["anchored_post_min_samples_leaf_selected_on_validation"]),
        "pre_features": [str(item) for item in raw["anchor_pre_features"]],
        "post_features": [str(item) for item in raw["anchor_post_features"]],
        "pre_model_path": model_dir / raw["anchored_pre_sam_model_path"],
        "post_model_path": model_dir / raw["anchored_post_sam_model_path"],
        "allowed_rounds": {str(item) for item in raw["allowed_rounds"]},
        "allowed_k": {int(item) for item in raw["allowed_k"]},
        "allowed_lambda": {float(item) for item in raw["allowed_lambda"]},
        "source_model_json": raw,
    }


def apply_saved_policy(
    candidate_rows: list[dict[str, str]],
    model_dir: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = load_policy_configuration(model_dir)
    filtered = filter_candidate_grid(
        candidate_rows,
        config["allowed_rounds"],
        config["allowed_k"],
        config["allowed_lambda"],
    )
    fallback = config["fallback_action"]
    episode_ids = {
        str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])
        for row in filtered
    }
    fixed = select_policy_candidates(filtered, {episode_id: fallback for episode_id in episode_ids}, fallback)
    features = build_candidate_features(filtered, workspace_root)
    anchor_rows = add_anchor_relative_features(features, fallback, config["source_model_json"]["post_sam_features"])
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
    shortlist, learned = apply_anchored_two_stage_policy(
        anchor_rows,
        pre_sam_model=pre_model,
        post_sam_model=post_model,
        budget=config["candidate_budget"],
        anchor_action=fallback,
        threshold=config["gain_threshold"],
        uncertainty_weight=config["uncertainty_weight"],
    )
    summary = {
        "model_dir": str(model_dir),
        "candidate_episode_count": len(episode_ids),
        "fallback_action": fallback,
        "gain_threshold": config["gain_threshold"],
        "uncertainty_weight": config["uncertainty_weight"],
        "candidate_budget": config["candidate_budget"],
        "fixed": summarize_selected_candidates(fixed),
        "learned": summarize_selected_candidates(learned, candidates_per_case=config["candidate_budget"]),
        "training_performed": False,
    }
    return fixed, shortlist, learned, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen saved anchored prompt policy without fitting on evaluation candidates.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    fixed, shortlist, learned, summary = apply_saved_policy(
        read_csv(resolve(args.candidates)),
        resolve(args.model_dir),
        ROOT,
    )
    out_dir = resolve(args.out_dir)
    write_csv(out_dir / "fixed_predictions.csv", fixed)
    write_csv(out_dir / "anchored_shortlist.csv", shortlist)
    write_csv(out_dir / "anchored_predictions.csv", learned)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
