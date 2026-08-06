from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from train_adaptive_prompt_policy import (
    add_anchor_relative_features,
    apply_anchored_two_stage_policy,
    read_csv,
    select_policy_candidates,
    summarize_selected_candidates,
    write_csv,
)
from train_negative_budget_policy import (
    NEG_ANCHOR_POST_FEATURES,
    NEG_ANCHOR_PRE_FEATURES,
    NEG_POST_FEATURES,
    base_features,
    join_negative_candidates,
)


ROOT = Path(__file__).resolve().parents[2]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def parse_negative_policy_configuration(raw: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    if raw.get("method") != "anchored_two_stage_negative_budget_policy":
        raise ValueError(f"Unsupported negative policy method: {raw.get('method')!r}")
    return {
        "anchor_action": str(raw["anchor_selected_on_validation"]),
        "candidate_budget": int(raw["candidate_budget"]),
        "gain_threshold": float(raw["selection"]["gain_threshold"]),
        "pre_model_path": model_dir / "negative_pre_sam.joblib",
        "post_model_path": model_dir / "negative_post_sam.joblib",
        "source_model_json": raw,
    }


def load_negative_policy_configuration(model_dir: Path) -> dict[str, Any]:
    raw = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    return parse_negative_policy_configuration(raw, model_dir)


def apply_saved_negative_policy(
    negative_path: Path,
    candidate_path: Path,
    model_dir: Path,
    fixed_action: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = load_negative_policy_configuration(model_dir)
    rows = join_negative_candidates(
        read_csv(negative_path),
        base_features(candidate_path, fixed_action),
    )
    anchor = config["anchor_action"]
    relative = add_anchor_relative_features(rows, anchor, NEG_POST_FEATURES)
    pre_model = {
        "feature_names": NEG_ANCHOR_PRE_FEATURES,
        "estimator": joblib.load(config["pre_model_path"]),
    }
    post_model = {
        "feature_names": NEG_ANCHOR_POST_FEATURES,
        "estimator": joblib.load(config["post_model_path"]),
    }
    shortlist, learned = apply_anchored_two_stage_policy(
        relative,
        pre_sam_model=pre_model,
        post_sam_model=post_model,
        budget=config["candidate_budget"],
        anchor_action=anchor,
        threshold=config["gain_threshold"],
    )
    episode_ids = {str(row["episode_id"]) for row in rows}
    fixed = select_policy_candidates(rows, {episode_id: anchor for episode_id in episode_ids}, anchor)
    summary = {
        "source_model_dir": str(model_dir),
        "anchor_action": anchor,
        "candidate_budget": config["candidate_budget"],
        "gain_threshold": config["gain_threshold"],
        "fixed": summarize_selected_candidates(fixed),
        "learned": summarize_selected_candidates(learned, candidates_per_case=config["candidate_budget"]),
    }
    return fixed, shortlist, learned, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen train-only negative-budget gate to a new dataset.")
    parser.add_argument("--negative", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed, shortlist, learned, summary = apply_saved_negative_policy(
        resolve(args.negative),
        resolve(args.candidates),
        resolve(args.model_dir),
        args.fixed_action,
    )
    write_csv(out_dir / "fixed_predictions.csv", fixed)
    write_csv(out_dir / "shortlist.csv", shortlist)
    write_csv(out_dir / "learned_predictions.csv", learned)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
