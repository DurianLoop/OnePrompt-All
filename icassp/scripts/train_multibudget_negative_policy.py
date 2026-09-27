"""Learn a four-action {0,5,10,20} background-prompt budget policy.

Unlike the frozen two-action policy, this experiment exposes every already
executed negative-budget action to the selector while keeping matcher, points,
SAM, and nested train/validation/test splits fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "surf/benchmarks"))
from train_adaptive_prompt_policy import (  # noqa: E402
    add_anchor_relative_features, apply_anchored_two_stage_policy, score_anchored_two_stage_candidates,
    select_anchor_gated_candidates, select_policy_candidates, summarize_selected_candidates, validation_best_action,
    write_csv,
)
from train_negative_budget_policy import (  # noqa: E402
    NEG_ANCHOR_POST_FEATURES, NEG_ANCHOR_PRE_FEATURES, NEG_POST_FEATURES, action_space_from_rows,
    base_features, feature_importance_rows, join_negative_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested four-action negative-budget policy.")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-negative", required=True, type=Path)
        parser.add_argument(f"--{split}-candidates", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--seed", type=int, default=20260715, help="ExtraTrees random seed; split and all selection rules remain fixed.")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Preserving output: {args.out_dir}")
    splits = {}
    for split in ("train", "val", "test"):
        from train_adaptive_prompt_policy import read_csv  # local import keeps source dependency explicit
        splits[split] = join_negative_candidates(read_csv(getattr(args, f"{split}_negative")),
            base_features(getattr(args, f"{split}_candidates"), args.fixed_action))
    actions = action_space_from_rows(splits["train"])
    if actions != ["neg0", "neg5", "neg10", "neg20"]:
        raise AssertionError(f"Expected four fixed actions, found {actions}")
    anchor = validation_best_action(splits["val"])
    budget = len(actions)
    train_relative = add_anchor_relative_features(splits["train"], anchor, NEG_POST_FEATURES)
    val_relative = add_anchor_relative_features(splits["val"], anchor, NEG_POST_FEATURES)
    leaves, thresholds = (1, 2, 4, 8), (-0.02, 0.0, 0.005, 0.01, 0.02, 0.05)
    def fit(rows, features, leaf):
        from sklearn.ensemble import ExtraTreesRegressor
        x=np.asarray([[float(row[name]) for name in features] for row in rows],dtype=np.float64)
        y=np.asarray([float(row["dice_gain_over_anchor"]) for row in rows],dtype=np.float64)
        estimator=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=leaf,random_state=args.seed,n_jobs=1)
        estimator.fit(x,y)
        return {"feature_names":features,"min_samples_leaf":leaf,"estimator":estimator}
    pre = {leaf: fit(train_relative, NEG_ANCHOR_PRE_FEATURES, leaf) for leaf in leaves}
    post = {leaf: fit(train_relative, NEG_ANCHOR_POST_FEATURES, leaf) for leaf in leaves}
    grid = []
    for pre_leaf, pre_model in pre.items():
        for post_leaf, post_model in post.items():
            _, scored = score_anchored_two_stage_candidates(val_relative, pre_model, post_model, budget, anchor)
            for threshold in thresholds:
                selected = select_anchor_gated_candidates(scored, anchor, threshold)
                grid.append({"pre_min_samples_leaf": pre_leaf, "post_min_samples_leaf": post_leaf, "gain_threshold": threshold,
                             **summarize_selected_candidates(selected, candidates_per_case=budget)})
    choice = max(grid, key=lambda x: (float(x["mean_dice"]), float(x["gain_threshold"]), -int(x["pre_min_samples_leaf"]), -int(x["post_min_samples_leaf"])))
    pre_model, post_model = pre[int(choice["pre_min_samples_leaf"])], post[int(choice["post_min_samples_leaf"])]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "validation_selection.csv", grid)
    import joblib
    joblib.dump(pre_model["estimator"], args.out_dir / "negative_pre_sam.joblib")
    joblib.dump(post_model["estimator"], args.out_dir / "negative_post_sam.joblib")
    imports = feature_importance_rows(pre_model["estimator"], NEG_ANCHOR_PRE_FEATURES, "pre_sam") + feature_importance_rows(post_model["estimator"], NEG_ANCHOR_POST_FEATURES, "post_sam")
    write_csv(args.out_dir / "feature_importance.csv", imports)
    summaries = []
    for split, rows in splits.items():
        relative = add_anchor_relative_features(rows, anchor, NEG_POST_FEATURES)
        shortlist, selected = apply_anchored_two_stage_policy(relative, pre_model, post_model, budget, anchor, float(choice["gain_threshold"]))
        fixed = select_policy_candidates(rows, {str(row["episode_id"]): anchor for row in rows}, anchor)
        write_csv(args.out_dir / f"{split}_fixed_predictions.csv", fixed)
        write_csv(args.out_dir / f"{split}_shortlist.csv", shortlist)
        write_csv(args.out_dir / f"{split}_learned_predictions.csv", selected)
        summaries += [{"split": split, "method": "validation_best_fixed_negative_budget", "anchor_action": anchor, **summarize_selected_candidates(fixed)},
                      {"split": split, "method": "learned_negative_budget_top4", "anchor_action": anchor, **summarize_selected_candidates(selected, candidates_per_case=budget)}]
    write_csv(args.out_dir / "summary.csv", summaries)
    (args.out_dir / "model.json").write_text(json.dumps({"method": "anchored_two_stage_four_action_negative_budget_policy",
        "frozen_variables": {"matcher": "DINOv2-L fg-bg", "positive_count": 40, "lambda_bg": 0.75, "layout": "topk", "segmenter": "SAM ViT-B"},
        "action_space": actions, "anchor_selected_on_validation": anchor, "candidate_budget": budget,
        "extra_trees_seed": args.seed, "feature_importance": imports, "selection": choice}, indent=2), encoding="utf-8")
    print(json.dumps({"anchor": anchor, "selection": choice, "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
