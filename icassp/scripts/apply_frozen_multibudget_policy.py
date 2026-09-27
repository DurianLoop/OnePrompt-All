"""Apply one frozen four-action MoNuSeg policy to a never-used target domain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "surf/benchmarks"))
import joblib  # noqa: E402
from train_adaptive_prompt_policy import (  # noqa: E402
    add_anchor_relative_features, apply_anchored_two_stage_policy, read_csv, select_policy_candidates,
    summarize_selected_candidates, write_csv,
)
from train_negative_budget_policy import NEG_ANCHOR_POST_FEATURES, NEG_ANCHOR_PRE_FEATURES, NEG_POST_FEATURES, base_features, join_negative_candidates  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    args = parser.parse_args()
    if args.out_dir.exists(): raise FileExistsError(f"Preserving output: {args.out_dir}")
    raw = json.loads((args.model_dir / "model.json").read_text(encoding="utf-8"))
    if raw.get("method") != "anchored_two_stage_four_action_negative_budget_policy" or int(raw.get("candidate_budget", 0)) != 4:
        raise ValueError("Expected a frozen four-action policy")
    anchor, threshold = raw["anchor_selected_on_validation"], float(raw["selection"]["gain_threshold"])
    rows = join_negative_candidates(read_csv(args.negative), base_features(args.candidates, args.fixed_action))
    action_space = sorted({str(r["action"]) for r in rows}, key=lambda x: int(x[3:]))
    if action_space != ["neg0", "neg5", "neg10", "neg20"]: raise AssertionError(f"Target actions: {action_space}")
    relative = add_anchor_relative_features(rows, anchor, NEG_POST_FEATURES)
    pre = {"feature_names": NEG_ANCHOR_PRE_FEATURES, "estimator": joblib.load(args.model_dir / "negative_pre_sam.joblib")}
    post = {"feature_names": NEG_ANCHOR_POST_FEATURES, "estimator": joblib.load(args.model_dir / "negative_post_sam.joblib")}
    shortlist, learned = apply_anchored_two_stage_policy(relative, pre, post, 4, anchor, threshold)
    fixed = select_policy_candidates(rows, {str(r["episode_id"]): anchor for r in rows}, anchor)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "fixed_predictions.csv", fixed)
    write_csv(args.out_dir / "shortlist.csv", shortlist)
    write_csv(args.out_dir / "learned_predictions.csv", learned)
    result = {"source_model_dir": str(args.model_dir), "target_negative": str(args.negative), "anchor_action": anchor,
        "candidate_budget": 4, "gain_threshold": threshold, "fixed": summarize_selected_candidates(fixed),
        "learned": summarize_selected_candidates(learned, candidates_per_case=4),
        "protocol_note": "Target labels are used only to calculate reported metrics after applying the frozen source-domain policy."}
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
