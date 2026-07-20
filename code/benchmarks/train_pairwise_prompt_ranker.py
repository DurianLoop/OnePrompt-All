from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_adaptive_prompt_policy import (
    POST_SAM_FEATURES,
    RIDGE_FEATURES,
    build_candidate_features,
    filter_candidate_grid,
    policy_group_id,
    read_csv,
    select_policy_candidates,
    summarize_selected_candidates,
    validation_best_action,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]


def build_pairwise_examples(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_key: str,
    margin: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[policy_group_id(row)].append(row)
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for candidates in grouped.values():
        for left_index in range(len(candidates)):
            for right_index in range(left_index + 1, len(candidates)):
                left = candidates[left_index]
                right = candidates[right_index]
                target_delta = float(left[target_key]) - float(right[target_key])
                if abs(target_delta) <= margin:
                    continue
                feature_delta = np.asarray(
                    [float(left[name]) - float(right[name]) for name in feature_names],
                    dtype=np.float64,
                )
                label = int(target_delta > 0.0)
                differences.extend([feature_delta, -feature_delta])
                labels.extend([label, 1 - label])
    if not differences:
        raise ValueError("No non-tied within-episode candidate pairs were available.")
    return np.vstack(differences), np.asarray(labels, dtype=np.int64)


def train_pairwise_ranker(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_key: str,
    c: float,
) -> dict[str, Any]:
    x, y = build_pairwise_examples(rows, feature_names, target_key)
    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c, max_iter=2000, random_state=20260715)),
    ])
    estimator.fit(x, y)
    return {
        "feature_names": list(feature_names),
        "target_key": target_key,
        "c": c,
        "pair_count": len(y) // 2,
        "estimator": estimator,
    }


def pairwise_probability(model: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> float:
    difference = np.asarray([[
        float(left[name]) - float(right[name])
        for name in model["feature_names"]
    ]], dtype=np.float64)
    return float(model["estimator"].predict_proba(difference)[0, 1])


def score_candidates_pairwise(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[policy_group_id(row)].append(row)
    scored: list[dict[str, Any]] = []
    for candidates in grouped.values():
        for candidate in candidates:
            opponents = [row for row in candidates if row is not candidate]
            score = (
                float(np.mean([pairwise_probability(model, candidate, opponent) for opponent in opponents]))
                if opponents else 0.5
            )
            scored.append({**candidate, "predicted_pairwise_score": score})
    return scored


def pairwise_shortlist_with_anchor(
    rows: list[dict[str, Any]],
    budget: int,
    anchor_action: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[policy_group_id(row)].append(row)
    shortlist: list[dict[str, Any]] = []
    for episode_id, candidates in sorted(grouped.items()):
        anchor = next((row for row in candidates if str(row["action"]) == anchor_action), None)
        if anchor is None:
            raise ValueError(f"Missing anchor {anchor_action!r} for {episode_id}.")
        alternatives = sorted(
            (row for row in candidates if row is not anchor),
            key=lambda row: (float(row["predicted_pairwise_score"]), str(row["action"])),
            reverse=True,
        )
        chosen = alternatives[: max(0, budget - 1)] + [anchor]
        shortlist.extend({**row, "policy_rank": rank} for rank, row in enumerate(chosen, start=1))
    return shortlist


def select_pairwise_anchor_gated(
    rows: list[dict[str, Any]],
    post_model: dict[str, Any],
    anchor_action: str,
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for episode_id, candidates in sorted(grouped.items()):
        anchor = next((row for row in candidates if str(row["action"]) == anchor_action), None)
        if anchor is None:
            raise ValueError(f"Missing anchor {anchor_action!r} for {episode_id}.")
        alternatives = [row for row in candidates if row is not anchor]
        best = max(
            alternatives,
            key=lambda row: (float(row["predicted_pairwise_score"]), str(row["action"])),
        ) if alternatives else anchor
        probability = pairwise_probability(post_model, best, anchor) if best is not anchor else 0.5
        if best is not anchor and probability > threshold:
            selected.append({
                **best,
                "selected_action": best["action"],
                "selection_source": "pairwise_anchor_switch",
                "anchor_beat_probability": probability,
                "switched": 1,
            })
        else:
            selected.append({
                **anchor,
                "selected_action": anchor["action"],
                "selection_source": "pairwise_anchor_fallback",
                "anchor_beat_probability": probability,
                "policy_best_action": best["action"],
                "switched": 0,
            })
    return selected


def apply_pairwise_policy(
    rows: list[dict[str, Any]],
    pre_model: dict[str, Any],
    post_model: dict[str, Any],
    anchor_action: str,
    budget: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre_scored = score_candidates_pairwise(rows, pre_model)
    shortlist = pairwise_shortlist_with_anchor(pre_scored, budget, anchor_action)
    post_scored = score_candidates_pairwise(shortlist, post_model)
    return shortlist, select_pairwise_anchor_gated(post_scored, post_model, anchor_action, threshold)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an episode-wise pairwise logistic prompt action ranker.")
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--test-candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allowed-rounds", default="round2")
    parser.add_argument("--allowed-k", default="5,10,20,40,75")
    parser.add_argument("--allowed-lambda", default="0.25,0.75,1.5")
    parser.add_argument("--candidate-budget", type=int, default=3)
    args = parser.parse_args()

    allowed_rounds = {item.strip() for item in args.allowed_rounds.split(",") if item.strip()}
    allowed_k = {int(item) for item in args.allowed_k.split(",") if item.strip()}
    allowed_lambda = {float(item) for item in args.allowed_lambda.split(",") if item.strip()}
    raw_splits = {
        "train": read_csv(resolve(args.train_candidates)),
        "val": read_csv(resolve(args.val_candidates)),
        "test": read_csv(resolve(args.test_candidates)),
    }
    filtered = {
        split: filter_candidate_grid(rows, allowed_rounds, allowed_k, allowed_lambda)
        for split, rows in raw_splits.items()
    }
    feature_splits = {split: build_candidate_features(rows, ROOT) for split, rows in filtered.items()}
    anchor = validation_best_action(filtered["val"])
    c_values = (0.01, 0.1, 1.0, 10.0)
    thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
    pre_models = {c: train_pairwise_ranker(feature_splits["train"], RIDGE_FEATURES, "dice", c) for c in c_values}
    post_models = {c: train_pairwise_ranker(feature_splits["train"], POST_SAM_FEATURES, "dice", c) for c in c_values}
    validation_rows: list[dict[str, Any]] = []
    for pre_c, pre_model in pre_models.items():
        for post_c, post_model in post_models.items():
            for threshold in thresholds:
                _, selected = apply_pairwise_policy(
                    feature_splits["val"], pre_model, post_model, anchor, args.candidate_budget, threshold,
                )
                validation_rows.append({
                    "pre_c": pre_c,
                    "post_c": post_c,
                    "threshold": threshold,
                    **summarize_selected_candidates(selected, candidates_per_case=args.candidate_budget),
                })
    choice = max(
        validation_rows,
        key=lambda row: (
            float(row["mean_dice"]),
            float(row["threshold"]),
            -float(row["pre_c"]),
            -float(row["post_c"]),
        ),
    )
    pre_model = pre_models[float(choice["pre_c"])]
    post_model = post_models[float(choice["post_c"])]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "validation_selection.csv", validation_rows)
    joblib.dump(pre_model, out_dir / "pairwise_pre_sam.joblib")
    joblib.dump(post_model, out_dir / "pairwise_post_sam.joblib")
    summaries: list[dict[str, Any]] = []
    for split, features in feature_splits.items():
        shortlist, learned = apply_pairwise_policy(
            features,
            pre_model,
            post_model,
            anchor,
            args.candidate_budget,
            float(choice["threshold"]),
        )
        episode_ids = {policy_group_id(row) for row in filtered[split]}
        fixed = select_policy_candidates(filtered[split], {episode_id: anchor for episode_id in episode_ids}, anchor)
        write_csv(out_dir / f"{split}_fixed_predictions.csv", fixed)
        write_csv(out_dir / f"{split}_pairwise_shortlist.csv", shortlist)
        write_csv(out_dir / f"{split}_pairwise_predictions.csv", learned)
        summaries.extend([
            {"split": split, "method": "fixed_majority_action", "fallback_action": anchor, **summarize_selected_candidates(fixed)},
            {"split": split, "method": "learned_pairwise_anchored_top3", "fallback_action": anchor, **summarize_selected_candidates(learned, candidates_per_case=args.candidate_budget)},
        ])
    write_csv(out_dir / "summary.csv", summaries)
    (out_dir / "model.json").write_text(json.dumps({
        "method": "episode_pairwise_logistic_ranker",
        "loss": "balanced within-episode pairwise logistic cross-entropy",
        "anchor_action": anchor,
        "candidate_budget": args.candidate_budget,
        "pre_c_selected_on_validation": choice["pre_c"],
        "post_c_selected_on_validation": choice["post_c"],
        "anchor_probability_threshold_selected_on_validation": choice["threshold"],
        "allowed_rounds": sorted(allowed_rounds),
        "allowed_k": sorted(allowed_k),
        "allowed_lambda": sorted(allowed_lambda),
        "train_candidates": str(resolve(args.train_candidates)),
        "val_candidates": str(resolve(args.val_candidates)),
        "test_candidates": str(resolve(args.test_candidates)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"validation_choice": choice, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
