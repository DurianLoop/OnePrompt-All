from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[2]
RIDGE_FEATURES = [
    "score_mean", "score_std", "score_max", "top_quarter_mean", "normalized_entropy",
    "point_score_mean", "point_score_std", "point_fg_bg_gap_mean", "point_fg_bg_gap_std", "point_pairwise_distance_mean",
    "k", "lambda_bg", "layout_topk", "layout_diverse_topk", "layout_local_maxima",
    "layout_boundary_mix", "layout_cluster_medoid", "layout_heatmap_cc",
]
POST_SAM_FEATURES = RIDGE_FEATURES + [
    "sam_score", "pred_area_pct", "pred_components", "prompt_count",
    "mask_score_mean", "outside_score_mean", "mask_score_contrast",
    "top_score_mask_coverage", "score_mass_in_mask", "prompt_in_mask_ratio",
    "pred_components_per_prompt", "pred_area_per_prompt", "mask_artifact_available",
]
ANCHOR_PRE_FEATURES = [f"delta_{name}" for name in RIDGE_FEATURES]
ANCHOR_POST_FEATURES = [f"delta_{name}" for name in POST_SAM_FEATURES]


def candidate_feature_keys() -> list[str]:
    return ["k", "lambda_bg", "layout", "clustering"]


def policy_group_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("case_name") or row["query_case_id"])


def filter_candidate_rounds(
    rows: list[dict[str, Any]],
    allowed_rounds: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("round_name", "")) in allowed_rounds]


def filter_candidate_grid(
    rows: list[dict[str, Any]],
    allowed_rounds: set[str],
    allowed_k: set[int],
    allowed_lambda: set[float],
) -> list[dict[str, Any]]:
    normalized_lambda = {round(value, 6) for value in allowed_lambda}
    return [
        row for row in rows
        if (not allowed_rounds or str(row.get("round_name", "")) in allowed_rounds)
        and (not allowed_k or int(float(row["k"])) in allowed_k)
        and (not normalized_lambda or round(float(row["lambda_bg"]), 6) in normalized_lambda)
    ]


def validation_best_action(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["dice"]))
    if not grouped:
        raise ValueError("At least one validation candidate is required.")
    return max(grouped, key=lambda action: (sum(grouped[action]) / len(grouped[action]), action))


def score_map_statistics(score_map: np.ndarray) -> dict[str, float]:
    values = np.asarray(score_map, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("score_map must contain at least one value.")
    top_count = max(1, int(np.ceil(values.size * 0.25)))
    top_values = np.partition(values, values.size - top_count)[-top_count:]
    shifted = values - float(values.max())
    weights = np.exp(shifted)
    probabilities = weights / float(weights.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    normalizer = float(np.log(values.size)) if values.size > 1 else 1.0
    return {
        "score_mean": float(values.mean()),
        "score_std": float(values.std()),
        "score_max": float(values.max()),
        "top_quarter_mean": float(top_values.mean()),
        "normalized_entropy": entropy / normalizer,
    }


def point_geometry_statistics(
    points: list[dict[str, str]],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    if not points:
        raise ValueError("At least one prompt point is required.")
    scores = np.asarray([float(row["score"]) for row in points], dtype=np.float64)
    fg_bg_gaps = np.asarray([float(row["s_fg"]) - float(row["s_bg"]) for row in points], dtype=np.float64)
    coordinates = np.asarray([[float(row["x"]), float(row["y"])] for row in points], dtype=np.float64)
    if len(coordinates) == 1:
        pairwise_mean = 0.0
    else:
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt(np.square(deltas).sum(axis=2))
        pairwise_mean = float(distances[np.triu_indices(len(coordinates), k=1)].mean()) / max(image_width, image_height, 1)
    return {
        "point_score_mean": float(scores.mean()),
        "point_score_std": float(scores.std()),
        "point_fg_bg_gap_mean": float(fg_bg_gaps.mean()),
        "point_fg_bg_gap_std": float(fg_bg_gaps.std()),
        "point_pairwise_distance_mean": pairwise_mean,
    }


def mask_matcher_consistency_statistics(
    mask: np.ndarray,
    score_map: np.ndarray,
    points: list[dict[str, str]],
) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    scores = np.asarray(score_map, dtype=np.float64)
    if binary.ndim != 2 or scores.ndim != 2:
        raise ValueError("mask and score_map must both be two-dimensional.")
    if not points:
        raise ValueError("At least one prompt point is required.")
    score_mask = np.asarray(
        Image.fromarray(binary.astype(np.uint8)).resize(
            (scores.shape[1], scores.shape[0]),
            Image.Resampling.NEAREST,
        ),
        dtype=bool,
    )
    inside = scores[score_mask]
    outside = scores[~score_mask]
    inside_mean = float(inside.mean()) if inside.size else 0.0
    outside_mean = float(outside.mean()) if outside.size else 0.0
    flat_scores = scores.reshape(-1)
    top_count = max(1, int(np.ceil(flat_scores.size * 0.25)))
    top_indices = np.argpartition(flat_scores, flat_scores.size - top_count)[-top_count:]
    top_coverage = float(score_mask.reshape(-1)[top_indices].mean())
    weights = np.exp(flat_scores - float(flat_scores.max()))
    mass = float(weights[score_mask.reshape(-1)].sum() / max(float(weights.sum()), 1e-12))
    height, width = binary.shape
    prompt_hits = 0
    for point in points:
        x = min(max(int(round(float(point["x"]))), 0), width - 1)
        y = min(max(int(round(float(point["y"]))), 0), height - 1)
        prompt_hits += int(binary[y, x])
    return {
        "mask_score_mean": inside_mean,
        "outside_score_mean": outside_mean,
        "mask_score_contrast": inside_mean - outside_mean,
        "top_score_mask_coverage": top_coverage,
        "score_mass_in_mask": mass,
        "prompt_in_mask_ratio": prompt_hits / len(points),
    }


def train_nearest_centroid(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    label_key: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one training row is required.")
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, vector in zip(rows, (x - mean) / std):
        grouped[str(row[label_key])].append(vector)
    return {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": {label: np.mean(vectors, axis=0).tolist() for label, vectors in sorted(grouped.items())},
    }


def predict_nearest_centroid(model: dict[str, Any], row: dict[str, Any]) -> str:
    feature_names = [str(name) for name in model["feature_names"]]
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    vector = (np.asarray([float(row[name]) for name in feature_names], dtype=np.float64) - mean) / std
    return min(
        model["centroids"],
        key=lambda label: (float(np.square(vector - np.asarray(model["centroids"][label], dtype=np.float64)).sum()), label),
    )


def score_path_from_candidate(row: dict[str, str], workspace_root: Path) -> Path:
    points_path = Path(row["points_path"])
    if not points_path.is_absolute():
        points_path = workspace_root / points_path
    score_name = points_path.name.replace("_points.csv", "_score.npy")
    return points_path.parent.parent / "t1_matching" / score_name


def points_path_from_candidate(row: dict[str, str], workspace_root: Path) -> Path:
    points_path = Path(row["points_path"])
    return points_path if points_path.is_absolute() else workspace_root / points_path


def build_case_features(
    rows: list[dict[str, str]],
    workspace_root: Path,
    feature_lambda: float = 0.75,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    features: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        eligible = [
            row for row in candidates
            if row.get("round_name") == "round2" and float(row.get("lambda_bg", 0.0)) == feature_lambda
        ]
        if not eligible:
            raise ValueError(f"Missing round2 lambda={feature_lambda:g} score map for {case_id}.")
        source = min(eligible, key=lambda row: int(float(row["k"])))
        score_path = score_path_from_candidate(source, workspace_root)
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        features.append({
            "episode_id": case_id,
            "query_case_id": source["query_case_id"],
            **score_map_statistics(np.load(score_path)),
        })
    return features


def build_candidate_features(rows: list[dict[str, str]], workspace_root: Path) -> list[dict[str, Any]]:
    stats_cache: dict[Path, dict[str, float]] = {}
    point_cache: dict[Path, dict[str, float]] = {}
    point_rows_cache: dict[Path, list[dict[str, str]]] = {}
    consistency_cache: dict[tuple[Path, Path, Path], dict[str, float]] = {}
    layout_names = ("topk", "diverse_topk", "local_maxima", "boundary_mix", "cluster_medoid", "heatmap_cc")
    features: list[dict[str, Any]] = []
    for row in rows:
        score_path = score_path_from_candidate(row, workspace_root)
        if score_path not in stats_cache:
            if not score_path.exists():
                raise FileNotFoundError(score_path)
            stats_cache[score_path] = score_map_statistics(np.load(score_path))
        points_path = points_path_from_candidate(row, workspace_root)
        if points_path not in point_cache:
            with points_path.open(newline="", encoding="utf-8") as handle:
                point_rows_cache[points_path] = list(csv.DictReader(handle))
            point_cache[points_path] = point_geometry_statistics(point_rows_cache[points_path], image_width=1000, image_height=1000)
        mask_path = Path(row["mask_path"])
        if not mask_path.is_absolute():
            mask_path = workspace_root / mask_path
        consistency_key = (mask_path, score_path, points_path)
        if consistency_key not in consistency_cache:
            if mask_path.exists():
                mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                consistency_cache[consistency_key] = {
                    **mask_matcher_consistency_statistics(
                        mask,
                        np.load(score_path),
                        point_rows_cache[points_path],
                    ),
                    "mask_artifact_available": 1.0,
                }
            else:
                consistency_cache[consistency_key] = {
                    "mask_score_mean": 0.0,
                    "outside_score_mean": 0.0,
                    "mask_score_contrast": 0.0,
                    "top_score_mask_coverage": 0.0,
                    "score_mass_in_mask": 0.0,
                    "prompt_in_mask_ratio": 0.0,
                    "mask_artifact_available": 0.0,
                }
        prompt_count = float(row["prompt_count"])
        item: dict[str, Any] = {
            "episode_id": policy_group_id(row),
            "query_case_id": row["query_case_id"],
            "action": row["method"],
            "round_name": row["round_name"],
            "layout": row["layout"],
            "clustering": row["clustering"],
            "mask_path": row["mask_path"],
            "overlay_path": row["overlay_path"],
            "points_path": row["points_path"],
            "k": float(row["k"]),
            "lambda_bg": float(row["lambda_bg"]),
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "object_recall": float(row["object_recall"]),
            "merge_error": float(row["merge_error"]),
            "sam_score": float(row["sam_score"]),
            "pred_area_pct": float(row["pred_area_pct"]),
            "pred_components": float(row["pred_components"]),
            "prompt_count": prompt_count,
            "pred_components_per_prompt": float(row["pred_components"]) / max(prompt_count, 1.0),
            "pred_area_per_prompt": float(row["pred_area_pct"]) / max(prompt_count, 1.0),
            **stats_cache[score_path],
            **point_cache[points_path],
            **consistency_cache[consistency_key],
        }
        for layout in layout_names:
            item[f"layout_{layout}"] = 1.0 if row["layout"] == layout else 0.0
        features.append(item)
    return features


def select_policy_candidates(
    rows: list[dict[str, str]],
    predicted_actions: dict[str, str],
    fallback_action: str,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        requested = predicted_actions[case_id]
        chosen = next((row for row in candidates if row["method"] == requested), None)
        source = "policy"
        if chosen is None:
            chosen = next((row for row in candidates if row["method"] == fallback_action), None)
            source = "fallback"
        if chosen is None:
            raise ValueError(f"Neither requested nor fallback action exists for {case_id}.")
        selected.append({**chosen, "selected_action": chosen["method"], "selection_source": source})
    return selected


def select_ridge_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        best = max(candidates, key=lambda row: (float(row["predicted_reward"]), str(row["action"])))
        selected.append({**best, "selected_action": best["action"], "selection_source": "ridge_reward"})
    return selected


def select_top_n_candidates(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for _, candidates in sorted(by_case.items()):
        ranked = sorted(candidates, key=lambda row: (float(row["predicted_reward"]), str(row["action"])), reverse=True)
        for rank, row in enumerate(ranked[:budget], start=1):
            selected.append({**row, "policy_rank": rank})
    return selected


def add_within_case_policy_targets(
    rows: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    enriched: list[dict[str, Any]] = []
    for _, candidates in sorted(by_case.items()):
        dice_values = np.asarray([float(row["dice"]) for row in candidates], dtype=np.float64)
        mean_dice = float(dice_values.mean())
        min_dice = float(dice_values.min())
        dice_range = float(dice_values.max() - min_dice)
        top_actions = {
            str(row["action"])
            for row in sorted(candidates, key=lambda row: (float(row["dice"]), str(row["action"])), reverse=True)[:top_n]
        }
        for row in candidates:
            dice = float(row["dice"])
            normalized = (dice - min_dice) / dice_range if dice_range > 1e-12 else 0.5
            enriched.append({
                **row,
                "case_centered_dice": dice - mean_dice,
                "case_normalized_dice": normalized,
                "case_top_action": 1.0 if str(row["action"]) in top_actions else 0.0,
            })
    return enriched


def add_anchor_relative_features(
    rows: list[dict[str, Any]],
    anchor_action: str,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    enriched: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        anchor = next((row for row in candidates if str(row["action"]) == anchor_action), None)
        if anchor is None:
            raise ValueError(f"Missing anchor action {anchor_action!r} for {case_id}.")
        for row in candidates:
            item = {
                **row,
                "dice_gain_over_anchor": float(row["dice"]) - float(anchor["dice"]),
            }
            for name in feature_names:
                item[f"delta_{name}"] = float(row[name]) - float(anchor[name])
            enriched.append(item)
    return enriched


def select_top_n_with_anchor(
    rows: list[dict[str, Any]],
    budget: int,
    anchor_action: str,
) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        anchor = next((row for row in candidates if str(row["action"]) == anchor_action), None)
        if anchor is None:
            raise ValueError(f"Missing anchor action {anchor_action!r} for {case_id}.")
        alternatives = [row for row in candidates if str(row["action"]) != anchor_action]
        ranked = sorted(alternatives, key=lambda row: (float(row["predicted_reward"]), str(row["action"])), reverse=True)
        shortlist = ranked[: max(0, budget - 1)] + [anchor]
        for rank, row in enumerate(shortlist, start=1):
            selected.append({**row, "policy_rank": rank})
    return selected


def select_anchor_gated_candidates(
    rows: list[dict[str, Any]],
    anchor_action: str,
    threshold: float,
    uncertainty_weight: float = 0.0,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)
    selected: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        anchor = next((row for row in candidates if str(row["action"]) == anchor_action), None)
        if anchor is None:
            raise ValueError(f"Missing anchor action {anchor_action!r} for {case_id}.")
        alternatives = [row for row in candidates if str(row["action"]) != anchor_action]
        def risk_adjusted_gain(row: dict[str, Any]) -> float:
            return float(row["predicted_reward"]) - uncertainty_weight * float(row.get("predicted_reward_std", 0.0))

        best = max(alternatives, key=lambda row: (risk_adjusted_gain(row), str(row["action"]))) if alternatives else anchor
        risk_score = risk_adjusted_gain(best)
        decision = {
            "policy_best_action": best["action"],
            "policy_predicted_gain": float(best["predicted_reward"]),
            "policy_predicted_gain_std": float(best.get("predicted_reward_std", 0.0)),
            "policy_risk_adjusted_gain": risk_score,
            "gain_threshold": threshold,
            "uncertainty_weight": uncertainty_weight,
        }
        if best is not anchor and risk_score > threshold:
            chosen = {**best, **decision, "selected_action": best["action"], "selection_source": "anchor_gated_policy", "switched": 1}
        else:
            chosen = {**anchor, **decision, "selected_action": anchor["action"], "selection_source": "anchor_fallback", "switched": 0}
        selected.append(chosen)
    return selected


def summarize_selected_candidates(
    rows: list[dict[str, Any]],
    candidates_per_case: int = 1,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("At least one selected candidate is required.")
    metrics = ["dice", "iou", "object_recall", "merge_error"]
    summary: dict[str, float | int] = {"case_count": len(rows), "candidates_per_case": candidates_per_case}
    for metric in metrics:
        summary[f"mean_{metric}"] = round(sum(float(row[metric]) for row in rows) / len(rows), 6)
    return summary


def train_ridge_reward(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_key: str,
    alpha: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one training candidate is required.")
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[target_key]) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (x - mean) / std
    design = np.column_stack([np.ones(len(normalized)), normalized])
    regularizer = alpha * np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return {"feature_names": feature_names, "mean": mean.tolist(), "std": std.tolist(), "weights": weights.tolist(), "alpha": alpha}


def predict_ridge_reward(model: dict[str, Any], row: dict[str, Any]) -> float:
    feature_names = [str(name) for name in model["feature_names"]]
    x = np.asarray([float(row[name]) for name in feature_names], dtype=np.float64)
    normalized = (x - np.asarray(model["mean"], dtype=np.float64)) / np.asarray(model["std"], dtype=np.float64)
    return float(np.dot(np.concatenate([[1.0], normalized]), np.asarray(model["weights"], dtype=np.float64)))


def train_extra_trees_reward(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_key: str,
    min_samples_leaf: int,
) -> dict[str, Any]:
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[target_key]) for row in rows], dtype=np.float64)
    estimator = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf, random_state=20260715, n_jobs=1)
    estimator.fit(x, y)
    return {"feature_names": feature_names, "min_samples_leaf": min_samples_leaf, "estimator": estimator}


def predict_extra_trees_reward(model: dict[str, Any], row: dict[str, Any]) -> float:
    vector = [[float(row[name]) for name in model["feature_names"]]]
    return float(model["estimator"].predict(vector)[0])


def predict_extra_trees_distribution(model: dict[str, Any], row: dict[str, Any]) -> tuple[float, float]:
    scored = score_extra_trees_distribution_rows(model, [row])[0]
    return float(scored["predicted_reward"]), float(scored["predicted_reward_std"])


def score_extra_trees_distribution_rows(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    vectors = np.asarray(
        [[float(row[name]) for name in model["feature_names"]] for row in rows],
        dtype=np.float64,
    )
    predictions = np.vstack([
        estimator.predict(vectors)
        for estimator in model["estimator"].estimators_
    ])
    means = predictions.mean(axis=0)
    stds = predictions.std(axis=0)
    return [
        {**row, "predicted_reward": float(mean), "predicted_reward_std": float(std)}
        for row, mean, std in zip(rows, means, stds)
    ]


def apply_two_stage_policy(
    rows: list[dict[str, Any]],
    pre_sam_model: dict[str, Any],
    post_sam_model: dict[str, Any],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre_scored = [
        {
            **row,
            "pre_sam_reward": predict_extra_trees_reward(pre_sam_model, row),
            "predicted_reward": predict_extra_trees_reward(pre_sam_model, row),
        }
        for row in rows
    ]
    shortlist = select_top_n_candidates(pre_scored, budget=budget)
    post_scored = [
        {**row, "predicted_reward": predict_extra_trees_reward(post_sam_model, row)}
        for row in shortlist
    ]
    selected = [
        {**row, "selection_source": "two_stage_post_sam"}
        for row in select_ridge_candidates(post_scored)
    ]
    return shortlist, selected


def apply_anchored_two_stage_policy(
    rows: list[dict[str, Any]],
    pre_sam_model: dict[str, Any],
    post_sam_model: dict[str, Any],
    budget: int,
    anchor_action: str,
    threshold: float,
    uncertainty_weight: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shortlist, post_scored = score_anchored_two_stage_candidates(
        rows,
        pre_sam_model=pre_sam_model,
        post_sam_model=post_sam_model,
        budget=budget,
        anchor_action=anchor_action,
    )
    selected = select_anchor_gated_candidates(
        post_scored,
        anchor_action=anchor_action,
        threshold=threshold,
        uncertainty_weight=uncertainty_weight,
    )
    return shortlist, selected


def score_anchored_two_stage_candidates(
    rows: list[dict[str, Any]],
    pre_sam_model: dict[str, Any],
    post_sam_model: dict[str, Any],
    budget: int,
    anchor_action: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre_scored = [
        {**row, "predicted_reward": predict_extra_trees_reward(pre_sam_model, row)}
        for row in rows
    ]
    shortlist = select_top_n_with_anchor(pre_scored, budget=budget, anchor_action=anchor_action)
    return shortlist, score_extra_trees_distribution_rows(post_sam_model, shortlist)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def build_labeled_examples(rows: list[dict[str, str]], workspace_root: Path) -> list[dict[str, Any]]:
    labels = {row["episode_id"]: row["oracle_method"] for row in build_oracle_actions(rows)}
    return [{**feature, "action": labels[feature["episode_id"]]} for feature in build_case_features(rows, workspace_root)]


def majority_action(labeled_examples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in labeled_examples:
        counts[str(row["action"])] += 1
    return max(counts, key=lambda action: (counts[action], action))


def run_split(
    rows: list[dict[str, str]],
    model: dict[str, Any],
    fallback_action: str,
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = build_case_features(rows, workspace_root)
    predictions = {row["episode_id"]: predict_nearest_centroid(model, row) for row in features}
    learned = select_policy_candidates(rows, predictions, fallback_action)
    fixed = select_policy_candidates(rows, {row["episode_id"]: fallback_action for row in features}, fallback_action)
    return learned, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a matcher-only nearest-centroid adaptive prompt policy and evaluate fixed vs learned one-candidate selection.")
    parser.add_argument("--train-candidates", default="runs/research_direction/prompt_policy_oracle/train_grid_dinov2l_samvitb/candidates.csv")
    parser.add_argument("--val-candidates", default="runs/research_direction/prompt_policy_oracle/val_grid_dinov2l_samvitb/candidates.csv")
    parser.add_argument("--test-candidates", default="runs/research_direction/prompt_policy_oracle/test_grid_dinov2l_samvitb_fixedr3/candidates.csv")
    parser.add_argument("--out-dir", default="runs/research_direction/prompt_policy/adaptive_centroid_v1")
    parser.add_argument("--allowed-rounds", default="", help="Optional comma-separated rounds, e.g. round1,round2.")
    parser.add_argument("--allowed-k", default="", help="Optional comma-separated K values shared by train/val/test.")
    parser.add_argument("--allowed-lambda", default="", help="Optional comma-separated lambda values shared by train/val/test.")
    parser.add_argument(
        "--anchor-uncertainty-weights",
        default="0",
        help="Validation-searched ExtraTrees uncertainty penalties; keep 0 for the primary anchored policy.",
    )
    args = parser.parse_args()

    train_rows = read_csv(resolve(args.train_candidates))
    val_rows = read_csv(resolve(args.val_candidates))
    test_rows = read_csv(resolve(args.test_candidates))
    allowed_rounds = {item.strip() for item in args.allowed_rounds.split(",") if item.strip()}
    allowed_k = {int(item.strip()) for item in args.allowed_k.split(",") if item.strip()}
    allowed_lambda = {float(item.strip()) for item in args.allowed_lambda.split(",") if item.strip()}
    if allowed_rounds or allowed_k or allowed_lambda:
        train_rows = filter_candidate_grid(train_rows, allowed_rounds, allowed_k, allowed_lambda)
        val_rows = filter_candidate_grid(val_rows, allowed_rounds, allowed_k, allowed_lambda)
        test_rows = filter_candidate_grid(test_rows, allowed_rounds, allowed_k, allowed_lambda)
    labeled = build_labeled_examples(train_rows, ROOT)
    feature_names = ["score_mean", "score_std", "score_max", "top_quarter_mean", "normalized_entropy"]
    model = train_nearest_centroid(labeled, feature_names, "action")
    fallback = validation_best_action(val_rows)
    ridge_train = build_candidate_features(train_rows, ROOT)
    ridge_val = build_candidate_features(val_rows, ROOT)
    ridge_candidates: list[dict[str, Any]] = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        ridge_model = train_ridge_reward(ridge_train, RIDGE_FEATURES, "dice", alpha)
        val_scored = [{**row, "predicted_reward": predict_ridge_reward(ridge_model, row)} for row in ridge_val]
        ridge_candidates.append({"alpha": alpha, **summarize_selected_candidates(select_ridge_candidates(val_scored))})
    ridge_choice = max(ridge_candidates, key=lambda row: (float(row["mean_dice"]), -float(row["alpha"])))
    ridge_model = train_ridge_reward(ridge_train, RIDGE_FEATURES, "dice", float(ridge_choice["alpha"]))
    tree_candidates: list[dict[str, Any]] = []
    for min_leaf in (1, 2, 4, 8):
        tree_model = train_extra_trees_reward(ridge_train, RIDGE_FEATURES, "dice", min_leaf)
        val_scored = [{**row, "predicted_reward": predict_extra_trees_reward(tree_model, row)} for row in ridge_val]
        tree_candidates.append({"min_samples_leaf": min_leaf, **summarize_selected_candidates(select_ridge_candidates(val_scored))})
    tree_choice = max(tree_candidates, key=lambda row: (float(row["mean_dice"]), -int(row["min_samples_leaf"])))
    tree_model = train_extra_trees_reward(ridge_train, RIDGE_FEATURES, "dice", int(tree_choice["min_samples_leaf"]))
    leaf_options = (1, 2, 4, 8)
    pre_target_options = ("dice", "case_centered_dice", "case_normalized_dice", "case_top_action")
    policy_train = add_within_case_policy_targets(ridge_train, top_n=3)
    pre_sam_models = {
        (target_key, min_leaf): train_extra_trees_reward(policy_train, RIDGE_FEATURES, target_key, min_leaf)
        for target_key in pre_target_options
        for min_leaf in leaf_options
    }
    post_sam_models = {
        min_leaf: train_extra_trees_reward(ridge_train, POST_SAM_FEATURES, "dice", min_leaf)
        for min_leaf in leaf_options
    }
    two_stage_candidates: list[dict[str, Any]] = []
    for (pre_target, pre_leaf), pre_sam_model in pre_sam_models.items():
        for post_leaf, post_sam_model in post_sam_models.items():
            _, val_selected = apply_two_stage_policy(
                ridge_val,
                pre_sam_model=pre_sam_model,
                post_sam_model=post_sam_model,
                budget=3,
            )
            two_stage_candidates.append({
                "pre_target": pre_target,
                "pre_min_samples_leaf": pre_leaf,
                "post_min_samples_leaf": post_leaf,
                **summarize_selected_candidates(val_selected, candidates_per_case=3),
            })
    two_stage_choice = max(
        two_stage_candidates,
        key=lambda row: (
            float(row["mean_dice"]),
            str(row["pre_target"]),
            -int(row["pre_min_samples_leaf"]),
            -int(row["post_min_samples_leaf"]),
        ),
    )
    pre_sam_model = pre_sam_models[(
        str(two_stage_choice["pre_target"]),
        int(two_stage_choice["pre_min_samples_leaf"]),
    )]
    post_sam_model = post_sam_models[int(two_stage_choice["post_min_samples_leaf"])]
    anchor_train = add_anchor_relative_features(ridge_train, fallback, POST_SAM_FEATURES)
    anchor_val = add_anchor_relative_features(ridge_val, fallback, POST_SAM_FEATURES)
    anchor_pre_models = {
        min_leaf: train_extra_trees_reward(anchor_train, ANCHOR_PRE_FEATURES, "dice_gain_over_anchor", min_leaf)
        for min_leaf in leaf_options
    }
    anchor_post_models = {
        min_leaf: train_extra_trees_reward(anchor_train, ANCHOR_POST_FEATURES, "dice_gain_over_anchor", min_leaf)
        for min_leaf in leaf_options
    }
    anchor_thresholds = (-0.02, 0.0, 0.01, 0.02, 0.05, 0.1)
    uncertainty_weights = tuple(
        float(item.strip())
        for item in args.anchor_uncertainty_weights.split(",")
        if item.strip()
    )
    if not uncertainty_weights:
        raise ValueError("At least one anchor uncertainty weight is required.")
    anchored_candidates: list[dict[str, Any]] = []
    for pre_leaf, anchor_pre_model in anchor_pre_models.items():
        for post_leaf, anchor_post_model in anchor_post_models.items():
            _, val_post_scored = score_anchored_two_stage_candidates(
                anchor_val,
                pre_sam_model=anchor_pre_model,
                post_sam_model=anchor_post_model,
                budget=3,
                anchor_action=fallback,
            )
            for threshold in anchor_thresholds:
                for uncertainty_weight in uncertainty_weights:
                    val_selected = select_anchor_gated_candidates(
                        val_post_scored,
                        anchor_action=fallback,
                        threshold=threshold,
                        uncertainty_weight=uncertainty_weight,
                    )
                    anchored_candidates.append({
                        "pre_min_samples_leaf": pre_leaf,
                        "post_min_samples_leaf": post_leaf,
                        "gain_threshold": threshold,
                        "uncertainty_weight": uncertainty_weight,
                        **summarize_selected_candidates(val_selected, candidates_per_case=3),
                    })
    anchored_choice = max(
        anchored_candidates,
        key=lambda row: (
            float(row["mean_dice"]),
            float(row["gain_threshold"]),
            float(row["uncertainty_weight"]),
            -int(row["pre_min_samples_leaf"]),
            -int(row["post_min_samples_leaf"]),
        ),
    )
    anchor_pre_model = anchor_pre_models[int(anchored_choice["pre_min_samples_leaf"])]
    anchor_post_model = anchor_post_models[int(anchored_choice["post_min_samples_leaf"])]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "ridge_validation_selection.csv", ridge_candidates)
    write_csv(out_dir / "extra_trees_validation_selection.csv", tree_candidates)
    write_csv(out_dir / "two_stage_validation_selection.csv", two_stage_candidates)
    write_csv(out_dir / "anchored_validation_selection.csv", anchored_candidates)
    joblib.dump(tree_model["estimator"], out_dir / "extra_trees_reward.joblib")
    joblib.dump(pre_sam_model["estimator"], out_dir / "two_stage_pre_sam_reward.joblib")
    joblib.dump(post_sam_model["estimator"], out_dir / "two_stage_post_sam_reward.joblib")
    joblib.dump(anchor_pre_model["estimator"], out_dir / "anchored_pre_sam_gain.joblib")
    joblib.dump(anchor_post_model["estimator"], out_dir / "anchored_post_sam_gain.joblib")
    all_summary: list[dict[str, Any]] = []
    for split_name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        learned, fixed = run_split(rows, model, fallback, ROOT)
        ridge_rows = build_candidate_features(rows, ROOT)
        ridge_scored = [{**row, "predicted_reward": predict_ridge_reward(ridge_model, row)} for row in ridge_rows]
        ridge_selected = select_ridge_candidates(ridge_scored)
        tree_scored = [{**row, "predicted_reward": predict_extra_trees_reward(tree_model, row)} for row in ridge_rows]
        tree_selected = select_ridge_candidates(tree_scored)
        two_stage_shortlist, two_stage_selected = apply_two_stage_policy(
            ridge_rows,
            pre_sam_model=pre_sam_model,
            post_sam_model=post_sam_model,
            budget=3,
        )
        anchor_rows = add_anchor_relative_features(ridge_rows, fallback, POST_SAM_FEATURES)
        anchored_shortlist, anchored_selected = apply_anchored_two_stage_policy(
            anchor_rows,
            pre_sam_model=anchor_pre_model,
            post_sam_model=anchor_post_model,
            budget=3,
            anchor_action=fallback,
            threshold=float(anchored_choice["gain_threshold"]),
            uncertainty_weight=float(anchored_choice["uncertainty_weight"]),
        )
        write_csv(out_dir / f"{split_name}_learned_predictions.csv", learned)
        write_csv(out_dir / f"{split_name}_fixed_predictions.csv", fixed)
        write_csv(out_dir / f"{split_name}_ridge_predictions.csv", ridge_selected)
        write_csv(out_dir / f"{split_name}_extra_trees_predictions.csv", tree_selected)
        write_csv(out_dir / f"{split_name}_two_stage_shortlist.csv", two_stage_shortlist)
        write_csv(out_dir / f"{split_name}_two_stage_predictions.csv", two_stage_selected)
        write_csv(out_dir / f"{split_name}_anchored_shortlist.csv", anchored_shortlist)
        write_csv(out_dir / f"{split_name}_anchored_predictions.csv", anchored_selected)
        all_summary.extend([
            {"split": split_name, "method": "fixed_majority_action", "fallback_action": fallback, **summarize_selected_candidates(fixed)},
            {"split": split_name, "method": "learned_nearest_centroid", "fallback_action": fallback, **summarize_selected_candidates(learned)},
            {"split": split_name, "method": "learned_ridge_reward", "fallback_action": fallback, **summarize_selected_candidates(ridge_selected)},
            {"split": split_name, "method": "learned_extra_trees_reward", "fallback_action": fallback, **summarize_selected_candidates(tree_selected)},
            {"split": split_name, "method": "learned_top3_post_sam", "fallback_action": fallback, **summarize_selected_candidates(two_stage_selected, candidates_per_case=3)},
            {"split": split_name, "method": "learned_anchored_top3", "fallback_action": fallback, **summarize_selected_candidates(anchored_selected, candidates_per_case=3)},
        ])
    write_csv(out_dir / "summary.csv", all_summary)
    (out_dir / "model.json").write_text(json.dumps({"nearest_centroid_model": model, "ridge_reward_model": ridge_model, "extra_trees_model_path": "extra_trees_reward.joblib", "extra_trees_min_samples_leaf_selected_on_validation": tree_choice["min_samples_leaf"], "ridge_features": RIDGE_FEATURES, "ridge_alpha_selected_on_validation": ridge_choice["alpha"], "two_stage_candidate_budget": 3, "two_stage_pre_sam_model_path": "two_stage_pre_sam_reward.joblib", "two_stage_post_sam_model_path": "two_stage_post_sam_reward.joblib", "two_stage_pre_target_selected_on_validation": two_stage_choice["pre_target"], "two_stage_pre_min_samples_leaf_selected_on_validation": two_stage_choice["pre_min_samples_leaf"], "two_stage_post_min_samples_leaf_selected_on_validation": two_stage_choice["post_min_samples_leaf"], "anchored_pre_sam_model_path": "anchored_pre_sam_gain.joblib", "anchored_post_sam_model_path": "anchored_post_sam_gain.joblib", "anchored_pre_min_samples_leaf_selected_on_validation": anchored_choice["pre_min_samples_leaf"], "anchored_post_min_samples_leaf_selected_on_validation": anchored_choice["post_min_samples_leaf"], "anchored_gain_threshold_selected_on_validation": anchored_choice["gain_threshold"], "anchored_uncertainty_weight_selected_on_validation": anchored_choice["uncertainty_weight"], "pre_sam_features": RIDGE_FEATURES, "post_sam_features": POST_SAM_FEATURES, "anchor_pre_features": ANCHOR_PRE_FEATURES, "anchor_post_features": ANCHOR_POST_FEATURES, "fallback_action": fallback, "fallback_selection_source": "validation_mean_dice", "allowed_rounds": sorted(allowed_rounds) if allowed_rounds else ["round1", "round2", "round3"], "allowed_k": sorted(allowed_k), "allowed_lambda": sorted(allowed_lambda), "feature_source": "pre-SAM matcher score maps, selected point geometry, and action metadata; post-SAM selector additionally uses SAM confidence, mask morphology, and matcher-mask consistency without GT; ExtraTrees ensemble spread supports validation-calibrated risk gating", "train_candidates": str(resolve(args.train_candidates)), "val_candidates": str(resolve(args.val_candidates)), "test_candidates": str(resolve(args.test_candidates))}, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.csv'}")


def build_oracle_actions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[policy_group_id(row)].append(row)

    oracle: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        best = max(candidates, key=lambda row: (float(row["dice"]), row["method"]))
        oracle.append(
            {
                "episode_id": case_id,
                "query_case_id": candidates[0]["query_case_id"],
                "oracle_method": best["method"],
                "oracle_dice": float(best["dice"]),
                "k": int(float(best["k"])),
                "lambda_bg": float(best["lambda_bg"]),
                "layout": best["layout"],
                "clustering": best["clustering"],
            }
        )
    return oracle


if __name__ == "__main__":
    main()
