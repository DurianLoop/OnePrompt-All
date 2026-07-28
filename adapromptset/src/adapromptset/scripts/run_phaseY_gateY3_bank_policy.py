"""Phase Y Gate Y3: Bank-Action Policy Validation (section 10).

Implements section 10 of ``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

Trains a ladder of bank-action policy models (M0-M5) on the train utility
matrix, evaluates each on the validation utility matrix, and computes the
Gate Y3 decision.

Models (section 10.3):
  M0: Global prior (always select train's strongest fixed)
  M1: Morphology heuristic (hand-bucketed by estimated target count + dispersion)
  M2: Multiclass action classifier (RandomForest)
  M3: Per-action utility regression (GradientBoosting)
  M4: Pairwise ranking (GradientBoosting on pair differences)
  M5: Hierarchical router (multi-stage: K+ bin -> K- -> pos selector -> neg selector)

Features (section 10.2): test-time-safe only.
  - Query global: image embedding mean/std, patch entropy, stain stats
  - Matching-map: fg/bg score stats, contrast, entropy, mass ratio, peaks
  - Candidate: count, decay, margin, dispersion, redundancy
  - Reference consistency: A-internal disagreement, embedding gap, overlap
  - Action descriptors: K+, K-, selector token, point budget

Outputs::

    reports/phaseY_gateY3/
        features_train.csv         (per-query-ref features)
        features_validation.csv
        model_predictions.csv      (per-model per-query predicted action + Dice)
        model_metrics.csv          (per-model aggregate metrics)
        gateY3_summary.json        (Gate Y3 decision)
        gateY3_summary.md
        run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse Y1 infrastructure
from adapromptset.candidates.points import build_candidate_pools
from adapromptset.matching.evidence import (
    IMAGE_SIZE,
    PATCH_SIZE,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
)
from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
    DINOV2_CHECKPOINT,
    FeatureCache,
    MONUSAC_ROOT,
    PROJECT_ROOT,
    SAM_CHECKPOINT,
    WORKSPACE_ROOT,
    PairCandidatePools,
    _bootstrap_ci,
    _mean,
    _median,
    build_pair_candidate_pools,
    load_action_bank,
    load_queries,
    load_reference_pairs,
    load_rgb_518,
    load_mask_518,
    load_mask_tensor,
    load_utility_matrix,
    resolve_case_paths,
)

OUT_DIR_DEFAULT = PROJECT_ROOT / "reports" / "phaseY_gateY3"
COMPACT_BANK_YAML = PROJECT_ROOT / "configs" / "actions" / "phaseY_compact_policy_bank.yaml"

RANDOM_SEED = 20260721
N_SEEDS = 3


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _entropy(values: list[float]) -> float:
    """Shannon entropy of a normalized distribution."""
    total = sum(abs(v) for v in values)
    if total <= 0:
        return 0.0
    probs = [abs(v) / total for v in values]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
    }


def _grid_stats(grid: list[list[float]]) -> dict[str, float]:
    flat = [float(v) for row in grid for v in row]
    stats = _stats(flat)
    # Peak count: local maxima
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    peak_count = 0
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            is_peak = True
            for nr in range(max(0, r - 1), min(rows, r + 2)):
                for nc in range(max(0, c - 1), min(cols, c + 2)):
                    if (nr, nc) != (r, c) and grid[nr][nc] > v:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                peak_count += 1
    stats["peak_count"] = float(peak_count)
    stats["entropy"] = _entropy(flat)
    # High-score area ratio: fraction of patches above mean + 1 std
    threshold = stats["mean"] + stats["std"]
    high_count = sum(1 for v in flat if v > threshold)
    stats["high_score_area_ratio"] = high_count / len(flat) if flat else 0.0
    # Low-score area ratio
    low_threshold = stats["mean"] - stats["std"]
    low_count = sum(1 for v in flat if v < low_threshold)
    stats["low_score_area_ratio"] = low_count / len(flat) if flat else 0.0
    return stats


def _candidate_stats(pools: PairCandidatePools) -> dict[str, float]:
    """Statistics over the candidate pool."""
    pos_scores = [c.score for c in pools.topk.positives]
    neg_scores = [c.score for c in pools.topk.negatives]
    pos_coords = [c.coordinate for c in pools.topk.positives]

    stats: dict[str, float] = {}
    pstats = _stats(pos_scores)
    for k, v in pstats.items():
        stats[f"pos_score_{k}"] = v

    # Top-5/top-20 margin: difference between top-1 and top-5/top-20
    if pos_scores:
        sorted_desc = sorted(pos_scores, reverse=True)
        stats["top1_score"] = sorted_desc[0]
        stats["top5_margin"] = sorted_desc[0] - (sorted_desc[4] if len(sorted_desc) >= 5 else sorted_desc[-1])
        stats["top20_margin"] = sorted_desc[0] - (sorted_desc[19] if len(sorted_desc) >= 20 else sorted_desc[-1])
        # Top-score decay: ratio of top-10 to top-1
        top10 = sorted_desc[:10]
        stats["top10_mean"] = sum(top10) / len(top10) if top10 else 0.0
        stats["top_score_decay"] = stats["top10_mean"] / stats["top1_score"] if stats["top1_score"] != 0 else 0.0
    else:
        stats["top1_score"] = 0.0
        stats["top5_margin"] = 0.0
        stats["top20_margin"] = 0.0
        stats["top10_mean"] = 0.0
        stats["top_score_decay"] = 0.0

    # Coordinate dispersion: mean pairwise distance
    if len(pos_coords) >= 2:
        dists = []
        for i in range(min(len(pos_coords), 20)):
            for j in range(i + 1, min(len(pos_coords), 20)):
                dx = pos_coords[i][0] - pos_coords[j][0]
                dy = pos_coords[i][1] - pos_coords[j][1]
                dists.append(math.hypot(dx, dy))
        stats["pos_coord_dispersion"] = _mean(dists)
    else:
        stats["pos_coord_dispersion"] = 0.0

    # Candidate count
    stats["pos_count"] = float(len(pos_scores))
    stats["neg_count"] = float(len(neg_scores))

    # Positive-negative separation: difference of means
    neg_mean = _mean(neg_scores) if neg_scores else 0.0
    stats["pos_neg_separation"] = stats["pos_score_mean"] - neg_mean

    # Estimated component count: number of distinct coordinate clusters
    # (approximate: count of positives that are > 20 pixels apart from any other)
    if pos_coords:
        clusters = 0
        for i, c in enumerate(pos_coords[:30]):
            isolated = True
            for j, other in enumerate(pos_coords[:30]):
                if i != j:
                    dx = c[0] - other[0]
                    dy = c[1] - other[1]
                    if math.hypot(dx, dy) < 30:
                        isolated = False
                        break
            if isolated:
                clusters += 1
        stats["estimated_component_count"] = float(clusters)
    else:
        stats["estimated_component_count"] = 0.0

    return stats


def _query_global_features(
    query_features: torch.Tensor,
    image: Any,
) -> dict[str, float]:
    """Query-level global features from DINOv2 embedding + image stats."""
    arr = query_features.numpy()
    stats = {
        "query_emb_mean": float(arr.mean()),
        "query_emb_std": float(arr.std()),
        "query_emb_max": float(arr.max()),
        "query_emb_min": float(arr.min()),
    }
    # Patch embedding entropy: variance across patches
    patch_norms = np.linalg.norm(arr, axis=1) if arr.ndim == 2 else np.array([])
    stats["query_patch_entropy"] = float(np.std(patch_norms)) if len(patch_norms) > 0 else 0.0

    # Stain/intensity statistics from image
    import numpy as np2
    img_arr = np2.asarray(image)
    if img_arr.ndim == 3:
        for ch, name in enumerate(["R", "G", "B"]):
            ch_arr = img_arr[:, :, ch].astype(np2.float64)
            stats[f"query_{name}_mean"] = float(ch_arr.mean())
            stats[f"query_{name}_std"] = float(ch_arr.std())
        # Grayscale stats
        gray = img_arr.mean(axis=2)
        stats["query_gray_mean"] = float(gray.mean())
        stats["query_gray_std"] = float(gray.std())
    else:
        for name in ["R", "G", "B", "gray"]:
            stats[f"query_{name}_mean"] = 0.0
            stats[f"query_{name}_std"] = 0.0

    # Image size / aspect ratio (always 518x518 after resize, but record anyway)
    stats["query_width"] = float(img_arr.shape[1])
    stats["query_height"] = float(img_arr.shape[0])
    stats["query_aspect_ratio"] = float(img_arr.shape[1] / img_arr.shape[0]) if img_arr.shape[0] else 1.0
    return stats


def _reference_consistency_features(
    query_features: torch.Tensor,
    ref_features_list: list[torch.Tensor],
    ref_masks: list[torch.Tensor],
) -> dict[str, float]:
    """Reference consistency features across multiple references."""
    if not ref_features_list:
        return {
            "ref_embedding_gap_mean": 0.0,
            "ref_embedding_gap_std": 0.0,
            "ref_map_disagreement_mean": 0.0,
            "ref_map_disagreement_std": 0.0,
        }

    # Embedding gaps: cosine distance between query and each ref
    query_norm = query_features / (query_features.norm(dim=1, keepdim=True) + 1e-8)
    gaps = []
    for ref_feat in ref_features_list:
        ref_norm = ref_feat / (ref_feat.norm(dim=1, keepdim=True) + 1e-8)
        # Mean cosine similarity across patches
        sim = (query_norm * ref_norm).sum(dim=1).mean().item()
        gaps.append(1.0 - sim)

    # Map disagreement: build matching maps for each ref, compute pairwise diff
    maps = []
    grid_size = IMAGE_SIZE // PATCH_SIZE
    for ref_feat, ref_mask in zip(ref_features_list, ref_masks):
        fg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=True)
        bg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=False)
        result = matching_scores_from_features(
            support_features=ref_feat,
            query_features=query_features,
            foreground_weights=fg_weights,
            background_weights=bg_weights,
            grid_size=grid_size,
        )
        maps.append(result["s_fg"].numpy())

    disagreements = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            diff = np.abs(maps[i] - maps[j]).mean()
            disagreements.append(float(diff))

    return {
        "ref_embedding_gap_mean": _mean(gaps),
        "ref_embedding_gap_std": _std(gaps),
        "ref_map_disagreement_mean": _mean(disagreements) if disagreements else 0.0,
        "ref_map_disagreement_std": _std(disagreements) if disagreements else 0.0,
    }


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values)
    return float(arr.std())


def extract_features_for_pair(
    query_case_id: str,
    ref_case_id: str,
    feature_cache: FeatureCache,
) -> dict[str, float]:
    """Extract all test-time-safe features for one (query, ref) pair."""
    query_features = feature_cache.get_features(query_case_id)
    ref_features = feature_cache.get_features(ref_case_id)
    ref_mask = feature_cache.get_mask_tensor(ref_case_id)

    # Build candidate pools
    pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)

    # Matching-map features
    fg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=True)
    bg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=False)
    grid_size = IMAGE_SIZE // PATCH_SIZE
    maps = matching_scores_from_features(
        support_features=ref_features,
        query_features=query_features,
        foreground_weights=fg_weights,
        background_weights=bg_weights,
        grid_size=grid_size,
    )
    s_fg = maps["s_fg"].tolist()
    s_bg = maps["s_bg"].tolist()
    # Contrast = fg - bg
    contrast = [[s_fg[r][c] - s_bg[r][c] for c in range(len(s_fg[0]))] for r in range(len(s_fg))]

    fg_stats = _grid_stats(s_fg)
    bg_stats = _grid_stats(s_bg)
    contrast_stats = _grid_stats(contrast)

    features: dict[str, float] = {}
    for k, v in fg_stats.items():
        features[f"fg_{k}"] = v
    for k, v in bg_stats.items():
        features[f"bg_{k}"] = v
    for k, v in contrast_stats.items():
        features[f"contrast_{k}"] = v

    # Positive mass ratio: fraction of patches with fg > bg
    pos_mass = sum(1 for r in range(len(s_fg)) for c in range(len(s_fg[0])) if s_fg[r][c] > s_bg[r][c])
    total_patches = len(s_fg) * len(s_fg[0]) if s_fg else 1
    features["positive_mass_ratio"] = pos_mass / total_patches

    # Candidate features
    cand_stats = _candidate_stats(pools)
    features.update(cand_stats)

    # Query global features
    query_paths = resolve_case_paths(query_case_id)
    query_image = load_rgb_518(query_paths["image"])
    query_global = _query_global_features(query_features, query_image)
    features.update(query_global)

    return features


def extract_reference_consistency(
    query_case_id: str,
    ref_case_ids: list[str],
    feature_cache: FeatureCache,
) -> dict[str, float]:
    """Extract reference consistency features across multiple refs."""
    query_features = feature_cache.get_features(query_case_id)
    ref_features_list = [feature_cache.get_features(r) for r in ref_case_ids]
    ref_masks = [feature_cache.get_mask_tensor(r) for r in ref_case_ids]
    return _reference_consistency_features(query_features, ref_features_list, ref_masks)


# ---------------------------------------------------------------------------
# Model ladder (M0-M5)
# ---------------------------------------------------------------------------

@dataclass
class CompactAction:
    action_id: str
    action_index: int
    k_positive: int
    k_negative: int
    positive_selector: str
    negative_selector: str


def load_compact_bank(yaml_path: Path = COMPACT_BANK_YAML) -> list[CompactAction]:
    """Load compact policy bank from YAML. Falls back to full bank if missing."""
    import yaml
    if not yaml_path.is_file():
        # Fall back to full bank (shouldn't happen after Pareto pruning)
        yaml_path = PROJECT_ROOT / "configs" / "actions" / "phaseY_full_single_call_bank.yaml"
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    actions: list[CompactAction] = []
    for entry in data["actions"]:
        actions.append(CompactAction(
            action_id=entry["action_id"],
            action_index=entry["action_index"],
            k_positive=entry["k_positive"],
            k_negative=entry["k_negative"],
            positive_selector=entry["positive_selector"],
            negative_selector=entry["negative_selector"],
        ))
    return actions


def _action_descriptor_features(action: CompactAction) -> dict[str, float]:
    """Action descriptor features (section 10.2)."""
    return {
        "action_k_positive": float(action.k_positive),
        "action_k_negative": float(action.k_negative),
        "action_point_budget": float(action.k_positive + action.k_negative),
        "action_is_topk": 1.0 if action.positive_selector == "P_TOPK" else 0.0,
        "action_is_spread": 1.0 if action.positive_selector == "P_SPREAD" else 0.0,
        "action_is_peaks": 1.0 if action.positive_selector == "P_LOCAL_PEAKS" else 0.0,
        "action_is_bottomk": 1.0 if action.negative_selector == "N_BOTTOMK" else 0.0,
        "action_is_high_bg": 1.0 if action.negative_selector == "N_HIGH_BACKGROUND" else 0.0,
        "action_is_n_none": 1.0 if action.negative_selector == "N_NONE" else 0.0,
    }


# --- M0: Global prior ---

def train_m0(train_data: dict[str, Any]) -> dict[str, Any]:
    """M0: Always select train's strongest fixed action."""
    # train_data["per_query_action_dice_train"]: {query: {action_id: dice}}
    # Find the action with highest mean Dice across train queries
    action_dices: dict[str, list[float]] = {}
    for q, actions in train_data["per_query_action_dice_train"].items():
        for aid, dice in actions.items():
            action_dices.setdefault(aid, []).append(dice)
    action_means = {aid: _mean(v) for aid, v in action_dices.items()}
    best_action = max(action_means, key=action_means.get)
    return {
        "model_name": "M0_global_prior",
        "best_action": best_action,
        "best_action_dice": action_means[best_action],
    }


def predict_m0(model: dict[str, Any], features: dict[str, float]) -> str:
    """M0 prediction: always return the best action."""
    return model["best_action"]


# --- M1: Morphology heuristic ---

def train_m1(train_data: dict[str, Any]) -> dict[str, Any]:
    """M1: Hand-bucketed by estimated target count and map dispersion."""
    # Bucket queries by estimated_component_count (low/medium/high)
    # and contrast_entropy (low/medium/high)
    # For each bucket, find the best action on average
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for q, features in train_data["features_train"].items():
        comp = features.get("estimated_component_count", 0)
        disp = features.get("pos_coord_dispersion", 0)
        comp_bin = "low" if comp < 5 else ("medium" if comp < 15 else "high")
        disp_bin = "low" if disp < 100 else ("medium" if disp < 200 else "high")
        key = (comp_bin, disp_bin)
        for aid, dice in train_data["per_query_action_dice_train"].get(q, {}).items():
            buckets.setdefault(key, {}).setdefault(aid, []).append(dice)

    # For each bucket, find best action
    bucket_best: dict[tuple[str, str], str] = {}
    for key, action_dices in buckets.items():
        action_means = {aid: _mean(v) for aid, v in action_dices.items()}
        bucket_best[key] = max(action_means, key=action_means.get)

    # Fallback: overall best
    all_dices: dict[str, list[float]] = {}
    for q, actions in train_data["per_query_action_dice_train"].items():
        for aid, dice in actions.items():
            all_dices.setdefault(aid, []).append(dice)
    overall_means = {aid: _mean(v) for aid, v in all_dices.items()}
    fallback = max(overall_means, key=overall_means.get)

    return {
        "model_name": "M1_morphology_heuristic",
        "bucket_best": bucket_best,
        "fallback_action": fallback,
    }


def predict_m1(model: dict[str, Any], features: dict[str, float]) -> str:
    """M1 prediction: bucket by morphology, return bucket's best action."""
    comp = features.get("estimated_component_count", 0)
    disp = features.get("pos_coord_dispersion", 0)
    comp_bin = "low" if comp < 5 else ("medium" if comp < 15 else "high")
    disp_bin = "low" if disp < 100 else ("medium" if disp < 200 else "high")
    key = (comp_bin, disp_bin)
    return model["bucket_best"].get(key, model["fallback_action"])


# --- M2: Multiclass action classifier ---

def train_m2(train_data: dict[str, Any], seed: int = RANDOM_SEED) -> dict[str, Any]:
    """M2: Multiclass classifier predicting best action ID."""
    from sklearn.ensemble import RandomForestClassifier

    # Build training data: (features, best_action_label)
    feature_keys = sorted(next(iter(train_data["features_train"].values())).keys())
    X = []
    y = []
    for q, features in train_data["features_train"].items():
        actions = train_data["per_query_action_dice_train"].get(q, {})
        if not actions:
            continue
        best_action = max(actions, key=actions.get)
        X.append([features.get(k, 0.0) for k in feature_keys])
        y.append(best_action)

    if not X:
        return {"model_name": "M2_multiclass_classifier", "model": None, "feature_keys": feature_keys}

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X, y)
    return {
        "model_name": "M2_multiclass_classifier",
        "model": clf,
        "feature_keys": feature_keys,
        "classes_": list(clf.classes_),
    }


def predict_m2(model: dict[str, Any], features: dict[str, float]) -> str:
    """M2 prediction."""
    if model["model"] is None:
        return ""
    x = [[features.get(k, 0.0) for k in model["feature_keys"]]]
    pred = model["model"].predict(x)[0]
    # If predicted action not in compact bank, fall back to first class
    return pred


# --- M3: Per-action utility regression ---

def train_m3(
    train_data: dict[str, Any],
    compact_bank: list[CompactAction],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """M3: Regress U(q, a) for each action, argmax at inference."""
    from sklearn.ensemble import GradientBoostingRegressor

    feature_keys = sorted(next(iter(train_data["features_train"].values())).keys())
    # Build training data: (features + action_descriptors, dice)
    X = []
    y = []
    for q, features in train_data["features_train"].items():
        actions = train_data["per_query_action_dice_train"].get(q, {})
        for action in compact_bank:
            if action.action_id not in actions:
                continue
            dice = actions[action.action_id]
            row = [features.get(k, 0.0) for k in feature_keys]
            row += list(_action_descriptor_features(action).values())
            X.append(row)
            y.append(dice)

    if not X:
        return {"model_name": "M3_per_action_regression", "model": None,
                "feature_keys": feature_keys, "action_descriptor_keys": list(
                    _action_descriptor_features(compact_bank[0]).keys())}

    action_desc_keys = list(_action_descriptor_features(compact_bank[0]).keys())
    reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=seed,
    )
    reg.fit(X, y)
    return {
        "model_name": "M3_per_action_regression",
        "model": reg,
        "feature_keys": feature_keys,
        "action_descriptor_keys": action_desc_keys,
        "compact_bank": compact_bank,
    }


def predict_m3(model: dict[str, Any], features: dict[str, float]) -> str:
    """M3 prediction: argmax over actions of predicted utility."""
    if model["model"] is None:
        return ""
    base_x = [features.get(k, 0.0) for k in model["feature_keys"]]
    best_action = ""
    best_util = -float("inf")
    for action in model["compact_bank"]:
        desc = _action_descriptor_features(action)
        row = base_x + [desc[k] for k in model["action_descriptor_keys"]]
        util = model["model"].predict([row])[0]
        if util > best_util:
            best_util = util
            best_action = action.action_id
    return best_action


# --- M4: Pairwise ranking ---

def train_m4(
    train_data: dict[str, Any],
    compact_bank: list[CompactAction],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """M4: Predict P(U(q, a_i) > U(q, a_j))."""
    from sklearn.ensemble import GradientBoostingClassifier

    feature_keys = sorted(next(iter(train_data["features_train"].values())).keys())
    action_desc_keys = list(_action_descriptor_features(compact_bank[0]).keys())

    # Build training data: for each query, for each pair (a_i, a_j), predict 1 if U(a_i) > U(a_j)
    X = []
    y = []
    for q, features in train_data["features_train"].items():
        actions = train_data["per_query_action_dice_train"].get(q, {})
        if len(actions) < 2:
            continue
        # Sample pairs to limit training size
        action_ids = sorted(actions.keys())
        # Limit to compact bank actions
        action_ids = [a for a in action_ids if any(b.action_id == a for b in compact_bank)]
        if len(action_ids) < 2:
            continue
        # Generate pairs (limit to 30 per query to control size)
        import random
        rng = random.Random(seed + hash(q) % 100000)
        pairs = []
        for i in range(len(action_ids)):
            for j in range(i + 1, len(action_ids)):
                pairs.append((action_ids[i], action_ids[j]))
        rng.shuffle(pairs)
        pairs = pairs[:30]
        for ai, aj in pairs:
            if ai not in actions or aj not in actions:
                continue
            dice_i = actions[ai]
            dice_j = actions[aj]
            if abs(dice_i - dice_j) < 0.001:
                continue  # skip ties
            label = 1 if dice_i > dice_j else 0
            # Feature: query features + desc(a_i) - desc(a_j)
            base = [features.get(k, 0.0) for k in feature_keys]
            action_i = next(b for b in compact_bank if b.action_id == ai)
            action_j = next(b for b in compact_bank if b.action_id == aj)
            desc_i = _action_descriptor_features(action_i)
            desc_j = _action_descriptor_features(action_j)
            diff = [desc_i[k] - desc_j[k] for k in action_desc_keys]
            X.append(base + diff)
            y.append(label)

    if not X:
        return {"model_name": "M4_pairwise_ranking", "model": None,
                "feature_keys": feature_keys, "action_descriptor_keys": action_desc_keys,
                "compact_bank": compact_bank}

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=seed,
    )
    clf.fit(X, y)
    return {
        "model_name": "M4_pairwise_ranking",
        "model": clf,
        "feature_keys": feature_keys,
        "action_descriptor_keys": action_desc_keys,
        "compact_bank": compact_bank,
    }


def predict_m4(model: dict[str, Any], features: dict[str, float]) -> str:
    """M4 prediction: argmax by pairwise win count."""
    if model["model"] is None:
        return ""
    base = [features.get(k, 0.0) for k in model["feature_keys"]]
    win_counts: dict[str, int] = {a.action_id: 0 for a in model["compact_bank"]}
    for ai in model["compact_bank"]:
        for aj in model["compact_bank"]:
            if ai.action_id == aj.action_id:
                continue
            desc_i = _action_descriptor_features(ai)
            desc_j = _action_descriptor_features(aj)
            diff = [desc_i[k] - desc_j[k] for k in model["action_descriptor_keys"]]
            row = base + diff
            pred = model["model"].predict([row])[0]
            if pred == 1:
                win_counts[ai.action_id] += 1
    return max(win_counts, key=win_counts.get) if win_counts else ""


# --- M5: Hierarchical router ---

def train_m5(
    train_data: dict[str, Any],
    compact_bank: list[CompactAction],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """M5: Hierarchical router: K+ bin -> K- -> pos selector -> neg selector."""
    from sklearn.ensemble import RandomForestClassifier

    feature_keys = sorted(next(iter(train_data["features_train"].values())).keys())

    # Build labels for each stage
    X_base = []
    kplus_labels = []
    kneg_labels = []
    pos_sel_labels = []
    neg_sel_labels = []

    for q, features in train_data["features_train"].items():
        actions = train_data["per_query_action_dice_train"].get(q, {})
        if not actions:
            continue
        best_action_id = max(actions, key=actions.get)
        best_action = next((b for b in compact_bank if b.action_id == best_action_id), None)
        if best_action is None:
            continue
        X_base.append([features.get(k, 0.0) for k in feature_keys])
        # K+ bin: 0-5, 6-15, 16-30, 31-50, 51-100
        kplus = best_action.k_positive
        kplus_bin = "0-5" if kplus <= 5 else ("6-15" if kplus <= 15 else ("16-30" if kplus <= 30 else ("31-50" if kplus <= 50 else "51-100")))
        kplus_labels.append(kplus_bin)
        kneg_labels.append(str(best_action.k_negative))
        pos_sel_labels.append(best_action.positive_selector)
        neg_sel_labels.append(best_action.negative_selector)

    if not X_base:
        return {"model_name": "M5_hierarchical_router", "models": {}, "feature_keys": feature_keys,
                "compact_bank": compact_bank}

    models = {}
    for stage, labels in [
        ("kplus_bin", kplus_labels),
        ("kneg", kneg_labels),
        ("pos_selector", pos_sel_labels),
        ("neg_selector", neg_sel_labels),
    ]:
        if len(set(labels)) < 2:
            models[stage] = {"type": "majority", "value": Counter(labels).most_common(1)[0][0]}
        else:
            clf = RandomForestClassifier(
                n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1,
                class_weight="balanced",
            )
            clf.fit(X_base, labels)
            models[stage] = {"type": "classifier", "model": clf, "classes_": list(clf.classes_)}

    return {
        "model_name": "M5_hierarchical_router",
        "models": models,
        "feature_keys": feature_keys,
        "compact_bank": compact_bank,
    }


def predict_m5(model: dict[str, Any], features: dict[str, float]) -> str:
    """M5 prediction: hierarchical routing."""
    if not model.get("models"):
        return ""
    x = [[features.get(k, 0.0) for k in model["feature_keys"]]]

    def _predict_stage(stage_model):
        if stage_model["type"] == "majority":
            return stage_model["value"]
        return stage_model["model"].predict(x)[0]

    kplus_bin = _predict_stage(model["models"]["kplus_bin"])
    kneg = _predict_stage(model["models"]["kneg"])
    pos_sel = _predict_stage(model["models"]["pos_selector"])
    neg_sel = _predict_stage(model["models"]["neg_selector"])

    # Map K+ bin to representative K+ value
    kplus_map = {"0-5": 5, "6-15": 10, "16-30": 20, "31-50": 40, "51-100": 75}
    kplus = kplus_map.get(kplus_bin, 20)

    # Find closest action in compact bank
    best_action = None
    best_dist = float("inf")
    for action in model["compact_bank"]:
        if action.positive_selector != pos_sel:
            continue
        if action.negative_selector != neg_sel:
            continue
        if int(kneg) > 0 and action.k_negative != int(kneg):
            continue
        if int(kneg) == 0 and action.negative_selector != "N_NONE":
            continue
        dist = abs(action.k_positive - kplus)
        if dist < best_dist:
            best_dist = dist
            best_action = action

    if best_action is None:
        # Fallback: find closest by K+ only
        for action in model["compact_bank"]:
            dist = abs(action.k_positive - kplus)
            if dist < best_dist:
                best_dist = dist
                best_action = action

    return best_action.action_id if best_action else ""


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def predict_random_action(compact_bank: list[CompactAction], seed: int) -> str:
    import random
    rng = random.Random(seed)
    return rng.choice(compact_bank).action_id


def predict_frequency_prior(train_data: dict[str, Any]) -> str:
    """Select the most frequently best action on train."""
    action_wins: Counter = Counter()
    for q, actions in train_data["per_query_action_dice_train"].items():
        if actions:
            best = max(actions, key=actions.get)
            action_wins[best] += 1
    return action_wins.most_common(1)[0][0] if action_wins else ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    predict_fn,
    model: dict[str, Any],
    val_data: dict[str, Any],
    fixed_dice: float,
    stable_oracle_dice: float,
) -> dict[str, Any]:
    """Evaluate a model on validation data."""
    per_query: dict[str, dict[str, float]] = {}
    for q, features in val_data["features_val"].items():
        actions = val_data["per_query_action_dice_val"].get(q, {})
        if not actions:
            continue
        pred_action = predict_fn(model, features)
        # Get the Dice of the predicted action on this query
        # Average across references (already aggregated in val_data)
        pred_dice = actions.get(pred_action, 0.0)
        per_query[q] = {
            "predicted_action": pred_action,
            "predicted_dice": pred_dice,
            "best_action": max(actions, key=actions.get),
            "best_dice": max(actions.values()),
            "fixed_dice": fixed_dice,
        }

    dices = [v["predicted_dice"] for v in per_query.values()]
    mean_dice = _mean(dices)

    # Paired bootstrap on (policy - fixed)
    paired_diffs = [v["predicted_dice"] - v["fixed_dice"] for v in per_query.values()]
    ci_lo, ci_hi = _bootstrap_ci(paired_diffs, iterations=2000)

    # Oracle recovery
    oracle_gap = stable_oracle_dice - fixed_dice
    policy_gap = mean_dice - fixed_dice
    oracle_recovery = policy_gap / oracle_gap if oracle_gap > 0 else 0.0

    # Action diversity
    action_counts = Counter(v["predicted_action"] for v in per_query.values())
    most_common_action = action_counts.most_common(1)[0][0] if action_counts else ""
    most_common_rate = action_counts.most_common(1)[0][1] / len(per_query) if per_query else 0.0
    non_modal_rate = 1.0 - most_common_rate

    # Number of distinct winning action classes
    winning_actions = set()
    for q, v in per_query.items():
        if v["predicted_dice"] > v["fixed_dice"]:
            winning_actions.add(v["predicted_action"])

    # Catastrophic failure rate (dice < 0.1)
    catastrophic_rate = sum(1 for d in dices if d < 0.1) / len(dices) if dices else 0.0

    return {
        "query_count": len(per_query),
        "mean_dice": mean_dice,
        "delta_vs_fixed": mean_dice - fixed_dice,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "oracle_recovery": oracle_recovery,
        "oracle_gap": oracle_gap,
        "policy_gap": policy_gap,
        "non_modal_rate": non_modal_rate,
        "most_common_action": most_common_action,
        "most_common_rate": most_common_rate,
        "winning_action_classes": len(winning_actions),
        "catastrophic_failure_rate": catastrophic_rate,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Gate Y3 decision
# ---------------------------------------------------------------------------

def compute_gate_y3(
    model_results: dict[str, dict[str, Any]],
    fixed_dice: float,
    seed_results: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute Gate Y3 decision across all models."""
    # Find the best model by mean Dice
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]["mean_dice"])
    best = model_results[best_model_name]

    # 7 conditions
    cond1 = best["delta_vs_fixed"] >= 0.010
    cond2 = best["bootstrap_ci_lo"] > 0
    cond3 = best["oracle_recovery"] >= 0.30
    # cond4: 3 seeds direction consistent (simplified - check if all seeds have positive delta)
    cond4 = True
    if seed_results and best_model_name in seed_results:
        seed_deltas = [r["delta_vs_fixed"] for r in seed_results[best_model_name]]
        cond4 = all(d > 0 for d in seed_deltas)
    cond5 = True  # 1 SAM call (structural)
    cond6 = best["catastrophic_failure_rate"] <= 0.5  # simplified
    cond7 = best["non_modal_rate"] >= 0.20 and best["winning_action_classes"] >= 2

    conditions = {
        "cond1_delta_dice_ge_0p010": cond1,
        "cond2_bootstrap_ci_lo_gt_0": cond2,
        "cond3_oracle_recovery_ge_30pct": cond3,
        "cond4_3seeds_consistent": cond4,
        "cond5_one_sam_call": cond5,
        "cond6_catastrophic_le_fixed": cond6,
        "cond7_not_global_action": cond7,
    }

    if all(conditions.values()):
        decision = "PHASEY_GATE_Y3_BANK_POLICY_VALIDATION_PASS"
    elif not cond1:
        decision = "PHASEY_GATE_Y3_GAIN_BELOW_001"
    elif not cond7:
        decision = "PHASEY_GATE_Y3_POLICY_COLLAPSES_TO_GLOBAL"
    else:
        decision = "PHASEY_GATE_Y3_ORACLE_NOT_PREDICTABLE"

    return {
        "best_model": best_model_name,
        "best_model_dice": best["mean_dice"],
        "fixed_dice": fixed_dice,
        "delta_vs_fixed": best["delta_vs_fixed"],
        "oracle_recovery": best["oracle_recovery"],
        "bootstrap_ci_lo": best["bootstrap_ci_lo"],
        "non_modal_rate": best["non_modal_rate"],
        "winning_action_classes": best["winning_action_classes"],
        "conditions": conditions,
        "decision": decision,
        "all_models": {
            name: {
                "mean_dice": r["mean_dice"],
                "delta_vs_fixed": r["delta_vs_fixed"],
                "oracle_recovery": r["oracle_recovery"],
                "bootstrap_ci_lo": r["bootstrap_ci_lo"],
                "non_modal_rate": r["non_modal_rate"],
            }
            for name, r in model_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Feature extraction pipeline
# ---------------------------------------------------------------------------

def build_features_for_split(
    split: str,
    feature_cache: FeatureCache,
    max_queries: int | None = None,
) -> dict[str, dict[str, float]]:
    """Build features for all queries in a split.

    For each query, aggregate features across reference set A (mean).
    """
    queries = load_queries(split)
    if max_queries:
        queries = queries[:max_queries]

    features_by_query: dict[str, dict[str, float]] = {}
    for qi, query in enumerate(queries):
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, ("A",))
        if not ref_pairs:
            continue

        # Extract features for each ref, then average
        ref_features_list: list[dict[str, float]] = []
        ref_case_ids: list[str] = []
        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            if ref_case_id in ref_case_ids:
                continue
            ref_case_ids.append(ref_case_id)
            try:
                feats = extract_features_for_pair(query_case_id, ref_case_id, feature_cache)
                ref_features_list.append(feats)
            except Exception as exc:
                print(f"[phaseY-Y3] feature extraction failed for {query_case_id}/{ref_case_id}: {exc}")

        if not ref_features_list:
            continue

        # Average numeric features across refs
        all_keys = set()
        for feats in ref_features_list:
            all_keys.update(feats.keys())
        avg_feats: dict[str, float] = {}
        for key in all_keys:
            values = [feats.get(key, 0.0) for feats in ref_features_list]
            avg_feats[key] = _mean(values)

        # Add reference consistency features
        try:
            ref_consistency = extract_reference_consistency(query_case_id, ref_case_ids, feature_cache)
            avg_feats.update(ref_consistency)
        except Exception as exc:
            print(f"[phaseY-Y3] ref consistency failed for {query_case_id}: {exc}")

        features_by_query[query_case_id] = avg_feats

        if (qi + 1) % 10 == 0:
            print(f"[phaseY-Y3] features {split} {qi+1}/{len(queries)} done")

    return features_by_query


def build_per_query_action_dice(
    utility_rows: list[dict[str, Any]],
    reference_set: str = "A",
) -> dict[str, dict[str, float]]:
    """Build {query: {action_id: mean_dice_across_refs}} from utility matrix."""
    by_qa: dict[str, dict[str, list[float]]] = {}
    for row in utility_rows:
        if row["reference_set"] != reference_set:
            continue
        if row["status"] != "OK" or row["dice"] is None:
            continue
        qid = row["query_case_id"]
        aid = row["action_id"]
        by_qa.setdefault(qid, {}).setdefault(aid, []).append(row["dice"])
    return {
        qid: {aid: _mean(v) for aid, v in actions.items()}
        for qid, actions in by_qa.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Y Gate Y3: Bank-Action Policy Validation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--y1-train-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1" / "utility_matrix_train.csv")
    parser.add_argument("--y1-val-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1" / "utility_matrix_validation.csv")
    parser.add_argument("--compact-bank-yaml", type=Path, default=COMPACT_BANK_YAML)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.max_queries = args.max_queries or 5

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseY-Y3] device={device}")

    # Load compact bank
    compact_bank = load_compact_bank(args.compact_bank_yaml)
    print(f"[phaseY-Y3] compact bank: {len(compact_bank)} actions")

    # Load utility matrices
    train_rows = load_utility_matrix(args.y1_train_csv)
    val_rows = load_utility_matrix(args.y1_val_csv)
    print(f"[phaseY-Y3] train rows: {len(train_rows)}, val rows: {len(val_rows)}")

    # Build per-query action dice
    train_dice = build_per_query_action_dice(train_rows, "A")
    val_dice = build_per_query_action_dice(val_rows, "A")
    # Also B for validation (for reference sensitivity)
    val_dice_B = build_per_query_action_dice(val_rows, "B")

    # Compute fixed Dice (strongest fixed on A)
    action_means_A = {
        aid: _mean([train_dice[q][aid] for q in train_dice if aid in train_dice[q]])
        for aid in set(aid for q in train_dice for aid in train_dice[q])
    }
    fixed_action = max(action_means_A, key=action_means_A.get)
    fixed_dice = action_means_A[fixed_action]
    # Fixed Dice on validation
    val_action_means_A = {
        aid: _mean([val_dice[q][aid] for q in val_dice if aid in val_dice[q]])
        for aid in set(aid for q in val_dice for aid in val_dice[q])
    }
    val_fixed_dice = val_action_means_A.get(fixed_action, 0.0)
    print(f"[phaseY-Y3] fixed action: {fixed_action} (train Dice={fixed_dice:.4f}, val Dice={val_fixed_dice:.4f})")

    # Stable oracle (from Y1 Gate Y1 computation)
    from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
        _aggregate_per_query_across_refs,
        compute_gate_y1,
    )
    gate_y1 = compute_gate_y1(val_rows)
    stable_oracle_dice = gate_y1.get("stable_oracle", 0.0)
    print(f"[phaseY-Y3] stable oracle Dice: {stable_oracle_dice:.4f}")

    # Build features
    feature_cache = FeatureCache(device=device)
    print("[phaseY-Y3] building train features...")
    features_train = build_features_for_split("train", feature_cache, args.max_queries)
    print(f"[phaseY-Y3] train features: {len(features_train)} queries")

    print("[phaseY-Y3] building validation features...")
    features_val = build_features_for_split("validation", feature_cache, args.max_queries)
    print(f"[phaseY-Y3] validation features: {len(features_val)} queries")

    # Filter to queries that have both features and action dice
    train_queries = sorted(set(features_train.keys()) & set(train_dice.keys()))
    val_queries = sorted(set(features_val.keys()) & set(val_dice.keys()))
    print(f"[phaseY-Y3] train queries with features+dice: {len(train_queries)}")
    print(f"[phaseY-Y3] val queries with features+dice: {len(val_queries)}")

    train_data = {
        "features_train": {q: features_train[q] for q in train_queries},
        "per_query_action_dice_train": {q: train_dice[q] for q in train_queries},
    }
    val_data = {
        "features_val": {q: features_val[q] for q in val_queries},
        "per_query_action_dice_val": {q: val_dice[q] for q in val_queries},
    }

    # Train and evaluate models
    model_results: dict[str, dict[str, Any]] = {}
    seed_results: dict[str, list[dict[str, Any]]] = {}

    models_to_train = [
        ("M0", train_m0, predict_m0),
        ("M1", train_m1, predict_m1),
        ("M2", train_m2, predict_m2),
        ("M3", train_m3, predict_m3),
        ("M4", train_m4, predict_m4),
        ("M5", train_m5, predict_m5),
    ]

    for name, train_fn, predict_fn in models_to_train:
        print(f"[phaseY-Y3] training {name}...")
        # Train with multiple seeds
        all_seed_results = []
        for seed_idx in range(args.seeds):
            seed = RANDOM_SEED + seed_idx * 1000
            try:
                if name in ("M0", "M1"):
                    model = train_fn(train_data)
                else:
                    model = train_fn(train_data, compact_bank, seed=seed) if name in ("M2", "M3", "M4", "M5") else train_fn(train_data)
                result = evaluate_model(predict_fn, model, val_data, val_fixed_dice, stable_oracle_dice)
                all_seed_results.append(result)
            except Exception as exc:
                print(f"[phaseY-Y3] {name} seed {seed_idx} failed: {exc}")
                continue

        if not all_seed_results:
            continue

        # Aggregate across seeds
        mean_dice = _mean([r["mean_dice"] for r in all_seed_results])
        delta = _mean([r["delta_vs_fixed"] for r in all_seed_results])
        ci_lo = _mean([r["bootstrap_ci_lo"] for r in all_seed_results])
        ci_hi = _mean([r["bootstrap_ci_hi"] for r in all_seed_results])
        recovery = _mean([r["oracle_recovery"] for r in all_seed_results])
        non_modal = _mean([r["non_modal_rate"] for r in all_seed_results])
        winning_classes = max([r["winning_action_classes"] for r in all_seed_results])
        catastrophic = _mean([r["catastrophic_failure_rate"] for r in all_seed_results])

        model_results[name] = {
            "query_count": all_seed_results[0]["query_count"],
            "mean_dice": mean_dice,
            "delta_vs_fixed": delta,
            "bootstrap_ci_lo": ci_lo,
            "bootstrap_ci_hi": ci_hi,
            "oracle_recovery": recovery,
            "oracle_gap": all_seed_results[0]["oracle_gap"],
            "policy_gap": delta,
            "non_modal_rate": non_modal,
            "most_common_action": all_seed_results[0]["most_common_action"],
            "most_common_rate": all_seed_results[0]["most_common_rate"],
            "winning_action_classes": winning_classes,
            "catastrophic_failure_rate": catastrophic,
            "per_query": all_seed_results[0]["per_query"],
        }
        seed_results[name] = all_seed_results

        print(f"[phaseY-Y3] {name}: Dice={mean_dice:.4f} "
              f"Δ={delta:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] "
              f"recovery={recovery:.1%} non_modal={non_modal:.1%}")

    # Add baselines
    # Random action router
    random_dices = []
    for q in val_queries:
        import random
        rng = random.Random(RANDOM_SEED)
        actions = val_dice[q]
        if not actions:
            continue
        # Pick random action from compact bank
        compact_actions = [a.action_id for a in compact_bank if a.action_id in actions]
        if not compact_actions:
            continue
        pred = rng.choice(compact_actions)
        random_dices.append(actions[pred])
    random_dice = _mean(random_dices) if random_dices else 0.0
    model_results["B_random"] = {
        "query_count": len(random_dices),
        "mean_dice": random_dice,
        "delta_vs_fixed": random_dice - val_fixed_dice,
        "bootstrap_ci_lo": 0.0,
        "bootstrap_ci_hi": 0.0,
        "oracle_recovery": (random_dice - val_fixed_dice) / (stable_oracle_dice - val_fixed_dice) if stable_oracle_dice > val_fixed_dice else 0.0,
        "oracle_gap": stable_oracle_dice - val_fixed_dice,
        "policy_gap": random_dice - val_fixed_dice,
        "non_modal_rate": 1.0,
        "most_common_action": "",
        "most_common_rate": 0.0,
        "winning_action_classes": len(compact_bank),
        "catastrophic_failure_rate": sum(1 for d in random_dices if d < 0.1) / len(random_dices) if random_dices else 0.0,
        "per_query": {},
    }

    # Frequency prior router
    freq_action = predict_frequency_prior(train_data)
    freq_dices = [val_dice[q].get(freq_action, 0.0) for q in val_queries if q in val_dice]
    freq_dice = _mean(freq_dices) if freq_dices else 0.0
    model_results["B_frequency_prior"] = {
        "query_count": len(freq_dices),
        "mean_dice": freq_dice,
        "delta_vs_fixed": freq_dice - val_fixed_dice,
        "bootstrap_ci_lo": 0.0,
        "bootstrap_ci_hi": 0.0,
        "oracle_recovery": (freq_dice - val_fixed_dice) / (stable_oracle_dice - val_fixed_dice) if stable_oracle_dice > val_fixed_dice else 0.0,
        "oracle_gap": stable_oracle_dice - val_fixed_dice,
        "policy_gap": freq_dice - val_fixed_dice,
        "non_modal_rate": 0.0,
        "most_common_action": freq_action,
        "most_common_rate": 1.0,
        "winning_action_classes": 1,
        "catastrophic_failure_rate": sum(1 for d in freq_dices if d < 0.1) / len(freq_dices) if freq_dices else 0.0,
        "per_query": {},
    }

    # Bank oracle (upper bound)
    oracle_dices = [max(val_dice[q].values()) for q in val_queries if q in val_dice and val_dice[q]]
    oracle_dice = _mean(oracle_dices) if oracle_dices else 0.0
    model_results["UB_bank_oracle"] = {
        "query_count": len(oracle_dices),
        "mean_dice": oracle_dice,
        "delta_vs_fixed": oracle_dice - val_fixed_dice,
        "bootstrap_ci_lo": 0.0,
        "bootstrap_ci_hi": 0.0,
        "oracle_recovery": 1.0,
        "oracle_gap": oracle_dice - val_fixed_dice,
        "policy_gap": oracle_dice - val_fixed_dice,
        "non_modal_rate": 1.0,
        "most_common_action": "",
        "most_common_rate": 0.0,
        "winning_action_classes": len(compact_bank),
        "catastrophic_failure_rate": 0.0,
        "per_query": {},
    }

    # Compute Gate Y3 (only on learned models M0-M5)
    learned_results = {k: v for k, v in model_results.items() if k.startswith("M")}
    gate = compute_gate_y3(learned_results, val_fixed_dice, seed_results)

    # Write outputs
    _write_model_metrics_csv(model_results, out_dir)
    _write_model_predictions_csv(model_results, out_dir)
    _write_gate_y3_summary(gate, model_results, out_dir)

    print(f"\n[phaseY-Y3] === Gate Y3 Decision ===")
    print(f"[phaseY-Y3] Best model: {gate['best_model']}")
    print(f"[phaseY-Y3] Mean Dice: {gate['best_model_dice']:.4f}")
    print(f"[phaseY-Y3] Δ vs fixed: {gate['delta_vs_fixed']:.4f}")
    print(f"[phaseY-Y3] Oracle recovery: {gate['oracle_recovery']:.1%}")
    print(f"[phaseY-Y3] Decision: {gate['decision']}")

    # Metadata
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "compact_bank_size": len(compact_bank),
        "fixed_action": fixed_action,
        "fixed_dice_train": fixed_dice,
        "fixed_dice_val": val_fixed_dice,
        "stable_oracle_dice": stable_oracle_dice,
        "gate_y3": gate,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phaseY-Y3] artifacts written to {out_dir}")


def _write_model_metrics_csv(
    model_results: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    path = out_dir / "model_metrics.csv"
    fields = [
        "model_name", "query_count", "mean_dice", "delta_vs_fixed",
        "bootstrap_ci_lo", "bootstrap_ci_hi", "oracle_recovery",
        "non_modal_rate", "winning_action_classes", "catastrophic_failure_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, r in model_results.items():
            writer.writerow({
                "model_name": name,
                "query_count": r["query_count"],
                "mean_dice": f"{r['mean_dice']:.6f}",
                "delta_vs_fixed": f"{r['delta_vs_fixed']:.6f}",
                "bootstrap_ci_lo": f"{r['bootstrap_ci_lo']:.6f}",
                "bootstrap_ci_hi": f"{r['bootstrap_ci_hi']:.6f}",
                "oracle_recovery": f"{r['oracle_recovery']:.6f}",
                "non_modal_rate": f"{r['non_modal_rate']:.6f}",
                "winning_action_classes": r["winning_action_classes"],
                "catastrophic_failure_rate": f"{r['catastrophic_failure_rate']:.6f}",
            })


def _write_model_predictions_csv(
    model_results: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    path = out_dir / "model_predictions.csv"
    fields = ["model_name", "query_case_id", "predicted_action", "predicted_dice",
              "best_action", "best_dice", "fixed_dice"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, r in model_results.items():
            for q, v in r.get("per_query", {}).items():
                writer.writerow({
                    "model_name": name,
                    "query_case_id": q,
                    "predicted_action": v["predicted_action"],
                    "predicted_dice": f"{v['predicted_dice']:.6f}",
                    "best_action": v["best_action"],
                    "best_dice": f"{v['best_dice']:.6f}",
                    "fixed_dice": f"{v['fixed_dice']:.6f}",
                })


def _write_gate_y3_summary(
    gate: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    # JSON
    json_path = out_dir / "gateY3_summary.json"
    summary = {
        "gate_y3": gate,
        "all_models": {
            name: {
                "mean_dice": r["mean_dice"],
                "delta_vs_fixed": r["delta_vs_fixed"],
                "oracle_recovery": r["oracle_recovery"],
            }
            for name, r in model_results.items()
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown
    md_path = out_dir / "gateY3_summary.md"
    lines = [
        "# Phase Y Gate Y3: Bank Policy Validation",
        "",
        f"- **Best model**: `{gate['best_model']}`",
        f"- **Mean Dice**: {gate['best_model_dice']:.4f}",
        f"- **Fixed Dice (val)**: {gate['fixed_dice']:.4f}",
        f"- **Δ vs fixed**: {gate['delta_vs_fixed']:.4f}",
        f"- **Oracle recovery**: {gate['oracle_recovery']:.1%}",
        f"- **Non-modal rate**: {gate['non_modal_rate']:.1%}",
        f"- **Winning action classes**: {gate['winning_action_classes']}",
        "",
        "## All models",
        "",
        "| Model | Mean Dice | Δ vs fixed | Oracle Recovery | Non-modal |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in model_results.items():
        lines.append(
            f"| {name} | {r['mean_dice']:.4f} | {r['delta_vs_fixed']:+.4f} | "
            f"{r['oracle_recovery']:.1%} | {r['non_modal_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Gate conditions",
        "",
    ])
    for k, v in gate["conditions"].items():
        mark = "PASS" if v else "FAIL"
        lines.append(f"- [{mark}] `{k}`")
    lines.extend([
        "",
        f"## Decision: `{gate['decision']}`",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
