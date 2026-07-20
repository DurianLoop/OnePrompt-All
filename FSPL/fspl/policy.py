from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from fspl.candidates import CandidatePools, contrast_map
from fspl.protocol import FixedRuleConfig


def _quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def _top_margin(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - scores[1])


def _mean_nearest_distance_patches(pools: CandidatePools, patch_size: float = 14.0) -> float:
    if len(pools.positives) < 2:
        return 0.0
    coordinates = np.asarray([candidate.coordinate for candidate in pools.positives], dtype=np.float32)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2) / patch_size
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).mean())


def build_action_features(
    foreground_scores: np.ndarray,
    background_scores: np.ndarray,
    pools: CandidatePools,
    rule: FixedRuleConfig,
) -> dict[str, float]:
    foreground = np.asarray(foreground_scores, dtype=np.float32)
    background = np.asarray(background_scores, dtype=np.float32)
    contrast = contrast_map(foreground, background, rule.lambda_bg)
    positive_scores = [candidate.score for candidate in pools.positives]
    negative_scores = [candidate.score for candidate in pools.negatives]
    return {
        "contrast_mean": float(contrast.mean()),
        "contrast_std": float(contrast.std()),
        "contrast_p10": _quantile(contrast, 0.10),
        "contrast_p50": _quantile(contrast, 0.50),
        "contrast_p90": _quantile(contrast, 0.90),
        "foreground_mean": float(foreground.mean()),
        "background_mean": float(background.mean()),
        "foreground_background_correlation": _correlation(foreground, background),
        "positive_score_mean": float(np.mean(positive_scores)) if positive_scores else 0.0,
        "positive_score_margin": _top_margin(positive_scores),
        "negative_score_mean": float(np.mean(negative_scores)) if negative_scores else 0.0,
        "negative_score_margin": _top_margin(negative_scores),
        "action_positive_count": float(rule.positive_limit),
        "action_negative_count": float(rule.negative_limit),
        "action_lambda_bg": float(rule.lambda_bg),
        "positive_layout_is_spread": float(rule.positive_layout == "spread"),
        "positive_mean_nearest_distance_patches": _mean_nearest_distance_patches(pools),
    }


def choose_actions(action_predictions: Sequence[Mapping[str, float | int | str]]) -> dict[str, str]:
    grouped: dict[str, list[Mapping[str, float | int | str]]] = defaultdict(list)
    for prediction in action_predictions:
        grouped[str(prediction["case_id"])].append(prediction)
    selected: dict[str, str] = {}
    for case_id, candidates in grouped.items():
        winner = min(
            candidates,
            key=lambda row: (
                -float(row["predicted_utility"]),
                int(row["prompt_count"]),
                str(row["rule_name"]),
            ),
        )
        selected[case_id] = str(winner["rule_name"])
    return selected


@dataclass(frozen=True)
class PolicyFit:
    estimator: ExtraTreesRegressor
    min_samples_leaf: int
    validation_selected_mean_utility: float
    regressor_name: str = "ExtraTreesRegressor"


def _matrix(rows: Sequence[Mapping[str, float]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def fit_utility_policy(
    train_rows: Sequence[Mapping[str, float | int | str]],
    validation_rows: Sequence[Mapping[str, float | int | str]],
    *,
    feature_names: Sequence[str],
    min_samples_leaf_options: Sequence[int] = (1, 2, 3, 5),
) -> PolicyFit:
    if not train_rows or not validation_rows:
        raise ValueError("train and validation utility rows are both required")
    train_features = _matrix(train_rows, feature_names)
    train_target = np.asarray([float(row["utility"]) for row in train_rows], dtype=np.float64)
    validation_features = _matrix(validation_rows, feature_names)
    best: tuple[float, int, ExtraTreesRegressor] | None = None
    for min_samples_leaf in min_samples_leaf_options:
        estimator = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=min_samples_leaf,
            max_features=1.0,
            random_state=20260716,
            n_jobs=1,
        )
        estimator.fit(train_features, train_target)
        predictions = estimator.predict(validation_features)
        choices = choose_actions(
            [
                {
                    "case_id": row["case_id"],
                    "rule_name": row["rule_name"],
                    "predicted_utility": float(prediction),
                    "prompt_count": row["prompt_count"],
                }
                for row, prediction in zip(validation_rows, predictions, strict=True)
            ]
        )
        selected_utility = [
            float(row["utility"])
            for row in validation_rows
            if choices[str(row["case_id"])] == str(row["rule_name"])
        ]
        score = sum(selected_utility) / len(selected_utility)
        candidate = (score, int(min_samples_leaf), estimator)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    assert best is not None
    return PolicyFit(
        estimator=best[2],
        min_samples_leaf=best[1],
        validation_selected_mean_utility=best[0],
    )


def fit_frozen_utility_policy(
    rows: Sequence[Mapping[str, float | int | str]], *, feature_names: Sequence[str], min_samples_leaf: int
) -> ExtraTreesRegressor:
    if not rows:
        raise ValueError("at least one utility row is required")
    estimator = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features=1.0,
        random_state=20260716,
        n_jobs=1,
    )
    estimator.fit(_matrix(rows, feature_names), np.asarray([float(row["utility"]) for row in rows], dtype=np.float64))
    return estimator
