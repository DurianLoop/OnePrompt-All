from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class FeedbackPoint:
    x: float
    y: float
    label: int = 1

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdLearningResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    positive_count: int
    negative_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def apply_feature_feedback(
    *,
    reference_proto: np.ndarray,
    target_features: np.ndarray,
    positive_proto: np.ndarray | None = None,
    negative_proto: np.ndarray | None = None,
    positive_weight: float = 0.35,
    negative_weight: float = 0.25,
) -> np.ndarray:
    """Update a Matcher reference prototype with positive/negative feedback."""
    updated = normalize_rows(reference_proto)[0]
    if positive_proto is not None:
        updated = updated + float(positive_weight) * normalize_rows(positive_proto)[0]
    if negative_proto is not None:
        updated = updated - float(negative_weight) * normalize_rows(negative_proto)[0]
    updated = normalize_rows(updated)[0]
    features = normalize_rows(target_features)
    return (features @ updated).astype(np.float32)


def select_threshold_by_budget(
    scores: Iterable[float],
    *,
    min_selected: int,
    max_selected: int,
    floor: float = -1.0,
    ceiling: float = 1.0,
) -> float:
    """Choose a score threshold that keeps the selected count inside a target budget."""
    values = np.asarray(list(scores), dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(floor)
    if min_selected <= 0 or max_selected <= 0 or min_selected > max_selected:
        raise ValueError("Expected 0 < min_selected <= max_selected")

    desc = np.sort(values)[::-1]
    max_index = min(max_selected, desc.size) - 1
    threshold = float(desc[max_index])
    threshold = min(max(threshold, float(floor)), float(ceiling))

    kept = int(np.count_nonzero(values >= threshold))
    if kept < min_selected:
        min_index = min(min_selected, desc.size) - 1
        threshold = float(desc[min_index])
    elif kept > max_selected:
        threshold = float(desc[max_index])
    return min(max(float(threshold), float(floor)), float(ceiling))


def f1_at_threshold(values: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    pred = values >= float(threshold)
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(values), 1)
    return float(f1), float(precision), float(recall), float(accuracy)


def learn_scalar_threshold(values: Iterable[float], labels: Iterable[int]) -> ThresholdLearningResult:
    """Learn one interpretable keep/reject threshold from labeled proposals."""
    score_values = np.asarray(list(values), dtype=np.float32)
    label_values = np.asarray(list(labels), dtype=np.int32)
    if score_values.size != label_values.size or score_values.size == 0:
        raise ValueError("values and labels must be non-empty and have the same length")
    positives = int(np.count_nonzero(label_values == 1))
    negatives = int(np.count_nonzero(label_values == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("threshold learning needs at least one positive and one negative label")

    unique = np.unique(score_values)
    candidates: list[float] = [float(unique[0] - 1e-6), float(unique[-1] + 1e-6)]
    if unique.size == 1:
        candidates.append(float(unique[0]))
    else:
        candidates.extend(float(v) for v in unique)
        mids = (unique[:-1] + unique[1:]) / 2.0
        candidates.extend(float(v) for v in mids)

    best: tuple[float, float, float, float, float] | None = None
    for threshold in candidates:
        f1, precision, recall, accuracy = f1_at_threshold(score_values, label_values, threshold)
        candidate = (f1, accuracy, precision, recall, float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    f1, accuracy, precision, recall, threshold = best
    return ThresholdLearningResult(
        threshold=round(float(threshold), 6),
        f1=round(float(f1), 6),
        precision=round(float(precision), 6),
        recall=round(float(recall), 6),
        accuracy=round(float(accuracy), 6),
        positive_count=positives,
        negative_count=negatives,
    )


def score_stage3_proposal(
    proposal: dict[str, Any],
    *,
    target_area_pct: float = 30.0,
    target_bbox_area_pct: float = 40.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Score one proposal with interpretable Matcher/SAM/shape features."""
    w = {
        "instance_similarity": 1.0,
        "sam_score": 0.1,
        "cluster_coverage": 0.1,
        "prompt_coverage": 0.05,
        "area_penalty": 0.15,
        "bbox_penalty": 0.15,
    }
    if weights:
        w.update(weights)
    area = float(proposal.get("mask_area_pct", 0.0))
    bbox = float(proposal.get("bbox_area_pct", 0.0))
    area_penalty = max(0.0, area - float(target_area_pct)) / max(float(target_area_pct), 1e-6)
    bbox_penalty = max(0.0, bbox - float(target_bbox_area_pct)) / max(float(target_bbox_area_pct), 1e-6)
    score = (
        w["instance_similarity"] * float(proposal.get("instance_similarity", 0.0))
        + w["sam_score"] * float(proposal.get("sam_score", 0.0))
        + w["cluster_coverage"] * float(proposal.get("cluster_coverage", 0.0))
        + w["prompt_coverage"] * float(proposal.get("prompt_coverage", 0.0))
        - w["area_penalty"] * area_penalty
        - w["bbox_penalty"] * bbox_penalty
    )
    return round(float(score), 6)


def labels_for_proposals(proposals: list[dict[str, Any]], label_payload: dict[str, Any]) -> list[int]:
    positives = {int(value) for value in label_payload.get("positive_group_ids", [])}
    negatives = {int(value) for value in label_payload.get("negative_group_ids", [])}
    labels: list[int] = []
    for proposal in proposals:
        group_id = int(proposal.get("group_id", -1))
        if group_id in positives:
            labels.append(1)
        elif group_id in negatives:
            labels.append(0)
    return labels


def labeled_values(
    proposals: list[dict[str, Any]],
    label_payload: dict[str, Any],
    value_key: str,
) -> tuple[list[float], list[int], list[int]]:
    positives = {int(value) for value in label_payload.get("positive_group_ids", [])}
    negatives = {int(value) for value in label_payload.get("negative_group_ids", [])}
    values: list[float] = []
    labels: list[int] = []
    group_ids: list[int] = []
    for proposal in proposals:
        group_id = int(proposal.get("group_id", -1))
        if group_id in positives or group_id in negatives:
            values.append(float(proposal.get(value_key, 0.0)))
            labels.append(1 if group_id in positives else 0)
            group_ids.append(group_id)
    return values, labels, group_ids


def learn_best_scalar_policy(
    rows: list[dict[str, Any]],
    *,
    feature_keys: list[str],
    label_key: str = "label",
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    policies: list[dict[str, Any]] = []
    for feature in feature_keys:
        values = [float(row.get(feature, 0.0)) for row in rows]
        labels = [int(row.get(label_key, 0)) for row in rows]
        result = learn_scalar_threshold(values, labels)
        policy = {"feature": feature, **result.as_dict()}
        policies.append(policy)
        candidate = (float(policy["f1"]), float(policy["accuracy"]), float(policy["precision"]), float(policy["recall"]))
        if best is None or candidate > (
            float(best["f1"]),
            float(best["accuracy"]),
            float(best["precision"]),
            float(best["recall"]),
        ):
            best = policy
    if best is None:
        raise ValueError("No feature policies could be learned")
    return {"best_policy": best, "policies": policies}


def apply_scalar_policy(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    feature = str(policy["feature"])
    threshold = float(policy["threshold"])
    applied: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["policy_score"] = float(item.get(feature, 0.0))
        item["policy_keep"] = item["policy_score"] >= threshold
        applied.append(item)
    return applied


def classification_metrics(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, float | int]:
    truth = np.asarray(list(labels), dtype=np.int32).astype(bool)
    pred = np.asarray(list(predictions), dtype=np.int32).astype(bool)
    if truth.size != pred.size:
        raise ValueError("labels and predictions must have the same length")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(truth.size, 1)
    return {
        "n": int(truth.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "accuracy": round(float(accuracy), 6),
    }


def rows_to_matrix(rows: list[dict[str, Any]], feature_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(key, 0.0)) for key in feature_keys] for row in rows], dtype=np.float32)
    y = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.float32)
    return x, y


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean.reshape(-1), std.reshape(-1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    steps: int = 800,
    l2: float = 0.01,
) -> dict[str, Any]:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x=(n,d), y=(n,)")
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    bias = np.float32(0.0)
    positive = max(float(y.sum()), 1.0)
    negative = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y > 0.5, negative / positive, 1.0).astype(np.float32)
    for _ in range(int(steps)):
        logits = x @ weights + bias
        probs = sigmoid(logits)
        error = (probs - y) * sample_weights
        grad_w = (x.T @ error) / max(float(len(y)), 1.0) + float(l2) * weights
        grad_b = error.mean()
        weights -= float(lr) * grad_w.astype(np.float32)
        bias -= np.float32(float(lr) * float(grad_b))
    return {"weights": weights, "bias": float(bias)}


def predict_logistic(model: dict[str, Any], x: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    probs = sigmoid(x @ np.asarray(model["weights"], dtype=np.float32) + float(model["bias"]))
    return probs, (probs >= float(threshold)).astype(np.int32)


def train_decision_stump(rows: list[dict[str, Any]], feature_keys: list[str]) -> dict[str, Any]:
    labels = np.asarray([int(row.get("label", 0)) for row in rows], dtype=np.int32)
    best: dict[str, Any] | None = None
    for feature in feature_keys:
        values = np.asarray([float(row.get(feature, 0.0)) for row in rows], dtype=np.float32)
        unique = np.unique(values)
        candidates = list(unique)
        if unique.size > 1:
            candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
        for threshold in candidates:
            for direction in ("ge", "lt"):
                pred = values >= float(threshold) if direction == "ge" else values < float(threshold)
                metrics = classification_metrics(labels, pred.astype(np.int32))
                candidate = (
                    float(metrics["f1"]),
                    float(metrics["accuracy"]),
                    float(metrics["precision"]),
                    float(metrics["recall"]),
                )
                if best is None or candidate > best["rank"]:
                    best = {
                        "feature": feature,
                        "threshold": round(float(threshold), 6),
                        "direction": direction,
                        "rank": candidate,
                        "train_metrics": metrics,
                    }
    if best is None:
        raise ValueError("No decision stump could be trained")
    best.pop("rank")
    return best


def predict_decision_stump(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[int]:
    feature = str(model["feature"])
    threshold = float(model["threshold"])
    direction = str(model.get("direction", "ge"))
    preds: list[int] = []
    for row in rows:
        value = float(row.get(feature, 0.0))
        keep = value >= threshold if direction == "ge" else value < threshold
        preds.append(1 if keep else 0)
    return preds
