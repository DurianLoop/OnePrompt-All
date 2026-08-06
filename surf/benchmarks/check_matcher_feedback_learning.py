from __future__ import annotations

import numpy as np

from matcher_feedback_learning import (
    FeedbackPoint,
    apply_feature_feedback,
    learn_scalar_threshold,
    select_threshold_by_budget,
    score_stage3_proposal,
)


def normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1e-8)


def test_feature_feedback_uses_positive_and_negative_prototypes() -> None:
    reference = normalize(np.asarray([[1.0, 0.0]], dtype=np.float32))
    positive = normalize(np.asarray([[0.75, 0.35]], dtype=np.float32))
    negative = normalize(np.asarray([[0.0, 1.0]], dtype=np.float32))
    target = normalize(
        np.asarray(
            [
                [1.0, 0.0],
                [0.75, 0.35],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    before = (reference @ target.T).reshape(-1)
    after = apply_feature_feedback(
        reference_proto=reference,
        target_features=target,
        positive_proto=positive,
        negative_proto=negative,
        positive_weight=0.35,
        negative_weight=0.45,
    )
    assert after[1] - after[2] > before[1] - before[2], "feedback should improve positive-vs-background margin"
    assert after[2] < before[2], "negative feedback should suppress background-like features"


def test_select_threshold_keeps_candidate_count_inside_budget() -> None:
    scores = np.linspace(0.1, 0.9, 100, dtype=np.float32)
    threshold = select_threshold_by_budget(scores, min_selected=10, max_selected=20, floor=0.2, ceiling=0.95)
    kept = int(np.count_nonzero(scores >= threshold))
    assert 10 <= kept <= 20


def test_learn_scalar_threshold_separates_labeled_proposals() -> None:
    values = [0.88, 0.81, 0.42, 0.35]
    labels = [1, 1, 0, 0]
    result = learn_scalar_threshold(values, labels)
    assert 0.42 < result.threshold <= 0.81
    assert result.f1 == 1.0


def test_score_stage3_proposal_penalizes_large_background_masks() -> None:
    good = {
        "instance_similarity": 0.82,
        "sam_score": 0.93,
        "cluster_coverage": 0.8,
        "prompt_coverage": 1.0,
        "mask_area_pct": 12.0,
        "bbox_area_pct": 16.0,
    }
    too_large = dict(good)
    too_large["mask_area_pct"] = 70.0
    too_large["bbox_area_pct"] = 85.0
    assert score_stage3_proposal(good) > score_stage3_proposal(too_large)


def test_feedback_point_parser_contract() -> None:
    point = FeedbackPoint(x=12.5, y=32.0, label=0)
    assert point.as_dict() == {"x": 12.5, "y": 32.0, "label": 0}


def main() -> int:
    tests = [
        test_feature_feedback_uses_positive_and_negative_prototypes,
        test_select_threshold_keeps_candidate_count_inside_budget,
        test_learn_scalar_threshold_separates_labeled_proposals,
        test_score_stage3_proposal_penalizes_large_background_masks,
        test_feedback_point_parser_contract,
    ]
    for test in tests:
        test()
    print("Matcher feedback learning checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
