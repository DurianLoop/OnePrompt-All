from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from fspl.candidates import build_candidate_pools
from fspl.dataset import CaseRecord, discover_cases
from fspl.models import (
    IMAGE_SIZE,
    PATCH_SIZE,
    load_binary_mask,
    load_dinov2,
    load_mask_for_features,
    load_rgb,
    load_sam_predictor,
    matching_maps_from_prototypes,
    predict_sam,
    reference_prototypes,
)
from fspl.policy import build_action_features, choose_actions, fit_utility_policy
from fspl.protocol import FixedRuleConfig, build_fixed_baseline_protocol
from fspl.stage1 import _dice, _scale_points, image_resample
from fspl.stage2 import DEFAULT_FIXED_RULES, FixedBaselineConfig, aggregate_rule_metrics, protocol_as_dict


POLICY_FEATURE_NAMES = (
    "contrast_mean",
    "contrast_std",
    "contrast_p10",
    "contrast_p50",
    "contrast_p90",
    "foreground_mean",
    "background_mean",
    "foreground_background_correlation",
    "positive_score_mean",
    "positive_score_margin",
    "negative_score_mean",
    "negative_score_margin",
    "action_positive_count",
    "action_negative_count",
    "action_lambda_bg",
    "positive_layout_is_spread",
    "positive_mean_nearest_distance_patches",
)


@dataclass(frozen=True)
class UtilityPolicyConfig(FixedBaselineConfig):
    output_dir: Path = Path("artifacts/stage3_utility_policy")


def _evaluate_case_actions(
    case: CaseRecord,
    split: str,
    rules: Sequence[FixedRuleConfig],
    *,
    matcher: torch.nn.Module,
    foreground_prototype: torch.Tensor,
    background_prototype: torch.Tensor,
    predictor: object,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    query_image = load_rgb(case.image_path)
    query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
    foreground_scores, background_scores = matching_maps_from_prototypes(
        matcher, foreground_prototype, background_prototype, query_resized, device
    )
    target = load_binary_mask(case.mask_path, size=query_image.size)
    rows: list[dict[str, float | int | str]] = []
    for rule in rules:
        pools = build_candidate_pools(
            foreground_scores,
            background_scores,
            patch_size=PATCH_SIZE,
            positive_limit=rule.positive_limit,
            negative_limit=rule.negative_limit,
            lambda_bg=rule.lambda_bg,
        )
        positive_points, negative_points = _scale_points(pools, query_image.size)
        prediction, sam_score = predict_sam(predictor, query_image, positive_points, negative_points)
        features = build_action_features(foreground_scores, background_scores, pools, rule)
        rows.append(
            {
                "case_id": case.case_id,
                "split": split,
                "rule_name": rule.name,
                "utility": _dice(prediction, target),
                "sam_score": sam_score,
                "positive_count": len(positive_points),
                "negative_count": len(negative_points),
                "prompt_count": len(positive_points) + len(negative_points),
                **features,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "case_id",
        "split",
        "rule_name",
        "utility",
        "sam_score",
        "positive_count",
        "negative_count",
        "prompt_count",
        *POLICY_FEATURE_NAMES,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("case_id", "split", "rule_name", "utility", "predicted_utility", "prompt_count")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rule_metrics(rows: Sequence[dict[str, float | int | str]], split: str) -> dict[str, dict[str, float | int]]:
    from fspl.stage2 import BaselineRow

    return aggregate_rule_metrics(
        [
            BaselineRow(
                case_id=str(row["case_id"]),
                split=str(row["split"]),
                rule_name=str(row["rule_name"]),
                dice=float(row["utility"]),
                sam_score=float(row["sam_score"]),
                positive_count=int(row["positive_count"]),
                negative_count=int(row["negative_count"]),
            )
            for row in rows
        ],
        split=split,
    )


def _policy_outcome(
    rows: Sequence[dict[str, float | int | str]], estimator: object
) -> tuple[list[dict[str, float | int | str]], float]:
    import numpy as np

    matrix = np.asarray([[float(row[name]) for name in POLICY_FEATURE_NAMES] for row in rows], dtype=np.float64)
    predictions = estimator.predict(matrix)
    prediction_rows = [
        {
            "case_id": row["case_id"],
            "split": row["split"],
            "rule_name": row["rule_name"],
            "utility": row["utility"],
            "predicted_utility": float(prediction),
            "prompt_count": row["prompt_count"],
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    choices = choose_actions(prediction_rows)
    selected = [row for row in prediction_rows if choices[str(row["case_id"])] == str(row["rule_name"])]
    return selected, sum(float(row["utility"]) for row in selected) / len(selected)


def run_utility_policy(
    config: UtilityPolicyConfig, rules: Sequence[FixedRuleConfig] = DEFAULT_FIXED_RULES
) -> dict:
    if not rules:
        raise ValueError("at least one fixed rule is required")
    if not config.dino_checkpoint.is_file() or not config.sam_checkpoint.is_file():
        raise FileNotFoundError("Stage 3 requires both DINOv2-L and SAM ViT-B checkpoints")

    train_cases = discover_cases(config.dataset_root, "train")
    protocol = build_fixed_baseline_protocol(
        train_cases,
        discover_cases(config.dataset_root, "test"),
        seed=config.seed,
        validation_count=config.validation_count,
    )
    validation_ids = {case.case_id for case in protocol.validation_cases}
    policy_train_cases = tuple(
        case
        for case in train_cases
        if case.case_id != protocol.reference.case_id and case.case_id not in validation_ids
    )
    if not policy_train_cases:
        raise ValueError("policy train partition is empty")

    started = time.perf_counter()
    device = torch.device(config.device)
    reference_image = load_rgb(protocol.reference.image_path).resize(
        (IMAGE_SIZE, IMAGE_SIZE), image_resample()
    )
    reference_mask = load_mask_for_features(protocol.reference.mask_path)
    matcher = load_dinov2(config.dino_checkpoint, device)
    foreground_prototype, background_prototype = reference_prototypes(
        matcher, reference_image, reference_mask, device
    )
    predictor = load_sam_predictor(config.sam_checkpoint, device)

    rows: list[dict[str, float | int | str]] = []
    for split, cases in (
        ("train", policy_train_cases),
        ("validation", protocol.validation_cases),
        ("test", protocol.test_cases),
    ):
        for case in cases:
            rows.extend(
                _evaluate_case_actions(
                    case,
                    split,
                    rules,
                    matcher=matcher,
                    foreground_prototype=foreground_prototype,
                    background_prototype=background_prototype,
                    predictor=predictor,
                    device=device,
                )
            )

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    fit = fit_utility_policy(train_rows, validation_rows, feature_names=POLICY_FEATURE_NAMES)
    validation_selected, validation_policy_dice = _policy_outcome(validation_rows, fit.estimator)
    test_selected, test_policy_dice = _policy_outcome(test_rows, fit.estimator)

    output_dir = config.output_dir
    _write_csv(output_dir / "case_action_utilities.csv", rows)
    _write_predictions_csv(output_dir / "test_policy_predictions.csv", test_selected)
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "protocol": {**protocol_as_dict(protocol), "policy_train_case_ids": [case.case_id for case in policy_train_cases]},
        "rules": [asdict(rule) for rule in rules],
        "feature_names": list(POLICY_FEATURE_NAMES),
        "fit": {
            "regressor": fit.regressor_name,
            "selected_min_samples_leaf_on_validation": fit.min_samples_leaf,
            "validation_selected_mean_utility": fit.validation_selected_mean_utility,
        },
        "validation": {"fixed_rules": _rule_metrics(rows, "validation"), "policy_selected_mean_dice": validation_policy_dice},
        "official_test": {"fixed_rules": _rule_metrics(rows, "test"), "policy_selected_mean_dice": test_policy_dice},
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
