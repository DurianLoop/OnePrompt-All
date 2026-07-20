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
from fspl.multiref import MultiReferenceProtocol, build_multireference_protocol, episode_id
from fspl.policy import build_action_features, fit_utility_policy
from fspl.protocol import FixedRuleConfig
from fspl.stage1 import _dice, _scale_points, image_resample
from fspl.stage2 import DEFAULT_FIXED_RULES, BaselineRow, aggregate_rule_metrics
from fspl.stage3 import POLICY_FEATURE_NAMES, _policy_outcome


SPATIAL_LAYOUT_RULES = (
    FixedRuleConfig("top5_n3", positive_limit=5, negative_limit=3),
    FixedRuleConfig(
        "spread5_n3",
        positive_limit=5,
        negative_limit=3,
        positive_layout="spread",
        min_positive_distance_patches=3.0,
    ),
)


def build_spatial_radius_rules(radii: Sequence[float]) -> tuple[FixedRuleConfig, ...]:
    """Return a matched-budget Top-K baseline plus deterministic spread actions."""
    if not radii or any(radius <= 0.0 for radius in radii):
        raise ValueError("at least one positive spread radius is required")
    return (
        FixedRuleConfig("top5_n3", positive_limit=5, negative_limit=3),
        *(
            FixedRuleConfig(
                f"spread5_n3_d{radius:g}",
                positive_limit=5,
                negative_limit=3,
                positive_layout="spread",
                min_positive_distance_patches=float(radius),
            )
            for radius in radii
        ),
    )


@dataclass(frozen=True)
class MultiReferenceFeasibilityConfig:
    dataset_root: Path = Path(r"D:\SAM\code\datasets\MoNuSAC\processed")
    dino_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth")
    sam_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth")
    output_dir: Path = Path("artifacts/stage4_monusac_multireference_feasibility")
    seed: int = 20260716
    reference_count: int = 3
    policy_train_query_count: int = 48
    validation_query_count: int = 12
    device: str = "cpu"


def rule_candidate_kwargs(rule: FixedRuleConfig) -> dict[str, float | int | str]:
    return {
        "positive_limit": rule.positive_limit,
        "negative_limit": rule.negative_limit,
        "lambda_bg": rule.lambda_bg,
        "positive_layout": rule.positive_layout,
        "min_positive_distance_patches": rule.min_positive_distance_patches,
    }


def _evaluate_reference_query(
    reference: CaseRecord,
    query: CaseRecord,
    split: str,
    rules: Sequence,
    *,
    matcher: torch.nn.Module,
    foreground_prototype: torch.Tensor,
    background_prototype: torch.Tensor,
    predictor: object,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    query_image = load_rgb(query.image_path)
    query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
    foreground_scores, background_scores = matching_maps_from_prototypes(
        matcher, foreground_prototype, background_prototype, query_resized, device
    )
    target = load_binary_mask(query.mask_path, size=query_image.size)
    rows: list[dict[str, float | int | str]] = []
    for rule in rules:
        pools = build_candidate_pools(foreground_scores, background_scores, patch_size=PATCH_SIZE, **rule_candidate_kwargs(rule))
        positive_points, negative_points = _scale_points(pools, query_image.size)
        prediction, sam_score = predict_sam(predictor, query_image, positive_points, negative_points)
        features = build_action_features(foreground_scores, background_scores, pools, rule)
        rows.append(
            {
                "case_id": episode_id(reference.case_id, query.case_id),
                "reference_case_id": reference.case_id,
                "query_case_id": query.case_id,
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


def _write_rows(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    fields = (
        "case_id",
        "reference_case_id",
        "query_case_id",
        "split",
        "rule_name",
        "utility",
        "sam_score",
        "positive_count",
        "negative_count",
        "prompt_count",
        *POLICY_FEATURE_NAMES,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: Sequence[dict[str, float | int | str]], split: str) -> dict[str, dict[str, float | int]]:
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


def _protocol_dict(protocol: MultiReferenceProtocol) -> dict:
    return {
        "seed": protocol.seed,
        "reference_case_ids": [case.case_id for case in protocol.references],
        "policy_train_query_ids": [case.case_id for case in protocol.policy_train_queries],
        "validation_query_ids": [case.case_id for case in protocol.validation_queries],
        "test_case_ids_reserved": [case.case_id for case in protocol.test_cases],
    }


def run_multireference_feasibility(
    config: MultiReferenceFeasibilityConfig, rules: Sequence = DEFAULT_FIXED_RULES
) -> dict:
    if not config.dino_checkpoint.is_file() or not config.sam_checkpoint.is_file():
        raise FileNotFoundError("Stage 4 requires both DINOv2-L and SAM ViT-B checkpoints")
    protocol = build_multireference_protocol(
        discover_cases(config.dataset_root, "train"),
        discover_cases(config.dataset_root, "test"),
        seed=config.seed,
        reference_count=config.reference_count,
        policy_train_query_count=config.policy_train_query_count,
        validation_query_count=config.validation_query_count,
    )
    started = time.perf_counter()
    device = torch.device(config.device)
    matcher = load_dinov2(config.dino_checkpoint, device)
    predictor = load_sam_predictor(config.sam_checkpoint, device)
    reference_contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for reference in protocol.references:
        reference_image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        reference_contexts[reference.case_id] = reference_prototypes(
            matcher, reference_image, load_mask_for_features(reference.mask_path), device
        )
    rows: list[dict[str, float | int | str]] = []
    for split, queries in (
        ("train", protocol.policy_train_queries),
        ("validation", protocol.validation_queries),
    ):
        for reference in protocol.references:
            foreground_prototype, background_prototype = reference_contexts[reference.case_id]
            for query in queries:
                rows.extend(
                    _evaluate_reference_query(
                        reference,
                        query,
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
    fit = fit_utility_policy(train_rows, validation_rows, feature_names=POLICY_FEATURE_NAMES)
    _, validation_policy_dice = _policy_outcome(validation_rows, fit.estimator)
    output_dir = config.output_dir
    _write_rows(output_dir / "case_action_utilities.csv", rows)
    validation_metrics = _metrics(rows, "validation")
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "protocol": _protocol_dict(protocol),
        "rules": [asdict(rule) for rule in rules],
        "fit": {
            "regressor": fit.regressor_name,
            "selected_min_samples_leaf_on_validation": fit.min_samples_leaf,
            "validation_selected_mean_utility": fit.validation_selected_mean_utility,
        },
        "validation": {"fixed_rules": validation_metrics, "policy_selected_mean_dice": validation_policy_dice},
        "test_status": "reserved and not evaluated in feasibility pass",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
