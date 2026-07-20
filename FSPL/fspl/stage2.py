from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
from fspl.protocol import (
    FixedBaselineProtocol,
    FixedRuleConfig,
    RuleSelection,
    build_fixed_baseline_protocol,
    select_primary_rule,
)
from fspl.stage1 import _dice, _scale_points, image_resample


DEFAULT_FIXED_RULES = (
    FixedRuleConfig("top1_n0", positive_limit=1, negative_limit=0),
    FixedRuleConfig("top3_n0", positive_limit=3, negative_limit=0),
    FixedRuleConfig("top5_n0", positive_limit=5, negative_limit=0),
    FixedRuleConfig("top5_n3", positive_limit=5, negative_limit=3),
)


@dataclass(frozen=True)
class FixedBaselineConfig:
    dataset_root: Path = Path(r"D:\SAM\code\datasets\MonuSeg\official")
    dino_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth")
    sam_checkpoint: Path = Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth")
    output_dir: Path = Path("artifacts/stage2_fixed_baselines")
    seed: int = 20260716
    validation_count: int = 6
    device: str = "cpu"


@dataclass(frozen=True)
class BaselineRow:
    case_id: str
    split: str
    rule_name: str
    dice: float
    sam_score: float
    positive_count: int
    negative_count: int


def aggregate_rule_metrics(rows: Iterable[BaselineRow], *, split: str) -> dict[str, dict[str, float | int]]:
    dice_by_rule: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.split == split:
            dice_by_rule[row.rule_name].append(row.dice)
    return {
        name: {"case_count": len(values), "mean_dice": round(sum(values) / len(values), 12)}
        for name, values in sorted(dice_by_rule.items())
    }


def select_rule_from_rows(rows: Iterable[BaselineRow], rules: Sequence[FixedRuleConfig]) -> RuleSelection:
    dice_by_rule: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.split == "validation":
            dice_by_rule[row.rule_name].append(row.dice)
    return select_primary_rule(dice_by_rule, rules)


def _evaluate_case_rules(
    case: CaseRecord,
    split: str,
    rules: Sequence[FixedRuleConfig],
    *,
    matcher: torch.nn.Module,
    foreground_prototype: torch.Tensor,
    background_prototype: torch.Tensor,
    predictor: object,
    device: torch.device,
) -> list[BaselineRow]:
    query_image = load_rgb(case.image_path)
    query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
    foreground_scores, background_scores = matching_maps_from_prototypes(
        matcher, foreground_prototype, background_prototype, query_resized, device
    )
    target = load_binary_mask(case.mask_path, size=query_image.size)
    rows: list[BaselineRow] = []
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
        rows.append(
            BaselineRow(
                case_id=case.case_id,
                split=split,
                rule_name=rule.name,
                dice=_dice(prediction, target),
                sam_score=sam_score,
                positive_count=len(positive_points),
                negative_count=len(negative_points),
            )
        )
    return rows


def _write_rows(path: Path, rows: Sequence[BaselineRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BaselineRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_fixed_baseline_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def protocol_as_dict(protocol: FixedBaselineProtocol) -> dict:
    return {
        "seed": protocol.seed,
        "reference_case_id": protocol.reference.case_id,
        "validation_case_ids": [case.case_id for case in protocol.validation_cases],
        "test_case_ids": [case.case_id for case in protocol.test_cases],
    }


def run_fixed_baselines(
    config: FixedBaselineConfig, rules: Sequence[FixedRuleConfig] = DEFAULT_FIXED_RULES
) -> dict:
    if not rules:
        raise ValueError("at least one fixed rule is required")
    if not config.dino_checkpoint.is_file() or not config.sam_checkpoint.is_file():
        raise FileNotFoundError("Stage 2 requires both DINOv2-L and SAM ViT-B checkpoints")

    protocol = build_fixed_baseline_protocol(
        discover_cases(config.dataset_root, "train"),
        discover_cases(config.dataset_root, "test"),
        seed=config.seed,
        validation_count=config.validation_count,
    )
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

    rows: list[BaselineRow] = []
    for case in protocol.validation_cases:
        rows.extend(
            _evaluate_case_rules(
                case,
                "validation",
                rules,
                matcher=matcher,
                foreground_prototype=foreground_prototype,
                background_prototype=background_prototype,
                predictor=predictor,
                device=device,
            )
        )
    selection = select_rule_from_rows(rows, rules)
    for case in protocol.test_cases:
        rows.extend(
            _evaluate_case_rules(
                case,
                "test",
                rules,
                matcher=matcher,
                foreground_prototype=foreground_prototype,
                background_prototype=background_prototype,
                predictor=predictor,
                device=device,
            )
        )

    output_dir = config.output_dir
    _write_rows(output_dir / "case_metrics.csv", rows)
    test_metrics = aggregate_rule_metrics(rows, split="test")
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "protocol": protocol_as_dict(protocol),
        "rules": [asdict(rule) for rule in rules],
        "validation": aggregate_rule_metrics(rows, split="validation"),
        "primary_rule_selected_on_validation": selection.selected.name,
        "validation_mean_dice_by_rule": selection.mean_dice_by_rule,
        "official_test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_fixed_baseline_summary(output_dir / "summary.json", summary)
    return summary
