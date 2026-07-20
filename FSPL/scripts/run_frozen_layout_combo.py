from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fspl.dataset import discover_cases
from fspl.models import (
    IMAGE_SIZE,
    load_dinov2,
    load_mask_for_features,
    load_rgb,
    load_sam_predictor,
    reference_prototypes,
)
from fspl.multiref import build_multireference_protocol
from fspl.stage1 import image_resample
from fspl.stage2 import BaselineRow, aggregate_rule_metrics
from fspl.stage4 import SPATIAL_LAYOUT_RULES, _evaluate_reference_query, _write_rows


DEFAULT_DATASET_ROOT = Path(r"D:\SAM\code\datasets\MoNuSAC\processed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the prespecified frozen Top-K versus radius-3 spatial-layout rules."
    )
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dino_checkpoint.is_file() or not args.sam_checkpoint.is_file():
        raise FileNotFoundError("both checkpoint paths must exist")
    protocol = build_multireference_protocol(
        discover_cases(args.dataset_root, "train"),
        discover_cases(args.dataset_root, "test"),
        seed=args.seed,
        reference_count=3,
        policy_train_query_count=48,
        validation_query_count=12,
    )
    if args.test_start < 0:
        raise ValueError("test-start must be non-negative")
    test_cases = protocol.test_cases[args.test_start :]
    if args.max_test_cases is not None:
        test_cases = test_cases[: args.max_test_cases]
    if not test_cases:
        raise ValueError("at least one test case must be selected")
    device = torch.device(args.device)
    started = time.perf_counter()
    matcher = load_dinov2(args.dino_checkpoint, device)
    predictor = load_sam_predictor(args.sam_checkpoint, device)
    contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for reference in protocol.references:
        image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        contexts[reference.case_id] = reference_prototypes(
            matcher, image, load_mask_for_features(reference.mask_path), device
        )

    rows: list[dict[str, object]] = []
    for reference in protocol.references:
        foreground, background = contexts[reference.case_id]
        for query in test_cases:
            rows.extend(
                _evaluate_reference_query(
                    reference,
                    query,
                    "test",
                    SPATIAL_LAYOUT_RULES,
                    matcher=matcher,
                    foreground_prototype=foreground,
                    background_prototype=background,
                    predictor=predictor,
                    device=device,
                )
            )

    _write_rows(args.output_dir / "case_action_utilities.csv", rows)
    metrics = aggregate_rule_metrics(
        [
            BaselineRow(
                str(row["case_id"]),
                "test",
                str(row["rule_name"]),
                float(row["utility"]),
                float(row["sam_score"]),
                int(row["positive_count"]),
                int(row["negative_count"]),
            )
            for row in rows
        ],
        split="test",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "seed": args.seed,
            "references": [case.case_id for case in protocol.references],
            "test_start": args.test_start,
            "test_case_count": len(test_cases),
            "episodes_per_rule": len(protocol.references) * len(test_cases),
        },
        "matcher_checkpoint": str(args.dino_checkpoint),
        "segmenter_checkpoint": str(args.sam_checkpoint),
        "rules": [rule.name for rule in SPATIAL_LAYOUT_RULES],
        "official_test": {"fixed_rules": metrics},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
