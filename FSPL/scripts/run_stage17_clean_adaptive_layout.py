from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from fspl.dataset import discover_cases, discover_idrid_cases
from fspl.models import (
    IMAGE_SIZE,
    load_dinov2,
    load_mask_for_features,
    load_rgb,
    load_sam_predictor,
    reference_prototypes,
)
from fspl.multiref import build_multireference_protocol
from fspl.policy import fit_frozen_utility_policy, fit_utility_policy
from fspl.stage1 import image_resample
from fspl.stage3 import POLICY_FEATURE_NAMES, _policy_outcome, _write_predictions_csv
from fspl.stage4 import _evaluate_reference_query, _metrics, _protocol_dict, _write_rows, build_spatial_radius_rules


MONUSAC_ROOT = Path(r"D:\SAM\code\datasets\MoNuSAC\processed")
IDRID_ROOT = Path(r"D:\SAM\code\datasets\IDRID")
EXCLUSION_SUMMARIES = (
    Path("artifacts/stage4_monusac_multireference_feasibility/summary.json"),
    Path("artifacts/stage8_monusac_seed20260717/summary.json"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop adaptive spatial layout on untouched MoNuSAC train cases and evaluate once on IDRiD test."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monusac-root", type=Path, default=MONUSAC_ROOT)
    parser.add_argument("--idrid-root", type=Path, default=IDRID_ROOT)
    parser.add_argument("--dino-checkpoint", type=Path, default=Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth"))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth"))
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--reference-count", type=int, default=3)
    parser.add_argument("--policy-train-query-count", type=int, default=48)
    parser.add_argument("--validation-query-count", type=int, default=12)
    parser.add_argument("--radii", type=float, nargs="+", default=(2.0, 3.0, 4.0))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    args.radii = tuple(args.radii)
    return args


def _prior_used_monusac_ids() -> set[str]:
    stage4 = json.loads(EXCLUSION_SUMMARIES[0].read_text(encoding="utf-8"))["protocol"]
    stage8 = json.loads(EXCLUSION_SUMMARIES[1].read_text(encoding="utf-8"))["protocol"]
    return set(stage4["reference_case_ids"]) | set(stage4["policy_train_query_ids"]) | set(
        stage4["validation_query_ids"]
    ) | set(stage8["references"])


def _reference_contexts(cases: tuple, matcher: torch.nn.Module, device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for reference in cases:
        image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        contexts[reference.case_id] = reference_prototypes(
            matcher, image, load_mask_for_features(reference.mask_path), device
        )
    return contexts


def _evaluate_splits(protocol, rules, matcher, predictor, device: torch.device) -> list[dict[str, float | int | str]]:
    contexts = _reference_contexts(protocol.references, matcher, device)
    rows: list[dict[str, float | int | str]] = []
    for split, queries in (("train", protocol.policy_train_queries), ("validation", protocol.validation_queries)):
        for reference in protocol.references:
            foreground, background = contexts[reference.case_id]
            for query in queries:
                rows.extend(
                    _evaluate_reference_query(
                        reference,
                        query,
                        split,
                        rules,
                        matcher=matcher,
                        foreground_prototype=foreground,
                        background_prototype=background,
                        predictor=predictor,
                        device=device,
                    )
                )
    return rows


def _select_idrid_references(train_cases: list, seed: int, count: int) -> tuple:
    ordered = sorted(train_cases, key=lambda case: case.case_id)
    if len(ordered) < count:
        raise ValueError("not enough IDRiD training cases for requested references")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    return tuple(sorted(shuffled[:count], key=lambda case: case.case_id))


def _evaluate_idrid_test(references: tuple, test_cases: list, rules, matcher, predictor, device: torch.device) -> list[dict[str, float | int | str]]:
    contexts = _reference_contexts(references, matcher, device)
    rows: list[dict[str, float | int | str]] = []
    for reference in references:
        foreground, background = contexts[reference.case_id]
        for query in test_cases:
            rows.extend(
                _evaluate_reference_query(
                    reference,
                    query,
                    "test",
                    rules,
                    matcher=matcher,
                    foreground_prototype=foreground,
                    background_prototype=background,
                    predictor=predictor,
                    device=device,
                )
            )
    return rows


def write_policy_selections(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write selected policy actions with their predicted utility audit field."""
    _write_predictions_csv(path, rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dino_checkpoint.is_file() or not args.sam_checkpoint.is_file():
        raise FileNotFoundError("both frozen checkpoints must exist")
    used_ids = _prior_used_monusac_ids()
    clean_train_cases = tuple(case for case in discover_cases(args.monusac_root, "train") if case.case_id not in used_ids)
    protocol = build_multireference_protocol(
        clean_train_cases,
        (),
        seed=args.seed,
        reference_count=args.reference_count,
        policy_train_query_count=args.policy_train_query_count,
        validation_query_count=args.validation_query_count,
    )
    rules = build_spatial_radius_rules(args.radii)
    device = torch.device(args.device)
    started = time.perf_counter()
    matcher = load_dinov2(args.dino_checkpoint, device)
    predictor = load_sam_predictor(args.sam_checkpoint, device)

    development_rows = _evaluate_splits(protocol, rules, matcher, predictor, device)
    train_rows = [row for row in development_rows if row["split"] == "train"]
    validation_rows = [row for row in development_rows if row["split"] == "validation"]
    fit = fit_utility_policy(train_rows, validation_rows, feature_names=POLICY_FEATURE_NAMES)
    validation_selected, validation_policy_dice = _policy_outcome(validation_rows, fit.estimator)
    frozen_estimator = fit_frozen_utility_policy(
        [*train_rows, *validation_rows], feature_names=POLICY_FEATURE_NAMES, min_samples_leaf=fit.min_samples_leaf
    )

    idrid_references = _select_idrid_references(
        discover_idrid_cases(args.idrid_root, "train"), args.seed, args.reference_count
    )
    idrid_rows = _evaluate_idrid_test(
        idrid_references,
        discover_idrid_cases(args.idrid_root, "test"),
        rules,
        matcher,
        predictor,
        device,
    )
    idrid_selected, idrid_policy_dice = _policy_outcome(idrid_rows, frozen_estimator)

    _write_rows(args.output_dir / "development_case_action_utilities.csv", development_rows)
    _write_rows(args.output_dir / "idrid_test_case_action_utilities.csv", idrid_rows)
    write_policy_selections(args.output_dir / "validation_policy_selections.csv", validation_selected)
    write_policy_selections(args.output_dir / "idrid_policy_selections.csv", idrid_selected)
    summary = {
        "method_status": "clean-development adaptive-layout policy; external IDRiD test evaluated once after action set was locked",
        "clean_monusac_train_case_count": len(clean_train_cases),
        "excluded_prior_monusac_train_case_ids": sorted(used_ids),
        "development_protocol": _protocol_dict(protocol),
        "rules": [rule.__dict__ for rule in rules],
        "development": {
            "train_action_row_count": len(train_rows),
            "validation_action_row_count": len(validation_rows),
            "validation_fixed_rules": _metrics(validation_rows, "validation"),
            "selected_min_samples_leaf": fit.min_samples_leaf,
            "validation_policy_selected_mean_dice": validation_policy_dice,
        },
        "idrid_external_test": {
            "reference_case_ids": [case.case_id for case in idrid_references],
            "test_case_count": len(discover_idrid_cases(args.idrid_root, "test")),
            "episodes_per_rule": len(idrid_references) * len(discover_idrid_cases(args.idrid_root, "test")),
            "fixed_rules": _metrics(idrid_rows, "test"),
            "policy_selected_mean_dice": idrid_policy_dice,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["idrid_external_test"], indent=2))


if __name__ == "__main__":
    main()
