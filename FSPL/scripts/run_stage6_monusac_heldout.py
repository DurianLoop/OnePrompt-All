from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from fspl.dataset import discover_cases
from fspl.models import IMAGE_SIZE, load_dinov2, load_mask_for_features, load_rgb, load_sam_predictor, reference_prototypes
from fspl.multiref import build_multireference_protocol
from fspl.policy import fit_frozen_utility_policy
from fspl.stage1 import image_resample
from fspl.stage2 import BaselineRow, aggregate_rule_metrics
from fspl.stage3 import POLICY_FEATURE_NAMES, _policy_outcome
from fspl.stage4 import SPATIAL_LAYOUT_RULES, _evaluate_reference_query, _write_rows


ROOT = Path(r"D:\SAM\code\datasets\MoNuSAC\processed")
SOURCE = Path("artifacts/stage5_spatial_layout_medium/case_action_utilities.csv")
OUT = Path("artifacts/stage6_monusac_heldout")


def typed_rows(path: Path) -> list[dict[str, float | int | str]]:
    float_fields = {*POLICY_FEATURE_NAMES, "utility", "sam_score"}
    int_fields = {"positive_count", "negative_count", "prompt_count"}
    return [
        {key: float(value) if key in float_fields else int(value) if key in int_fields else value for key, value in row.items()}
        for row in csv.DictReader(path.open(encoding="utf-8"))
    ]


def main() -> None:
    prior_rows = typed_rows(SOURCE)
    estimator = fit_frozen_utility_policy(prior_rows, feature_names=POLICY_FEATURE_NAMES, min_samples_leaf=2)
    protocol = build_multireference_protocol(
        discover_cases(ROOT, "train"), discover_cases(ROOT, "test"), seed=20260716,
        reference_count=3, policy_train_query_count=48, validation_query_count=12,
    )
    started = time.perf_counter()
    device = torch.device("cpu")
    matcher = load_dinov2(Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth"), device)
    predictor = load_sam_predictor(Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth"), device)
    contexts = {}
    for reference in protocol.references:
        image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        contexts[reference.case_id] = reference_prototypes(matcher, image, load_mask_for_features(reference.mask_path), device)
    rows = []
    for reference in protocol.references:
        fg, bg = contexts[reference.case_id]
        for query in protocol.test_cases:
            rows.extend(_evaluate_reference_query(reference, query, "test", SPATIAL_LAYOUT_RULES, matcher=matcher, foreground_prototype=fg, background_prototype=bg, predictor=predictor, device=device))
    selected, policy_dice = _policy_outcome(rows, estimator)
    _write_rows(OUT / "case_action_utilities.csv", rows)
    with (OUT / "policy_selected.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    metrics = aggregate_rule_metrics([BaselineRow(str(row["case_id"]), "test", str(row["rule_name"]), float(row["utility"]), float(row["sam_score"]), int(row["positive_count"]), int(row["negative_count"])) for row in rows], split="test")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"protocol":{"references":[x.case_id for x in protocol.references],"test_case_count":len(protocol.test_cases)},"frozen_policy":{"source":str(SOURCE),"min_samples_leaf":2},"official_test":{"fixed_rules":metrics,"policy_selected_mean_dice":policy_dice},"elapsed_seconds":time.perf_counter()-started}, indent=2), encoding="utf-8")
    print(f"policy_test_dice={policy_dice:.6f}")


if __name__ == "__main__":
    main()
