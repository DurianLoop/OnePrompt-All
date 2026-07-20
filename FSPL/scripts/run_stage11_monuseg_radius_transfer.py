from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from fspl.dataset import discover_cases
from fspl.models import IMAGE_SIZE, load_dinov2, load_mask_for_features, load_rgb, load_sam_predictor, reference_prototypes
from fspl.multiref import build_multireference_protocol
from fspl.protocol import FixedRuleConfig
from fspl.stage1 import image_resample
from fspl.stage2 import BaselineRow, aggregate_rule_metrics
from fspl.stage4 import _evaluate_reference_query, _write_rows


ROOT = Path(r"D:\SAM\code\datasets\MonuSeg\official")
OUT = Path("artifacts/stage11_monuseg_radius_transfer")
RULES = tuple(FixedRuleConfig(f"spread5_n3_d{distance}", 5, 3, positive_layout="spread", min_positive_distance_patches=float(distance)) for distance in (2, 4))


def main() -> None:
    protocol = build_multireference_protocol(discover_cases(ROOT, "train"), discover_cases(ROOT, "test"), seed=20260716, reference_count=3, policy_train_query_count=12, validation_query_count=6)
    started = time.perf_counter(); device = torch.device("cpu")
    matcher = load_dinov2(Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth"), device)
    predictor = load_sam_predictor(Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth"), device)
    contexts = {}
    for reference in protocol.references:
        image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        contexts[reference.case_id] = reference_prototypes(matcher, image, load_mask_for_features(reference.mask_path), device)
    rows = []
    for reference in protocol.references:
        foreground, background = contexts[reference.case_id]
        for query in protocol.test_cases:
            rows.extend(_evaluate_reference_query(reference, query, "test", RULES, matcher=matcher, foreground_prototype=foreground, background_prototype=background, predictor=predictor, device=device))
    _write_rows(OUT / "case_action_utilities.csv", rows)
    metrics = aggregate_rule_metrics([BaselineRow(str(row["case_id"]), "test", str(row["rule_name"]), float(row["utility"]), float(row["sam_score"]), int(row["positive_count"]), int(row["negative_count"])) for row in rows], split="test")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"protocol":{"seed":20260716,"references":[case.case_id for case in protocol.references],"test_case_count":len(protocol.test_cases)},"rules":[rule.name for rule in RULES],"official_test":{"fixed_rules":metrics},"elapsed_seconds":time.perf_counter()-started}, indent=2),encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
