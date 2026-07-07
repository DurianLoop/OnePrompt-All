from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate whether Adapter/LoRA training is ready after P0-P2 closures.")
    parser.add_argument("--p0-feature-json", default="runs/research_direction/p0_p3_closure/p0_feature_feedback/elephant_pair/feedback_points.json")
    parser.add_argument("--p1-taxonomy-json", default="runs/research_direction/p0_p3_closure/p1_text_taxonomy/taxonomy.json")
    parser.add_argument("--p2-medical-json", default="runs/research_direction/p0_p3_closure/p2_medical_gate/medical_gate_report.json")
    parser.add_argument("--p2-eval-json", default="runs/research_direction/p0_p3_closure/p2_medical_eval/medical_eval.json")
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p3_adapter_gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = read_json(resolve_path(args.p0_feature_json))
    p1 = read_json(resolve_path(args.p1_taxonomy_json))
    p2 = read_json(resolve_path(args.p2_medical_json))
    p2_eval = read_json(resolve_path(args.p2_eval_json))
    p0_ready = bool(p0.get("after_selected_count") is not None)
    p1_ready = bool(p1.get("record_count", 0) > 0)
    p2_ready = bool(p2.get("can_run_dice_iou"))
    p2_eval_ready = bool(p2_eval.get("pair_count", 0) > 0)
    ready_for_training = bool(p0_ready and p1_ready and p2_ready)
    if ready_for_training:
        decision = "adapter_training_ready"
        next_action = "Prepare train/val split and run a small adapter/LoRA smoke experiment."
    else:
        decision = "training_not_recommended"
        next_action = "Do not start Adapter/LoRA yet; first obtain medical data with masks and finish P2 Dice/IoU closure."
    report = {
        "gate_criteria": {
            "p0_feature_feedback_evidence": p0_ready,
            "p1_text_taxonomy_evidence": p1_ready,
            "p2_medical_data_with_masks": p2_ready,
            "p2_medical_eval_pair_exists": p2_eval_ready,
        },
        "required_condition": "ready_for_training = true only if P0 evidence, P1 evidence, and medical data with masks are all present",
        "ready_for_training": ready_for_training,
        "decision": decision,
        "next_action": next_action,
        "notes": [
            "medical data with masks is required before any Adapter/LoRA claim",
            f"P2 current Dice mean is {p2_eval.get('dice_mean')} on {p2_eval.get('pair_count', 0)} available prediction/GT pair(s)",
            "adapter/LoRA may be prepared as a smoke experiment, but no adapter performance claim exists until training and held-out evaluation are run",
        ],
    }
    (out_dir / "adapter_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# Adapter/LoRA Gate Summary",
            "",
            f"- Ready for training: `{ready_for_training}`",
            f"- Decision: `{decision}`",
            f"- Next action: `{next_action}`",
            f"- Gate criteria: `{report['gate_criteria']}`",
            "",
            "## Artifacts",
            "",
            "- `adapter_gate_report.json`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Ready for training: {ready_for_training}")
    print(f"Decision: {decision}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
