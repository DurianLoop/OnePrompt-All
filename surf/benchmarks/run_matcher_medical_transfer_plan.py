from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def run_command(command: list[str], cwd: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-3000:],
        "stderr": completed.stderr[-3000:],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a medical transfer plan for Feedback/Learning Matcher.")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_transfer")
    parser.add_argument("--gate-out-dir", default="runs/research_direction/matcher_medical_transfer/medical_gate")
    parser.add_argument("--eval-out-dir", default="runs/research_direction/matcher_medical_transfer/medical_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    gate_dir = resolve_path(args.gate_out_dir)
    eval_dir = resolve_path(args.eval_out_dir)
    py = sys.executable
    gate_result = run_command([py, "-B", "code/benchmarks/medical_small_data_gate.py", "--out-dir", str(gate_dir)])
    eval_result = run_command([py, "-B", "code/benchmarks/evaluate_medical_prediction_pairs.py", "--out-dir", str(eval_dir)])
    gate_json = read_json(gate_dir / "medical_gate_report.json")
    eval_json = read_json(eval_dir / "medical_eval.json")
    can_run_masks = bool(gate_json.get("can_run_dice_iou"))
    pair_count = int(eval_json.get("pair_count", 0) or 0)
    readiness = {
        "data_with_masks_available": can_run_masks,
        "prediction_pairs_available": pair_count > 0,
        "ready_for_matcher_medical_casebook": can_run_masks,
        "ready_for_quantitative_dice_iou": pair_count > 0,
    }
    protocol = {
        "support_definition": "Use one medical support image with mask/box to define target anatomy or lesion.",
        "query_generation": "Run Matcher Stage1 on query images, then feedback prototype update if positive/negative points are provided.",
        "proposal_policy": "Use the natural-image learned feedback_learning_score as initialization, then relabel medical proposals and retrain a medical casebook policy.",
        "metrics": ["Dice", "IoU", "false_positive_count", "interaction_rounds", "K_pred_error"],
        "minimum_cases": {
            "pilot": 5,
            "stronger_internal_check": 20
        },
        "recommended_domains": ["MonuSeg nuclei", "IDRID/fundus lesion", "MedSAM demo masks if available"],
    }
    report = {
        "medical_gate": {
            "command_result": gate_result,
            "report": gate_json,
        },
        "prediction_eval": {
            "command_result": eval_result,
            "report": eval_json,
        },
        "matcher_transfer_protocol": protocol,
        "readiness": readiness,
        "next_commands": [
            "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\medical_small_data_gate.py --out-dir runs\\research_direction\\matcher_medical_transfer\\medical_gate",
            "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\evaluate_medical_prediction_pairs.py --out-dir runs\\research_direction\\matcher_medical_transfer\\medical_eval",
            "After selecting support/query medical cases, run Stage1/2/3 and add proposal accept/reject labels exactly like the natural-image casebook.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "medical_transfer_plan.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Medical Transfer Plan",
        "",
        f"- Data with masks available: `{readiness['data_with_masks_available']}`",
        f"- Prediction pairs available: `{readiness['prediction_pairs_available']}`",
        f"- Medical image candidates: `{gate_json.get('image_candidate_count')}`",
        f"- Medical mask candidates: `{gate_json.get('mask_candidate_count')}`",
        f"- Existing prediction pairs: `{pair_count}`",
        f"- Existing Dice mean: `{eval_json.get('dice_mean')}`",
        "",
        "## Transfer Protocol",
        "",
    ]
    for key, value in protocol.items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'medical_transfer_plan.json'}")


if __name__ == "__main__":
    main()
