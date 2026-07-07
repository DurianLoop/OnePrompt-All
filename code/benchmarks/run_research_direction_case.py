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


def run_step(name: str, command: list[str], dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"name": name, "command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def fail_if_needed(result: dict[str, Any]) -> None:
    if result.get("returncode") not in (0, None):
        raise SystemExit(f"Step failed: {result['name']}\nSTDOUT:\n{result.get('stdout', '')}\nSTDERR:\n{result.get('stderr', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 1/2/3 research-direction pipeline for one case.")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--reference-box", required=True)
    parser.add_argument("--positive-points", default=" ")
    parser.add_argument("--negative-points", default=" ")
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--eps", type=float, default=60.0)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sample-sizes", default="1,3")
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    parser.add_argument("--max-anchor-distance", type=float, default=110.0)
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--min-area-pct", type=float, default=0.5)
    parser.add_argument("--max-area-pct", type=float, default=45.0)
    parser.add_argument("--max-bbox-area-pct", type=float, default=50.0)
    parser.add_argument("--max-retained-per-cluster", type=int, default=1)
    parser.add_argument("--labels-json", default="")
    parser.add_argument("--out-root", default="runs/research_direction/p0_p3_closure/p0_runner")
    parser.add_argument("--checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--dinov2-weights", default="assets/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = resolve_path(args.out_root) / args.case_name
    stage1 = case_dir / "stage1"
    feature_feedback = case_dir / "feature_feedback"
    stage2_sampling = case_dir / "stage2_sampling"
    stage2_proposals = case_dir / "stage2_proposals"
    stage3_feature = case_dir / "stage3_feature"
    learning_policy = case_dir / "learning_policy"
    py = sys.executable
    steps = [
        (
            "stage1",
            [
                py, "-B", "code/benchmarks/export_matcher_stage1_points.py",
                "--reference-image", args.reference_image,
                "--target-image", args.target_image,
                "--reference-box", args.reference_box,
                "--top-k", str(args.top_k),
                "--threshold", str(args.threshold),
                "--out-dir", str(stage1),
            ],
        ),
        (
            "feature_feedback",
            [
                py, "-B", "code/benchmarks/apply_stage1_feature_feedback.py",
                "--stage1-dir", str(stage1),
                "--positive-points", args.positive_points,
                "--negative-points", args.negative_points,
                "--top-k", str(args.top_k),
                "--threshold", str(args.threshold),
                "--out-dir", str(feature_feedback),
                "--dinov2-weights", args.dinov2_weights,
            ],
        ),
        (
            "stage2_sampling",
            [
                py, "-B", "code/benchmarks/sample_stage2_prompts.py",
                "--feedback-dir", str(feature_feedback),
                "--selected-only",
                "--cluster-method", "dbscan",
                "--eps", str(args.eps),
                "--min-samples", str(args.min_samples),
                "--sample-sizes", args.sample_sizes,
                "--samples-per-cluster", str(args.samples_per_cluster),
                "--max-anchor-distance", str(args.max_anchor_distance),
                "--out-dir", str(stage2_sampling),
            ],
        ),
        (
            "stage2_proposals",
            [
                py, "-B", "code/benchmarks/generate_stage2_sam_proposals.py",
                "--stage2-dir", str(stage2_sampling),
                "--checkpoint", args.checkpoint,
                "--model-type", "vit_b",
                "--device", "auto",
                "--max-groups", str(args.max_groups),
                "--out-dir", str(stage2_proposals),
            ],
        ),
        (
            "stage3_feature",
            [
                py, "-B", "code/benchmarks/feature_match_stage3_proposals.py",
                "--proposals-dir", str(stage2_proposals),
                "--stage1-dir", str(stage1),
                "--dinov2-weights", args.dinov2_weights,
                "--min-instance-similarity", "0.45",
                "--min-score", "0.80",
                "--min-area-pct", str(args.min_area_pct),
                "--max-area-pct", str(args.max_area_pct),
                "--max-bbox-area-pct", str(args.max_bbox_area_pct),
                "--min-cluster-coverage", "0.30",
                "--nms-iou", "0.75",
                "--max-retained-per-cluster", str(args.max_retained_per_cluster),
                "--out-dir", str(stage3_feature),
            ],
        ),
    ]
    if args.labels_json.strip():
        steps.append(
            (
                "learning_policy",
                [
                    py, "-B", "code/benchmarks/run_matcher_feedback_learning_policy.py",
                    "--stage3-feature-json", str(stage3_feature / "feature_scores.json"),
                    "--labels-json", args.labels_json,
                    "--out-dir", str(learning_policy),
                ],
            )
        )
    case_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, command in steps:
        result = run_step(name, command, args.dry_run)
        results.append(result)
        fail_if_needed(result)
    summary = {"case_name": args.case_name, "case_dir": str(case_dir.relative_to(WORKSPACE_ROOT)), "steps": results}
    feature_json = stage3_feature / "feature_scores.json"
    if feature_json.exists():
        payload = json.loads(feature_json.read_text(encoding="utf-8"))
        summary["stage3"] = {
            "proposal_count": payload.get("proposal_count"),
            "candidate_count": payload.get("candidate_count"),
            "k_pred": payload.get("k_pred"),
            "retained_group_ids": payload.get("retained_group_ids"),
        }
    policy_json = learning_policy / "matcher_learning_policy.json"
    if policy_json.exists():
        policy = json.loads(policy_json.read_text(encoding="utf-8"))
        summary["learning_policy"] = {
            "threshold_source": policy.get("threshold_source"),
            "instance_similarity_threshold": policy.get("instance_similarity_threshold"),
            "selected_group_ids": policy.get("selected_group_ids"),
        }
    (case_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {case_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
