from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }


def parse_box(text: str) -> str:
    box = json.loads(text)
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"bad support_box_xyxy: {text}")
    return ",".join(str(int(v)) for v in box)


def initial_feedback_from_stage1(stage1_dir: Path, feedback_dir: Path) -> dict[str, Any]:
    points_path = stage1_dir / "points.json"
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    payload = json.loads(points_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload["points"]:
        similarity = float(row["similarity"])
        rows.append(
            {
                "rank": int(row["rank"]),
                "x_original": float(row["x_original"]),
                "y_original": float(row["y_original"]),
                "similarity_before": similarity,
                "similarity_after": similarity,
                "delta": 0.0,
                "selected_by_threshold": bool(row["selected_by_threshold"]),
            }
        )
    feedback_payload = {
        "target_image": payload["target_image"],
        "reference_image": payload["reference_image"],
        "reference_mask": payload.get("reference_mask"),
        "mode": "initial_no_manual_feedback",
        "points": rows,
        "source_stage1": rel(points_path),
    }
    feedback_dir.mkdir(parents=True, exist_ok=True)
    (feedback_dir / "feedback_points.json").write_text(json.dumps(feedback_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (feedback_dir / "feedback_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "rank",
            "x_original",
            "y_original",
            "similarity_before",
            "similarity_after",
            "delta",
            "selected_by_threshold",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return feedback_payload


def build_case_plan(row: dict[str, str], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = out_dir / row["pilot_id"]
    stage1_dir = case_dir / "stage1"
    feedback_dir = case_dir / "feedback_initial"
    stage2_dir = case_dir / "stage2_sampling"
    reference_box = parse_box(row["support_box_xyxy"])
    stage1_command = [
        sys.executable,
        "-B",
        "code/benchmarks/export_matcher_stage1_points.py",
        "--reference-image",
        row["support_image_path"],
        "--target-image",
        row["query_image_path"],
        "--reference-box",
        reference_box,
        "--reference-mask",
        row["support_mask_path"],
        "--top-k",
        str(args.top_k),
        "--threshold",
        str(args.stage1_threshold),
        "--device",
        args.device,
        "--out-dir",
        rel(stage1_dir),
    ]
    adapter_command = [
        sys.executable,
        "-B",
        "code/benchmarks/run_matcher_medical_pilot.py",
        "--manifest",
        rel(resolve_path(args.manifest)),
        "--out-dir",
        rel(out_dir),
        "--adapter-only",
        row["pilot_id"],
    ]
    stage2_command = [
        sys.executable,
        "-B",
        "code/benchmarks/sample_stage2_prompts.py",
        "--feedback-dir",
        rel(feedback_dir),
        "--cluster-method",
        "dbscan",
        "--eps",
        str(args.cluster_eps),
        "--min-samples",
        str(args.min_samples),
        "--sample-sizes",
        args.sample_sizes,
        "--samples-per-cluster",
        str(args.samples_per_cluster),
        "--out-dir",
        rel(stage2_dir),
    ]
    return {
        "pilot_id": row["pilot_id"],
        "dataset": row["dataset"],
        "support_case_id": row["support_case_id"],
        "query_case_id": row["query_case_id"],
        "target_hint": row["target_hint"],
        "stage1_dir": rel(stage1_dir),
        "feedback_dir": rel(feedback_dir),
        "stage2_dir": rel(stage2_dir),
        "stage1_command": command_text(stage1_command),
        "feedback_adapter_command": command_text(adapter_command),
        "stage2_sampler_command": command_text(stage2_command),
        "_stage1_command_list": stage1_command,
        "_stage2_command_list": stage2_command,
    }


def write_plan(out_dir: Path, plans: list[dict[str, Any]], executed: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    public_plans = [{key: value for key, value in plan.items() if not key.startswith("_")} for plan in plans]
    payload = {
        "case_count": len(public_plans),
        "cases": public_plans,
        "executed": executed,
    }
    (out_dir / "medical_pilot_run_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Medical Pilot Run Plan",
        "",
        f"- Cases: `{len(public_plans)}`",
        f"- Executed steps: `{[item['step'] for item in executed]}`",
        "",
        "## Planned Cases",
        "",
    ]
    for plan in public_plans:
        lines.extend(
            [
                f"### {plan['pilot_id']} ({plan['dataset']})",
                f"- Support: `{plan['support_case_id']}`",
                f"- Query: `{plan['query_case_id']}`",
                f"- Stage1: `{plan['stage1_command']}`",
                f"- Feedback adapter: `{plan['feedback_adapter_command']}`",
                f"- Stage2 sampler: `{plan['stage2_sampler_command']}`",
                "",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_adapter_only(args: argparse.Namespace) -> None:
    manifest = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    rows = read_manifest(manifest)
    matched = [row for row in rows if row["pilot_id"] == args.adapter_only]
    if not matched:
        raise ValueError(f"Unknown pilot_id: {args.adapter_only}")
    stage1_dir = out_dir / args.adapter_only / "stage1"
    feedback_dir = out_dir / args.adapter_only / "feedback_initial"
    initial_feedback_from_stage1(stage1_dir, feedback_dir)
    print(f"Wrote {rel(feedback_dir / 'feedback_points.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan the medical Matcher pilot over support/query cases.")
    parser.add_argument("--manifest", default="runs/research_direction/matcher_medical_pilot/medical_matcher_pilot_manifest.csv")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_pilot/run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--run-stage2-sampling", action="store_true")
    parser.add_argument("--adapter-only", default="")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--stage1-threshold", type=float, default=0.55)
    parser.add_argument("--cluster-eps", type=float, default=28.0)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--sample-sizes", default="1,2,3")
    parser.add_argument("--samples-per-cluster", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.adapter_only:
        run_adapter_only(args)
        return

    manifest = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    rows = read_manifest(manifest)
    plans = [build_case_plan(row, out_dir, args) for row in rows]
    executed: list[dict[str, Any]] = []

    if args.run_stage1:
        for plan in plans:
            result = run_command(plan["_stage1_command_list"])
            executed.append({"pilot_id": plan["pilot_id"], "step": "stage1", **result})
            if result["returncode"] != 0:
                write_plan(out_dir, plans, executed)
                raise SystemExit(result["returncode"])

    if args.run_stage2_sampling:
        for plan in plans:
            initial_feedback_from_stage1(resolve_path(plan["stage1_dir"]), resolve_path(plan["feedback_dir"]))
            result = run_command(plan["_stage2_command_list"])
            executed.append({"pilot_id": plan["pilot_id"], "step": "stage2_sampling", **result})
            if result["returncode"] != 0:
                write_plan(out_dir, plans, executed)
                raise SystemExit(result["returncode"])

    write_plan(out_dir, plans, executed)
    print(f"Wrote medical pilot run plan to {rel(out_dir / 'medical_pilot_run_plan.json')}")


if __name__ == "__main__":
    main()
