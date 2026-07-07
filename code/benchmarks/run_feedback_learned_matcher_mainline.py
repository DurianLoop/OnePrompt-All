from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_ROOT):
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def run_casebook(
    manifest: Path,
    run_dir: Path,
    out_dir: Path,
    labels_json: Path | None = None,
    dataset_filter: str = "",
    positive_dice: float = 0.30,
) -> None:
    command = [
        sys.executable,
        "-B",
        "code/benchmarks/build_matcher_medical_casebook.py",
        "--manifest",
        rel(manifest),
        "--run-dir",
        rel(run_dir),
        "--out-dir",
        rel(out_dir),
        "--positive-dice",
        f"{positive_dice:g}",
    ]
    if labels_json is not None:
        command.extend(["--labels-json", rel(labels_json)])
    if dataset_filter:
        command.extend(["--dataset-filter", dataset_filter])
    run_command(command)


def metric_value(policy: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    current: Any = policy
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def table_row(setting: str, policy_dir: Path, labels: str) -> dict[str, Any]:
    policy = load_json(policy_dir / "medical_casebook_policy.json")
    loco_metrics = policy.get("leave_one_case_out", {}).get("metrics", {})
    return {
        "setting": setting,
        "cases": policy.get("case_count", 0),
        "proposals": policy.get("labeled_count", 0),
        "positive": policy.get("positive_count", 0),
        "negative": policy.get("negative_count", 0),
        "labels": labels,
        "best_model": policy.get("best_policy", {}).get("model", ""),
        "best_feature": policy.get("best_policy", {}).get("feature", ""),
        "stage3_f1": metric_value(policy, ["baseline_stage3", "metrics", "f1"]),
        "learned_f1": metric_value(policy, ["best_policy", "metrics", "f1"]),
        "loco_f1": float(loco_metrics.get("f1", 0.0) or 0.0),
        "stage3_dice_mean": metric_value(policy, ["baseline_stage3", "retained_per_case", "dice_mean"]),
        "learned_dice_mean": metric_value(policy, ["learned_best_per_case", "dice_mean"]),
        "oracle_dice_mean": metric_value(policy, ["oracle_best_per_case", "dice_mean"]),
        "policy_dir": rel(policy_dir),
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]], labels_path: Path, expanded_50_dir: Path) -> None:
    lines = [
        "# Feedback-Learned Matcher 医学主线结果",
        "",
        "## 产物",
        "",
        f"- 20 例 accept/reject 标签：`{rel(labels_path)}`",
        f"- 50 例扩展结果：`{rel(expanded_50_dir)}`",
        f"- 论文主表：`{rel(out_dir / 'paper_main_table.csv')}`",
        "",
        "## 论文主表",
        "",
        "| Setting | Cases | Proposals | Pos/Neg | Stage3 F1 | Learned F1 | LOCO F1 | Stage3 Dice | Learned Dice | Oracle Dice |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {cases} | {proposals} | {positive}/{negative} | {stage3_f1:.6f} | {learned_f1:.6f} | {loco_f1:.6f} | {stage3_dice_mean:.6f} | {learned_dice_mean:.6f} | {oracle_dice_mean:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. 20 例上，overall、IDRID-only、MonuSeg-only 都单独输出了 proposal-level learning policy，可直接用于论文 ablation。",
            "2. `manual_labels_20.json` 是一版 GT-assisted accept/reject 审查种子标签；它比纯 Stage3 retained/rejected 伪标签更适合作为下一轮人工修订入口。",
            "3. 50 例扩展用于主表规模化验证；若后续要投稿，应继续把 50 例中的伪标签替换为人工审查标签，并补充统计显著性或 bootstrap 置信区间。",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Feedback-Learned Matcher medical mainline.")
    parser.add_argument("--source-20-dir", default="runs/research_direction/matcher_medical_expanded")
    parser.add_argument("--expanded-50-dir", default="runs/research_direction/matcher_medical_expanded_50")
    parser.add_argument("--out-dir", default="runs/research_direction/feedback_learned_matcher_mainline")
    parser.add_argument("--positive-dice", type=float, default=0.30)
    parser.add_argument("--idrid-cases-50", type=int, default=30)
    parser.add_argument("--monuseg-cases-50", type=int, default=20)
    parser.add_argument("--skip-50-run", action="store_true")
    parser.add_argument("--force-50-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_20 = resolve_path(args.source_20_dir)
    expanded_50 = resolve_path(args.expanded_50_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_20 = out_dir / "manual_labels_20.json"
    run_command(
        [
            sys.executable,
            "-B",
            "code/benchmarks/build_matcher_medical_manual_labels.py",
            "--casebook-rows",
            rel(source_20 / "casebook" / "medical_casebook_rows.csv"),
            "--out-json",
            rel(labels_20),
            "--positive-dice",
            f"{float(args.positive_dice):g}",
        ]
    )

    manifest_20 = source_20 / "medical_matcher_pilot_manifest.csv"
    run_dir_20 = source_20 / "run"
    run_casebook(manifest_20, run_dir_20, out_dir / "overall_20", labels_json=labels_20, positive_dice=args.positive_dice)
    run_casebook(manifest_20, run_dir_20, out_dir / "idrid_20", labels_json=labels_20, dataset_filter="IDRID", positive_dice=args.positive_dice)
    run_casebook(manifest_20, run_dir_20, out_dir / "monuseg_20", labels_json=labels_20, dataset_filter="MonuSeg", positive_dice=args.positive_dice)

    manifest_50 = expanded_50 / "medical_matcher_pilot_manifest.csv"
    rows_50 = expanded_50 / "casebook" / "medical_casebook_rows.csv"
    need_50 = args.force_50_run or not (manifest_50.exists() and rows_50.exists())
    if need_50 and not args.skip_50_run:
        run_command(
            [
                sys.executable,
                "-B",
                "code/benchmarks/run_matcher_medical_one_click.py",
                "--out-dir",
                rel(expanded_50),
                "--idrid-cases",
                str(args.idrid_cases_50),
                "--monuseg-cases",
                str(args.monuseg_cases_50),
            ]
        )
    if not (manifest_50.exists() and rows_50.exists()):
        raise FileNotFoundError(f"Missing 50-case outputs under {expanded_50}")

    rows = [
        table_row("overall_20", out_dir / "overall_20", "gt-assisted accept/reject"),
        table_row("idrid_20", out_dir / "idrid_20", "gt-assisted accept/reject"),
        table_row("monuseg_20", out_dir / "monuseg_20", "gt-assisted accept/reject"),
        table_row("overall_50", expanded_50 / "casebook", "Dice threshold pseudo labels"),
    ]
    fields = [
        "setting",
        "cases",
        "proposals",
        "positive",
        "negative",
        "labels",
        "best_model",
        "best_feature",
        "stage3_f1",
        "learned_f1",
        "loco_f1",
        "stage3_dice_mean",
        "learned_dice_mean",
        "oracle_dice_mean",
        "policy_dir",
    ]
    write_csv(out_dir / "paper_main_table.csv", rows, fields)
    write_summary(out_dir, rows, labels_20, expanded_50)
    print(f"Wrote {rel(out_dir / 'paper_main_table.csv')}")


if __name__ == "__main__":
    main()
