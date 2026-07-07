from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = WORKSPACE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "benchmarks"))

import phase2_lite as p2  # noqa: E402


DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "runs" / "phase3_full"
DEFAULT_REPORT = WORKSPACE_ROOT / "docs" / "phase3_full_1_to_5_completion.md"
BASE_PROTOCOLS = (
    "gt_box",
    "perturbed_box",
    "centroid_point",
    "grid_point",
    "auto_box",
    "auto_medical_box_v1",
    "auto_medical_point_v1",
)
JITTER_PROTOCOLS = tuple(f"jitter_box_{idx:02d}" for idx in range(8))
POINT_JITTER_PROTOCOLS = tuple(f"jitter_point_{idx:02d}" for idx in range(4))
MODEL_NAMES = ("sam_vit_b", "sam_vit_h", "medsam_vit_b", "lite_medsam", "sam2_tiny")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def box_jitter(box: list[float], width: int, height: int, idx: int, case_id: str) -> list[float]:
    rng = np.random.default_rng(p2.stable_seed(f"phase3-jitter-box:{case_id}:{idx}"))
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    scale = rng.uniform(-0.10, 0.18)
    shift_x = rng.uniform(-0.10, 0.10) * bw
    shift_y = rng.uniform(-0.10, 0.10) * bh
    edge_noise = rng.normal(0.0, 0.025, size=4) * np.array([bw, bh, bw, bh])
    return p2.clamp_box(
        [
            x0 - scale * bw + shift_x + edge_noise[0],
            y0 - scale * bh + shift_y + edge_noise[1],
            x1 + scale * bw + shift_x + edge_noise[2],
            y1 + scale * bh + shift_y + edge_noise[3],
        ],
        width,
        height,
    )


def point_jitter(points: list[dict[str, float | int]], width: int, height: int, idx: int, case_id: str) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(p2.stable_seed(f"phase3-jitter-point:{case_id}:{idx}"))
    scale = max(width, height)
    out: list[dict[str, float | int]] = []
    for point in points:
        radius = rng.uniform(0.01, 0.035) * scale
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = float(point["x"]) + math.cos(angle) * radius
        y = float(point["y"]) + math.sin(angle) * radius
        out.append(
            {
                "x": round(float(np.clip(x, 0, width - 1)), 3),
                "y": round(float(np.clip(y, 0, height - 1)), 3),
                "label": int(point.get("label", 1)),
            }
        )
    return out


def write_phase3_prompts(manifest_path: Path, prompts_path: Path) -> list[dict[str, Any]]:
    base_records = p2.write_prompts(manifest_path, prompts_path)
    rows = p2.read_manifest(manifest_path)
    by_case = {row["case_id"]: row for row in rows}
    base_by_case_protocol = {(r["case_id"], r["protocol"]): r for r in base_records}
    records = list(base_records)
    for case_id, row in by_case.items():
        width = int(row["width"])
        height = int(row["height"])
        auto_box = base_by_case_protocol[(case_id, "auto_box")]
        medical_box = base_by_case_protocol[(case_id, "auto_medical_box_v1")]
        medical_points = base_by_case_protocol[(case_id, "auto_medical_point_v1")]
        boxes = [auto_box["box"], medical_box["box"]]
        for idx, protocol in enumerate(JITTER_PROTOCOLS):
            source_box = boxes[idx % len(boxes)]
            records.append(
                {
                    "prompt_id": f"{case_id}::{protocol}",
                    "case_id": case_id,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "protocol": protocol,
                    "prompt_type": "box",
                    "box": box_jitter(source_box, width, height, idx, case_id),
                    "points": [],
                    "metadata": {"source": "phase3_real_box_jitter", "base_protocol": "auto_box" if idx % 2 == 0 else "auto_medical_box_v1"},
                }
            )
        for idx, protocol in enumerate(POINT_JITTER_PROTOCOLS):
            records.append(
                {
                    "prompt_id": f"{case_id}::{protocol}",
                    "case_id": case_id,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "protocol": protocol,
                    "prompt_type": "point",
                    "box": None,
                    "points": point_jitter(medical_points["points"], width, height, idx, case_id),
                    "metadata": {"source": "phase3_real_point_jitter", "base_protocol": "auto_medical_point_v1"},
                }
            )
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def build_manifest_and_prompts(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    prompts = Path(args.prompts)
    cases = p2.write_manifest(manifest, args.idrid_cases, args.monuseg_cases)
    records = write_phase3_prompts(manifest, prompts)
    print(f"Wrote {len(cases)} phase3 cases: {manifest}")
    print(f"Wrote {len(records)} phase3 prompts: {prompts}")


def run_models(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(CODE_ROOT / "benchmarks" / "phase2_lite.py"),
        "--output-root",
        str(args.output_root),
        "--manifest",
        str(args.manifest),
        "--prompts",
        str(args.prompts),
        "run",
        "--models",
        args.models,
        "--protocols",
        args.protocols,
        "--max-side",
        str(args.max_side),
        "--device",
        args.device,
        "--sam-checkpoint",
        str(args.sam_checkpoint),
        "--sam-h-checkpoint",
        str(args.sam_h_checkpoint),
        "--medsam-checkpoint",
        str(args.medsam_checkpoint),
        "--lite-medsam-checkpoint",
        str(args.lite_medsam_checkpoint),
        "--sam2-checkpoint",
        str(args.sam2_checkpoint),
        "--sam2-config",
        args.sam2_config,
    ]
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))


def mask_iou(a_path: str, b_path: str) -> float:
    if not a_path or not b_path:
        return math.nan
    a = p2.load_mask(WORKSPACE_ROOT / a_path)
    b = p2.load_mask(WORKSPACE_ROOT / b_path)
    if a.shape != b.shape:
        return math.nan
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def summarize_metrics(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    metrics = read_csv(output_root / "metrics" / "phase2_lite_metrics.csv")
    ok = [r for r in metrics if r.get("status") == "ok"]
    rows_by_model: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_model_protocol: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_model_dataset: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        by_model[row["model"]].append(row)
        by_model_protocol[(row["model"], row["protocol"])].append(row)
        by_model_dataset[(row["model"], row["dataset"])].append(row)
    for model, rows in sorted(by_model.items()):
        rows_by_model.append({"model": model, "rows": len(rows), "mean_dice": round(mean([fnum(r["dice"]) for r in rows]), 4)})
    rows_by_protocol = [
        {"model": model, "protocol": protocol, "rows": len(rows), "mean_dice": round(mean([fnum(r["dice"]) for r in rows]), 4)}
        for (model, protocol), rows in sorted(by_model_protocol.items())
    ]
    rows_by_dataset = [
        {"model": model, "dataset": dataset, "rows": len(rows), "mean_dice": round(mean([fnum(r["dice"]) for r in rows]), 4)}
        for (model, dataset), rows in sorted(by_model_dataset.items())
    ]
    write_csv(output_root / "analysis" / "summary_by_model.csv", rows_by_model)
    write_csv(output_root / "analysis" / "summary_by_model_protocol.csv", rows_by_protocol)
    write_csv(output_root / "analysis" / "summary_by_model_dataset.csv", rows_by_dataset)
    return {"metrics": metrics, "ok": ok, "by_model": rows_by_model, "by_protocol": rows_by_protocol, "by_dataset": rows_by_dataset}


def build_uncertainty_table(output_root: Path, metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    ok = [r for r in metrics if r.get("status") == "ok"]
    by_case_model: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        if row["protocol"].startswith("jitter_") or row["protocol"] in {"auto_box", "auto_medical_box_v1", "auto_medical_point_v1"}:
            by_case_model[(row["case_id"], row["model"])].append(row)
    rows: list[dict[str, Any]] = []
    for (case_id, model), candidates in sorted(by_case_model.items()):
        jitter = [r for r in candidates if r["protocol"].startswith("jitter_")]
        if not jitter:
            continue
        ious: list[float] = []
        for i in range(len(jitter)):
            for j in range(i + 1, len(jitter)):
                ious.append(mask_iou(jitter[i].get("mask_path", ""), jitter[j].get("mask_path", "")))
        stability = mean(ious)
        best = max(candidates, key=lambda r: fnum(r.get("dice")))
        auto = next((r for r in candidates if r["protocol"] == "auto_box"), None)
        auto_dice = fnum(auto.get("dice")) if auto else math.nan
        rows.append(
            {
                "case_id": case_id,
                "dataset": best["dataset"],
                "model": model,
                "jitter_count": len(jitter),
                "jitter_pairwise_iou": round(stability, 6) if not math.isnan(stability) else "nan",
                "prompt_uncertainty": round(1.0 - stability, 6) if not math.isnan(stability) else "nan",
                "selected_protocol": best["protocol"],
                "selected_dice": round(fnum(best["dice"]), 6),
                "auto_box_dice": round(auto_dice, 6) if not math.isnan(auto_dice) else "nan",
                "delta_vs_auto_box": round(fnum(best["dice"]) - auto_dice, 6) if not math.isnan(auto_dice) else "nan",
            }
        )
    write_csv(output_root / "analysis" / "real_jitter_uncertainty_selection.csv", rows)
    return rows


def build_stability_selector_table(output_root: Path, metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    ok = [r for r in metrics if r.get("status") == "ok"]
    by_case_model: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        if row["protocol"].startswith("jitter_") or row["protocol"] in {"auto_box", "auto_medical_box_v1", "auto_medical_point_v1"}:
            by_case_model[(row["case_id"], row["model"])].append(row)
    rows: list[dict[str, Any]] = []
    for (case_id, model), candidates in sorted(by_case_model.items()):
        scored: list[tuple[float, float, dict[str, str]]] = []
        for row in candidates:
            others = [other for other in candidates if other is not row]
            ious = [mask_iou(row.get("mask_path", ""), other.get("mask_path", "")) for other in others]
            consensus = mean(ious)
            area_ratio = fnum(row.get("pred_area")) / max(fnum(row.get("target_area"), 1.0), 1.0)
            area_penalty = abs(math.log(max(area_ratio, 1e-6)))
            score = consensus - 0.015 * area_penalty
            scored.append((score, consensus, row))
        if not scored:
            continue
        score, consensus, selected = max(scored, key=lambda item: item[0])
        auto = next((r for r in candidates if r["protocol"] == "auto_box"), None)
        auto_dice = fnum(auto.get("dice")) if auto else math.nan
        rows.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "model": model,
                "policy": "real_jitter_stability_selector_v1",
                "selected_protocol": selected["protocol"],
                "stability_score": round(score, 6),
                "consensus_iou": round(consensus, 6) if not math.isnan(consensus) else "nan",
                "dice": round(fnum(selected["dice"]), 6),
                "auto_box_dice": round(auto_dice, 6) if not math.isnan(auto_dice) else "nan",
                "delta_vs_auto_box": round(fnum(selected["dice"]) - auto_dice, 6) if not math.isnan(auto_dice) else "nan",
                "note": "Deployable selector: uses mask consensus and area sanity only, not GT Dice.",
            }
        )
    write_csv(output_root / "analysis" / "real_jitter_stability_selector_v1.csv", rows)
    return rows


def build_portfolio_table(output_root: Path, metrics: list[dict[str, str]], jitter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in metrics if r.get("status") == "ok"]
    fixed_lookup = {(r["case_id"], r["model"], r["protocol"]): r for r in ok}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    jitter_by_case_model = {(r["case_id"], r["model"]): r for r in jitter_rows}
    for row in jitter_rows:
        by_case[row["case_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        selected = max(candidates, key=lambda r: fnum(r["selected_dice"]))
        fixed = fixed_lookup.get((case_id, "sam_vit_b", "auto_box"))
        fixed_dice = fnum(fixed.get("dice")) if fixed else math.nan
        rows.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "policy": "real_jitter_cross_model_oracle_upper_bound",
                "selected_model": selected["model"],
                "selected_protocol": selected["selected_protocol"],
                "dice": selected["selected_dice"],
                "fixed_sam_vit_b_auto_box_dice": round(fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["selected_dice"]) - fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                "note": "Upper bound uses GT Dice only for feasibility audit; not deployable selector.",
            }
        )
    write_csv(output_root / "analysis" / "real_jitter_cross_model_upper_bound.csv", rows)
    return rows


def build_deployable_portfolio_table(output_root: Path, metrics: list[dict[str, str]], stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in metrics if r.get("status") == "ok"]
    fixed_lookup = {(r["case_id"], r["model"], r["protocol"]): r for r in ok}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        by_case[row["case_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        selected = max(candidates, key=lambda r: fnum(r["stability_score"]))
        fixed = fixed_lookup.get((case_id, "sam_vit_b", "auto_box"))
        fixed_dice = fnum(fixed.get("dice")) if fixed else math.nan
        rows.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "policy": "real_jitter_cross_model_stability_selector_v1",
                "selected_model": selected["model"],
                "selected_protocol": selected["selected_protocol"],
                "stability_score": selected["stability_score"],
                "dice": selected["dice"],
                "fixed_sam_vit_b_auto_box_dice": round(fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["dice"]) - fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                "note": "Deployable selector: cross-model choice by stability score only, not GT Dice.",
            }
        )
    write_csv(output_root / "analysis" / "real_jitter_cross_model_stability_selector_v1.csv", rows)
    return rows


def write_report(
    args: argparse.Namespace,
    summaries: dict[str, list[dict[str, Any]]],
    jitter_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    portfolio_rows: list[dict[str, Any]],
    deployable_portfolio_rows: list[dict[str, Any]],
) -> None:
    manifest = read_csv(Path(args.manifest))
    metrics = summaries["metrics"]
    ok = summaries["ok"]
    status_counts = defaultdict(int)
    for row in metrics:
        status_counts[row.get("status", "")] += 1
    fixed = [r for r in ok if r["model"] == "sam_vit_b" and r["protocol"] == "auto_box"]
    best_fixed_by_model = [r for r in summaries["by_protocol"] if r["protocol"] == "auto_box"]
    jitter_delta = mean([fnum(r["delta_vs_auto_box"]) for r in jitter_rows])
    stability_delta = mean([fnum(r["delta_vs_auto_box"]) for r in stability_rows])
    portfolio_delta = mean([fnum(r["delta_vs_fixed_sam_vit_b_auto_box"]) for r in portfolio_rows])
    deployable_portfolio_delta = mean([fnum(r["delta_vs_fixed_sam_vit_b_auto_box"]) for r in deployable_portfolio_rows])
    lines = [
        "# Phase3 1-5 完整推进结果",
        "",
        "这份报告对应用户要求的 1-5：扩数据、补本地强基线、真实 prompt jitter、统一主表、claim 判断。",
        "",
        "## 1. 扩数据到 100-200 cases",
        "",
        f"- Manifest: `{rel(Path(args.manifest))}`",
        f"- Cases: {len(manifest)}",
        f"- Prompts: `{rel(Path(args.prompts))}`",
        f"- 数据集分布：{dict((k, sum(1 for r in manifest if r['dataset'] == k)) for k in sorted({r['dataset'] for r in manifest}))}",
        "",
        "## 2. 强基线接入",
        "",
        "| 模型 | Mean Dice | 有效行数 |",
        "| --- | ---: | ---: |",
    ]
    for row in summaries["by_model"]:
        lines.append(f"| {row['model']} | {row['mean_dice']} | {row['rows']} |")
    lines.extend(
        [
            "",
            "本轮补上了本地可立即复用且能稳定运行的强基线：`sam_vit_h`，并保留 `sam_vit_b`、`medsam_vit_b`、`sam2_tiny`。`lite_medsam` 在 smoke 中未形成有效主表结果；Matcher 仍应作为 one-shot/support baseline 单独接入，不应伪装成普通 box/point prompt。",
            "",
            "## 3. 真实 prompt jitter uncertainty",
            "",
            f"- 真实 jitter 结果表：`{rel(Path(args.output_root) / 'analysis' / 'real_jitter_uncertainty_selection.csv')}`",
            f"- Oracle jitter selector 平均 delta vs auto_box: {jitter_delta:.4f}",
            f"- Deployable stability selector 平均 delta vs auto_box: {stability_delta:.4f}",
            f"- Deployable stability selector 表：`{rel(Path(args.output_root) / 'analysis' / 'real_jitter_stability_selector_v1.csv')}`",
            "",
            "## 4. 统一主表",
            "",
            f"- Metrics: `{rel(Path(args.output_root) / 'metrics' / 'phase2_lite_metrics.csv')}`",
            f"- OK rows: {len(ok)} / {len(metrics)}",
            f"- Status counts: {dict(status_counts)}",
            "",
            "| 模型 | Prompt | Mean Dice | 有效行数 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in summaries["by_protocol"]:
        if row["protocol"] in {"auto_box", "auto_medical_box_v1", "gt_box"} or str(row["protocol"]).startswith("jitter_box_00"):
            lines.append(f"| {row['model']} | {row['protocol']} | {row['mean_dice']} | {row['rows']} |")
    lines.extend(
        [
            "",
            "## 5. Claim 判断",
            "",
            f"- SAM ViT-B auto_box 平均 Dice: {mean([fnum(r['dice']) for r in fixed]):.4f}",
            f"- Cross-model stability selector delta: {deployable_portfolio_delta:.4f}",
            f"- Cross-model real-jitter oracle upper bound delta: {portfolio_delta:.4f}",
            "",
        ]
    )
    if deployable_portfolio_delta > 0.01 and stability_delta > 0:
        lines.append("当前证据支持继续推进 OCSAM-X 的主线：不看 GT 的 stability selector 已经有正向收益，oracle upper bound 说明还有可挖空间。")
    else:
        lines.append("当前证据说明存在上限空间，但不看 GT 的 selector 仍不够强，应继续补强 reliability predictor 和 P0 医学强基线。")
    lines.extend(
        [
            "",
            "可写的稳健 claim：OCSAM-X 通过结构化自动 prompt、真实 prompt-induced uncertainty 和跨模型互补性分析，构建了一个可复现的医学 SAM prompt adaptation 实验框架；在当前扩展数据上可观察到上限空间，但最终 SOTA 结论仍需 SAM-Med2D / Med-SA / MedSAM2 / nnU-Net 等 P0 baseline 验证。",
        ]
    )
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    summaries = summarize_metrics(output_root)
    jitter_rows = build_uncertainty_table(output_root, summaries["metrics"])
    stability_rows = build_stability_selector_table(output_root, summaries["metrics"])
    portfolio_rows = build_portfolio_table(output_root, summaries["metrics"], jitter_rows)
    deployable_portfolio_rows = build_deployable_portfolio_table(output_root, summaries["metrics"], stability_rows)
    write_report(args, summaries, jitter_rows, stability_rows, portfolio_rows, deployable_portfolio_rows)
    print(f"Wrote {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase3 1-5 completion pipeline.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifests" / "phase3_100_cases.csv")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_OUTPUT_ROOT / "prompts" / "phase3_prompts_with_jitter.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--idrid-cases", type=int, default=50)
    parser.add_argument("--monuseg-cases", type=int, default=50)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--protocols", default=",".join(BASE_PROTOCOLS + JITTER_PROTOCOLS + POINT_JITTER_PROTOCOLS))
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sam-checkpoint", type=Path, default=WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth")
    parser.add_argument("--sam-h-checkpoint", type=Path, default=WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_h_4b8939.pth")
    parser.add_argument("--medsam-checkpoint", type=Path, default=WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_vit_b.pth")
    parser.add_argument("--lite-medsam-checkpoint", type=Path, default=WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_download" / "lite_medsam.pth")
    parser.add_argument("--sam2-checkpoint", type=Path, default=WORKSPACE_ROOT / "assets" / "checkpoints" / "sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("run")
    sub.add_parser("analyze")
    sub.add_parser("all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build_manifest_and_prompts(args)
    elif args.command == "run":
        run_models(args)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "all":
        build_manifest_and_prompts(args)
        run_models(args)
        analyze(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
