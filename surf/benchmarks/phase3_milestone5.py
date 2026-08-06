from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "runs" / "phase3_milestone5"
DEFAULT_REPORT = WORKSPACE_ROOT / "docs" / "phase3_milestone5_acceptance.md"
AUTO_PROTOCOLS = ("auto_box", "auto_medical_box_v1", "auto_medical_point_v1")
MODELS = ("sam_vit_b", "medsam_vit_b", "sam2_tiny")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
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
    return float(sum(clean) / len(clean)) if clean else math.nan


def std(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_mask(path_value: str) -> np.ndarray | None:
    if not path_value:
        return None
    path = workspace_path(path_value)
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    else:
        arr = arr > 0
    return arr.astype(bool)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return math.nan
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def paired_bootstrap_delta(a: list[float], b: list[float], seed: int = 7, rounds: int = 5000) -> dict[str, float]:
    pairs = [(x, y) for x, y in zip(a, b) if not math.isnan(x) and not math.isnan(y)]
    if not pairs:
        return {"delta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_value": math.nan}
    diff = np.asarray([x - y for x, y in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[i] = float(diff[idx].mean())
    # Two-sided sign-flip p-value for paired deltas.
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(rounds, len(diff)))
    null = (signs * diff).mean(axis=1)
    p_value = float((np.abs(null) >= abs(float(diff.mean()))).mean())
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": p_value,
    }


def case_feature(row: dict[str, str]) -> np.ndarray:
    width = max(fnum(row["width"], 1.0), 1.0)
    height = max(fnum(row["height"], 1.0), 1.0)
    x0, y0, x1, y1 = (fnum(row[k], 0.0) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    return np.asarray(
        [
            fnum(row["target_area_ratio"], 0.0),
            ((x0 + x1) * 0.5) / width,
            ((y0 + y1) * 0.5) / height,
            bw / width,
            bh / height,
            math.log1p(fnum(row["component_count_approx"], 0.0)) / 8.0,
            math.log1p(bw / max(bh, 1.0)),
        ],
        dtype=np.float64,
    )


def build_case_index(manifest: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in manifest:
        rec = dict(row)
        rec["_feature"] = case_feature(row)
        index[row["case_id"]] = rec
    return index


def prepare_auto_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        if row.get("status") != "ok" or row.get("protocol") not in AUTO_PROTOCOLS:
            continue
        eval_area = max(fnum(row["eval_width"]) * fnum(row["eval_height"]), 1.0)
        rec: dict[str, Any] = dict(row)
        rec["dice_f"] = fnum(row["dice"])
        rec["iou_f"] = fnum(row["iou"])
        rec["score_f"] = fnum(row.get("score"), 0.0)
        rec["score_missing"] = 1.0 if math.isnan(fnum(row.get("score"))) else 0.0
        rec["pred_area_ratio"] = fnum(row["pred_area"], 0.0) / eval_area
        rec["_mask"] = load_mask(row.get("mask_path", ""))
        groups[(row["case_id"], row["model"])].append(rec)

    out: list[dict[str, Any]] = []
    for (_, _), rows in groups.items():
        masks = {row["protocol"]: row["_mask"] for row in rows}
        areas = [float(row["pred_area_ratio"]) for row in rows]
        median_area = float(np.median(areas)) if areas else 0.0
        for row in rows:
            ious = [mask_iou(row["_mask"], other_mask) for protocol, other_mask in masks.items() if protocol != row["protocol"]]
            clean = [v for v in ious if not math.isnan(v)]
            row["consensus_iou_mean"] = mean(clean)
            row["consensus_iou_min"] = min(clean) if clean else 0.0
            row["prompt_uncertainty"] = 1.0 - row["consensus_iou_mean"] if not math.isnan(row["consensus_iou_mean"]) else 1.0
            row["area_delta_to_median"] = abs(float(row["pred_area_ratio"]) - median_area)
            row.pop("_mask", None)
            out.append(row)
    return out


def nearest_supports(case_id: str, case_index: dict[str, dict[str, Any]], k: int) -> list[tuple[str, float]]:
    query = case_index[case_id]
    candidates: list[tuple[str, float]] = []
    for other_id, other in case_index.items():
        if other_id == case_id or other["dataset"] != query["dataset"]:
            continue
        dist = float(np.linalg.norm(query["_feature"] - other["_feature"]))
        candidates.append((other_id, dist))
    candidates.sort(key=lambda item: item[1])
    return candidates[:k]


def metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["case_id"], row["model"], row["protocol"]): row for row in rows}


def evaluate_support_guided(rows: list[dict[str, Any]], case_index: dict[str, dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    lookup = metric_lookup(rows)
    out: list[dict[str, Any]] = []
    for case_id in sorted({row["case_id"] for row in rows}):
        supports = nearest_supports(case_id, case_index, k)
        for model in MODELS:
            scored: list[tuple[float, str, float]] = []
            for protocol in AUTO_PROTOCOLS:
                vals: list[float] = []
                weights: list[float] = []
                for support_id, dist in supports:
                    support_row = lookup.get((support_id, model, protocol))
                    if support_row is None:
                        continue
                    weight = 1.0 / (1.0 + dist)
                    vals.append(fnum(support_row["dice_f"]) * weight)
                    weights.append(weight)
                support_score = sum(vals) / sum(weights) if weights else 0.0
                scored.append((support_score, protocol, mean([d for _, d in supports])))
            support_score, selected_protocol, support_distance = max(scored, key=lambda item: item[0])
            selected = lookup[(case_id, model, selected_protocol)]
            auto = lookup.get((case_id, model, "auto_box"))
            auto_dice = fnum(auto["dice_f"]) if auto is not None else math.nan
            out.append(
                {
                    "case_id": case_id,
                    "dataset": selected["dataset"],
                    "model": model,
                    "policy": "support_guided_prompt_v1",
                    "selected_protocol": selected_protocol,
                    "support_score": round(support_score, 6),
                    "support_distance_mean": round(support_distance, 6),
                    "support_case_ids": ";".join([sid for sid, _ in supports]),
                    "dice": round(fnum(selected["dice_f"]), 6),
                    "delta_vs_auto_box": round(fnum(selected["dice_f"]) - auto_dice, 6) if not math.isnan(auto_dice) else "nan",
                    "mask_path": selected.get("mask_path", ""),
                }
            )
    return out


def uncertainty_score(row: dict[str, Any], protocol_prior: float) -> float:
    stability = 1.0 - fnum(row["prompt_uncertainty"], 1.0)
    area_delta = fnum(row["area_delta_to_median"], 1.0)
    score = fnum(row["score_f"], 0.0)
    return 0.68 * protocol_prior + 0.18 * stability + 0.08 * score - 0.06 * min(area_delta, 1.0)


def evaluate_uncertainty_ensemble(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_model[(row["case_id"], row["model"])].append(row)
    out: list[dict[str, Any]] = []
    case_ids = sorted({row["case_id"] for row in rows})
    for case_id in case_ids:
        train = [row for row in rows if row["case_id"] != case_id]
        priors: dict[tuple[str, str, str], float] = {}
        for dataset in ("IDRID", "MonuSeg"):
            for model in MODELS:
                for protocol in AUTO_PROTOCOLS:
                    vals = [fnum(r["dice_f"]) for r in train if r["dataset"] == dataset and r["model"] == model and r["protocol"] == protocol]
                    priors[(dataset, model, protocol)] = mean(vals)
        for model in MODELS:
            candidates = by_case_model[(case_id, model)]
            scored = []
            for row in candidates:
                prior = priors.get((row["dataset"], model, row["protocol"]), mean([fnum(r["dice_f"]) for r in train]))
                scored.append((uncertainty_score(row, prior), row))
            predicted, selected = max(scored, key=lambda item: item[0])
            auto = next((r for r in candidates if r["protocol"] == "auto_box"), None)
            auto_dice = fnum(auto["dice_f"]) if auto is not None else math.nan
            out.append(
                {
                    "case_id": case_id,
                    "dataset": selected["dataset"],
                    "model": model,
                    "policy": "perturbation_uncertainty_proxy_v1",
                    "selected_protocol": selected["protocol"],
                    "predicted_reliability": round(predicted, 6),
                    "dice": round(fnum(selected["dice_f"]), 6),
                    "delta_vs_auto_box": round(fnum(selected["dice_f"]) - auto_dice, 6) if not math.isnan(auto_dice) else "nan",
                    "prompt_uncertainty": round(fnum(selected["prompt_uncertainty"]), 6),
                    "consensus_iou_mean": round(fnum(selected["consensus_iou_mean"]), 6),
                    "mask_path": selected.get("mask_path", ""),
                }
            )
    return out


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def portfolio_feature(row: dict[str, Any], support_rank: float) -> list[float]:
    return (
        [1.0]
        + one_hot(row["dataset"], ("IDRID", "MonuSeg"))
        + one_hot(row["model"], MODELS)
        + one_hot(row["protocol"], AUTO_PROTOCOLS)
        + [
            fnum(row["score_f"], 0.0),
            fnum(row["score_missing"], 0.0),
            fnum(row["pred_area_ratio"], 0.0),
            math.log1p(fnum(row["pred_area_ratio"], 0.0) * 100.0),
            fnum(row["consensus_iou_mean"], 0.0),
            fnum(row["consensus_iou_min"], 0.0),
            fnum(row["prompt_uncertainty"], 1.0),
            fnum(row["area_delta_to_median"], 0.0),
            support_rank,
        ]
    )


def fit_ridge(x_rows: list[list[float]], y_values: list[float], alpha: float = 1.2) -> np.ndarray:
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    reg = alpha * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def evaluate_cross_model_calibration(
    rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    selector_v3_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    support_choice = {(row["case_id"], row["model"], row["selected_protocol"]): 1.0 for row in support_rows}
    out: list[dict[str, Any]] = []
    case_ids = sorted(by_case)

    if selector_v3_rows:
        v3_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in selector_v3_rows:
            v3_by_case[row["case_id"]].append(row)
        metric_index = metric_lookup(rows)
        for case_id in case_ids:
            scored = []
            for row in v3_by_case.get(case_id, []):
                selected_protocol = row["selected_protocol"]
                support_rank = support_choice.get((case_id, row["model"], selected_protocol), 0.0)
                uncertainty = fnum(row.get("prompt_uncertainty"), 1.0)
                # v3 reliability already learned within-model prompt choice; v4 adds a
                # small cross-model calibration term from support agreement and stability.
                calibrated = fnum(row["predicted_reliability"], 0.0) + 0.018 * support_rank - 0.012 * uncertainty
                scored.append((calibrated, row, support_rank))
            if not scored:
                continue
            predicted, selected, support_rank = max(scored, key=lambda item: item[0])
            fixed = next((r for r in by_case[case_id] if r["model"] == "sam_vit_b" and r["protocol"] == "auto_box"), None)
            fixed_dice = fnum(fixed["dice_f"]) if fixed is not None else math.nan
            raw = metric_index.get((case_id, selected["model"], selected["selected_protocol"]))
            out.append(
                {
                    "case_id": case_id,
                    "dataset": selected["dataset"],
                    "policy": "cross_model_calibrated_portfolio_v4",
                    "selected_model": selected["model"],
                    "selected_protocol": selected["selected_protocol"],
                    "support_guided_match": int(support_rank),
                    "predicted_reliability": round(predicted, 6),
                    "dice": round(fnum(selected["dice"]), 6),
                    "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["dice"]) - fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                    "prompt_uncertainty": round(uncertainty, 6),
                    "mask_path": raw.get("mask_path", "") if raw else selected.get("mask_path", ""),
                }
            )
        return out

    for case_id in case_ids:
        train_rows = [row for row in rows if row["case_id"] != case_id]
        x_rows = [portfolio_feature(row, support_choice.get((row["case_id"], row["model"], row["protocol"]), 0.0)) for row in train_rows]
        y_values = [fnum(row["dice_f"]) for row in train_rows]
        weights = fit_ridge(x_rows, y_values)
        scored = []
        for row in by_case[case_id]:
            support_rank = support_choice.get((row["case_id"], row["model"], row["protocol"]), 0.0)
            predicted = float(np.asarray(portfolio_feature(row, support_rank), dtype=np.float64) @ weights)
            scored.append((predicted, row, support_rank))
        predicted, selected, support_rank = max(scored, key=lambda item: item[0])
        fixed = next((r for r in by_case[case_id] if r["model"] == "sam_vit_b" and r["protocol"] == "auto_box"), None)
        fixed_dice = fnum(fixed["dice_f"]) if fixed is not None else math.nan
        out.append(
            {
                "case_id": case_id,
                "dataset": selected["dataset"],
                "policy": "cross_model_calibrated_portfolio_v4",
                "selected_model": selected["model"],
                "selected_protocol": selected["protocol"],
                "support_guided_match": int(support_rank),
                "predicted_reliability": round(predicted, 6),
                "dice": round(fnum(selected["dice_f"]), 6),
                "delta_vs_fixed_sam_vit_b_auto_box": round(fnum(selected["dice_f"]) - fixed_dice, 6) if not math.isnan(fixed_dice) else "nan",
                "prompt_uncertainty": round(fnum(selected["prompt_uncertainty"]), 6),
                "mask_path": selected.get("mask_path", ""),
            }
        )
    return out


def summarize_policy(rows: list[dict[str, Any]], dice_key: str = "dice", delta_key: str | None = None) -> dict[str, Any]:
    dice = [fnum(row[dice_key]) for row in rows]
    out: dict[str, Any] = {
        "rows": len(rows),
        "mean_dice": round(mean(dice), 6),
        "std_dice": round(std(dice), 6),
    }
    if delta_key:
        delta = [fnum(row[delta_key]) for row in rows]
        out["mean_delta"] = round(mean(delta), 6)
        out["positive_rate"] = round(sum(1 for v in delta if v > 0) / max(len([v for v in delta if not math.isnan(v)]), 1), 6)
    return out


def summarize_by_keys(rows: list[dict[str, Any]], keys: list[str], delta_key: str | None = None) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[k]) for k in keys)].append(row)
    out = []
    for group, group_rows in sorted(buckets.items()):
        rec = {key: value for key, value in zip(keys, group)}
        rec.update(summarize_policy(group_rows, delta_key=delta_key))
        out.append(rec)
    return out


def fixed_baseline(rows: list[dict[str, Any]], model: str = "sam_vit_b", protocol: str = "auto_box") -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "dataset": row["dataset"],
            "model": model,
            "protocol": protocol,
            "dice": round(fnum(row["dice_f"]), 6),
        }
        for row in rows
        if row["model"] == model and row["protocol"] == protocol
    ]


def baseline_registry() -> list[dict[str, Any]]:
    ckpt = WORKSPACE_ROOT / "assets" / "checkpoints"
    entries = [
        ("SAM ViT-B", "sam_family", WORKSPACE_ROOT / "code" / "segment_anything", ckpt / "sam_vit_b.pth", "runnable", "Phase2-lite already evaluated."),
        ("MedSAM ViT-B", "medical_sam", WORKSPACE_ROOT / "MedSAM", ckpt / "medsam_vit_b.pth", "runnable", "Phase2-lite already evaluated."),
        ("SAM2 Tiny", "video_image_sam", WORKSPACE_ROOT / "sam2", ckpt / "sam2.1_hiera_tiny.pt", "runnable", "Phase2-lite already evaluated."),
        ("Matcher", "one_shot", WORKSPACE_ROOT / "Matcher", ckpt / "swint_only_sam_many2many.pth", "adapter_ready", "Workbench integration exists; benchmark adapter should export mask CSV."),
        ("SAM3", "text_concept_sam", WORKSPACE_ROOT / "sam3", ckpt / "sam3_modelscope" / "sam3.pt", "demo_ready", "Text/concept model; not a direct box/point strong medical baseline."),
        ("MedSAM2 / Medical SAM2", "medical_video_sam", WORKSPACE_ROOT / "MedSAM2", ckpt / "medsam2.pt", "missing", "Need repository, checkpoint, and dataset adapter."),
        ("SAM-Med2D", "medical_sam_adapter", WORKSPACE_ROOT / "SAM-Med2D", ckpt / "sam_med2d.pth", "missing", "High-priority strong baseline for 2D medical segmentation."),
        ("Med-SA / Medical SAM Adapter", "adapter", WORKSPACE_ROOT / "Med-SA", ckpt / "med_sa.pth", "missing", "High-priority adapter baseline."),
        ("nnU-Net", "supervised", WORKSPACE_ROOT / "nnUNet", WORKSPACE_ROOT / "runs" / "nnunet", "missing", "Requires training/evaluation split; strongest supervised reference."),
        ("U-Net", "supervised", WORKSPACE_ROOT / "code" / "benchmarks" / "unet_baseline.py", WORKSPACE_ROOT / "runs" / "unet", "missing", "Fast in-house supervised sanity baseline."),
        ("UniverSeg", "few_shot_medical", WORKSPACE_ROOT / "UniverSeg", ckpt / "universeg.pt", "missing", "Useful support-set baseline if code/checkpoint available."),
    ]
    rows = []
    for name, family, code_path, weight_path, planned_status, note in entries:
        code_exists = code_path.exists()
        weight_exists = weight_path.exists()
        if planned_status in {"runnable", "adapter_ready", "demo_ready"}:
            status = planned_status if code_exists and weight_exists else "partial"
        else:
            status = "available_unverified" if code_exists or weight_exists else "missing"
        rows.append(
            {
                "baseline": name,
                "family": family,
                "status": status,
                "code_path": str(code_path.relative_to(WORKSPACE_ROOT)) if code_path.exists() or str(code_path).startswith(str(WORKSPACE_ROOT)) else str(code_path),
                "code_exists": int(code_exists),
                "weight_path": str(weight_path.relative_to(WORKSPACE_ROOT)) if str(weight_path).startswith(str(WORKSPACE_ROOT)) else str(weight_path),
                "weight_exists": int(weight_exists),
                "phase3_priority": "P0" if name in {"MedSAM2 / Medical SAM2", "SAM-Med2D", "Med-SA / Medical SAM Adapter", "nnU-Net"} else "P1",
                "note": note,
            }
        )
    return rows


def write_report(
    path: Path,
    support_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
    portfolio_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    support_summary = summarize_policy(support_rows, delta_key="delta_vs_auto_box")
    uncertainty_summary = summarize_policy(uncertainty_rows, delta_key="delta_vs_auto_box")
    portfolio_summary = summarize_policy(portfolio_rows, delta_key="delta_vs_fixed_sam_vit_b_auto_box")
    fixed_summary = summarize_policy(fixed_rows)
    portfolio_sig = paired_bootstrap_delta(
        [fnum(row["dice"]) for row in portfolio_rows],
        [fnum(row["dice"]) for row in fixed_rows],
    )
    runnable = [row for row in registry_rows if row["status"] in {"runnable", "adapter_ready", "demo_ready"}]
    p0_missing = [row for row in registry_rows if row["phase3_priority"] == "P0" and row["status"] == "missing"]

    text = f"""# Phase3 第五验收标准推进报告

## 1. 本次目标

按 `docs/phase3_topconf_upgrade_plan.md` 的近期执行路线推进到第五个验收标准：

1. Support Prompt v1
2. Perturbation Uncertainty v1
3. Cross-model Calibration
4. 扩数据准备
5. 强基线比较

这次没有伪造任何未运行模型的指标。前 3 项基于已有 Phase2-lite 预测结果做可复现的二次评估；第 5 项给出本地强基线接入审计，区分“已经可跑”和“仍需接入”。

## 2. 已生成文件

- `{output_root.relative_to(WORKSPACE_ROOT) / 'support_guided_prompt_v1.csv'}`
- `{output_root.relative_to(WORKSPACE_ROOT) / 'perturbation_uncertainty_proxy_v1.csv'}`
- `{output_root.relative_to(WORKSPACE_ROOT) / 'cross_model_calibrated_portfolio_v4.csv'}`
- `{output_root.relative_to(WORKSPACE_ROOT) / 'strong_baseline_registry.csv'}`
- `{output_root.relative_to(WORKSPACE_ROOT) / 'summary_by_policy.csv'}`
- `{output_root.relative_to(WORKSPACE_ROOT) / 'summary_by_dataset.csv'}`

## 3. 方法链条结果

| 方法 | rows | Mean Dice | Mean Delta |
| --- | ---: | ---: | ---: |
| fixed SAM ViT-B auto_box | {fixed_summary['rows']} | {fixed_summary['mean_dice']:.4f} | - |
| support_guided_prompt_v1 | {support_summary['rows']} | {support_summary['mean_dice']:.4f} | {support_summary['mean_delta']:.4f} vs per-model auto_box |
| perturbation_uncertainty_proxy_v1 | {uncertainty_summary['rows']} | {uncertainty_summary['mean_dice']:.4f} | {uncertainty_summary['mean_delta']:.4f} vs per-model auto_box |
| cross_model_calibrated_portfolio_v4 | {portfolio_summary['rows']} | {portfolio_summary['mean_dice']:.4f} | {portfolio_summary['mean_delta']:.4f} vs fixed SAM ViT-B auto_box |

portfolio_v4 paired bootstrap:

```text
mean delta = {portfolio_sig['delta']:.4f}
95% CI     = [{portfolio_sig['ci_low']:.4f}, {portfolio_sig['ci_high']:.4f}]
p-value    = {portfolio_sig['p_value']:.4f}
```

## 4. 五个验收标准状态

| 验收项 | 状态 | 结论 |
| --- | --- | --- |
| Step 1 Support Prompt v1 | 已完成可复现评估 | 用同域 support case 的历史表现选择 query prompt，验证 support 信息是否能带来互补性。 |
| Step 2 Perturbation Uncertainty v1 | 已完成 proxy 版 | 当前基于 auto-family mask consensus 近似 prompt-induced uncertainty；真正 jitter 重跑版仍是下一步。 |
| Step 3 Cross-model Calibration | 已完成 v4 | 已从单模型 selector 扩展到 SAM / MedSAM / SAM2 的跨模型 portfolio 选择。 |
| Step 4 扩数据准备 | 部分完成 | 代码仍基于 IDRID + MonuSeg 40 cases；扩到 100-200 cases、3+ domains 仍未完成。 |
| Step 5 强基线比较 | 工程验收完成，科学验收未完成 | 已生成强基线 registry；目前只有 SAM ViT-B / MedSAM / SAM2 Tiny 有 Phase2-lite 指标，P0 强基线仍需下载/接入/运行。 |

## 5. 强基线接入审计

当前本地可直接进入比较或接入的条目数：{len(runnable)} / {len(registry_rows)}。

P0 仍缺失：

{chr(10).join([f"- {row['baseline']}: {row['note']}" for row in p0_missing]) if p0_missing else "- 无"}

## 6. 诚实判断

我们已经达到“第五个验收标准”的工程版本：知道强基线在哪里、哪些已经有结果、哪些必须补齐，并且有了从 support 到 uncertainty 再到 cross-model calibration 的可复现实验链条。

但还没有达到顶会论文所需的科学版本。现在还不能写“超过近几年强医学分割基线”。下一步真正要做的是：

1. 接入并运行 SAM-Med2D / Med-SA / MedSAM2 中至少两个 P0 基线。
2. 加入一个轻量 U-Net 或 nnU-Net supervised baseline。
3. 把当前 40 cases 扩到至少 100-200 cases、3 个医学域。
4. 把 uncertainty proxy 升级成真实 prompt jitter 重跑版。
5. 用同一 manifest、同一 prompt、同一 metric 输出最终主表。

## 7. 当前可用 claim

目前可以保守写：

```text
OCSAM-X 将 support-guided prompt selection、prompt-induced uncertainty proxy
和 cross-model reliability calibration 串成了一个可复现的医学 SAM 自动提示适配框架。
在 Phase2-lite 上，它已经形成了比固定自动 prompt 更完整的候选选择闭环，
但仍需要更大数据和强医学基线来支撑顶会级 SOTA claim。
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advance OCSAM-X Phase3 to milestone 5 acceptance artifacts.")
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--support-k", type=int, default=5)
    args = parser.parse_args()

    manifest = read_csv(args.phase2_root / "manifests" / "phase2_lite_40_cases.csv")
    metrics = read_csv(args.phase2_root / "metrics" / "phase2_lite_metrics.csv")
    case_index = build_case_index(manifest)
    auto_rows = prepare_auto_rows(metrics)

    support_rows = evaluate_support_guided(auto_rows, case_index, k=args.support_k)
    uncertainty_rows = evaluate_uncertainty_ensemble(auto_rows)
    selector_v3_path = args.phase2_root / "analysis" / "prompt_selector_v3_results.csv"
    selector_v3_rows = read_csv(selector_v3_path) if selector_v3_path.exists() else None
    portfolio_rows = evaluate_cross_model_calibration(auto_rows, support_rows, selector_v3_rows)
    fixed_rows = fixed_baseline(auto_rows, model="sam_vit_b", protocol="auto_box")
    registry_rows = baseline_registry()

    write_csv(args.output_root / "support_guided_prompt_v1.csv", support_rows)
    write_csv(args.output_root / "perturbation_uncertainty_proxy_v1.csv", uncertainty_rows)
    write_csv(args.output_root / "cross_model_calibrated_portfolio_v4.csv", portfolio_rows)
    write_csv(args.output_root / "strong_baseline_registry.csv", registry_rows)

    summary_rows = [
        {"policy": "fixed_sam_vit_b_auto_box", **summarize_policy(fixed_rows)},
        {"policy": "support_guided_prompt_v1", **summarize_policy(support_rows, delta_key="delta_vs_auto_box")},
        {"policy": "perturbation_uncertainty_proxy_v1", **summarize_policy(uncertainty_rows, delta_key="delta_vs_auto_box")},
        {"policy": "cross_model_calibrated_portfolio_v4", **summarize_policy(portfolio_rows, delta_key="delta_vs_fixed_sam_vit_b_auto_box")},
    ]
    write_csv(args.output_root / "summary_by_policy.csv", summary_rows)
    write_csv(args.output_root / "summary_by_dataset.csv", summarize_by_keys(portfolio_rows, ["dataset"], delta_key="delta_vs_fixed_sam_vit_b_auto_box"))

    write_report(args.report, support_rows, uncertainty_rows, portfolio_rows, fixed_rows, registry_rows, args.output_root)
    print(f"Wrote {args.report}")
    print(f"Wrote outputs under {args.output_root}")


if __name__ == "__main__":
    main()
