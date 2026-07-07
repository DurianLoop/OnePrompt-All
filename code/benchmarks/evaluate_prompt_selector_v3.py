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
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"
AUTO_PROTOCOLS = ("auto_box", "auto_medical_box_v1", "auto_medical_point_v1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def fnum(value: str | float | int) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def load_mask(mask_path: str) -> np.ndarray | None:
    if not mask_path:
        return None
    path = workspace_path(mask_path)
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


def enrich_auto_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        if row["status"] != "ok" or row["protocol"] not in AUTO_PROTOCOLS:
            continue
        eval_area = max(fnum(row["eval_width"]) * fnum(row["eval_height"]), 1.0)
        enriched = dict(row)
        enriched["pred_area_ratio"] = fnum(row["pred_area"]) / eval_area
        enriched["score_value"] = fnum(row.get("score", "nan"))
        enriched["score_missing"] = 1.0 if math.isnan(enriched["score_value"]) else 0.0
        if math.isnan(enriched["score_value"]):
            enriched["score_value"] = 0.0
        enriched["_mask"] = load_mask(row.get("mask_path", ""))
        grouped[(row["case_id"], row["model"])].append(enriched)

    rows: list[dict[str, Any]] = []
    for _, candidates in grouped.items():
        areas = [float(row["pred_area_ratio"]) for row in candidates]
        median_area = float(np.median(areas)) if areas else math.nan
        auto_area = next((float(row["pred_area_ratio"]) for row in candidates if row["protocol"] == "auto_box"), median_area)
        enriched_candidates: list[dict[str, Any]] = []
        for row in candidates:
            ious = [mask_iou(row["_mask"], other["_mask"]) for other in candidates if other is not row]
            clean_ious = [v for v in ious if not math.isnan(v)]
            consensus_mean = mean(clean_ious)
            consensus_min = min(clean_ious) if clean_ious else math.nan
            consensus_max = max(clean_ious) if clean_ious else math.nan
            row["consensus_iou_mean"] = consensus_mean
            row["consensus_iou_min"] = consensus_min
            row["consensus_iou_max"] = consensus_max
            row["prompt_uncertainty"] = 1.0 - consensus_mean if not math.isnan(consensus_mean) else 1.0
            row["area_delta_to_median"] = abs(float(row["pred_area_ratio"]) - median_area)
            row["area_delta_to_auto"] = abs(float(row["pred_area_ratio"]) - auto_area)
            enriched_candidates.append(row)
        for row in enriched_candidates:
            row.pop("_mask", None)
            rows.append(row)
    return rows


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def feature_vector(row: dict[str, Any]) -> list[float]:
    pred_area_ratio = float(row["pred_area_ratio"])
    score = float(row["score_value"])
    consensus = fnum(row["consensus_iou_mean"])
    uncertainty = fnum(row["prompt_uncertainty"])
    if math.isnan(consensus):
        consensus = 0.0
    if math.isnan(uncertainty):
        uncertainty = 1.0
    return (
        [1.0]
        + one_hot(row["dataset"], ("IDRID", "MonuSeg"))
        + one_hot(row["model"], ("sam_vit_b", "medsam_vit_b", "sam2_tiny"))
        + one_hot(row["protocol"], AUTO_PROTOCOLS)
        + [
            score,
            float(row["score_missing"]),
            pred_area_ratio,
            math.log1p(pred_area_ratio * 100.0),
            1.0 if pred_area_ratio < 0.01 else 0.0,
            1.0 if pred_area_ratio > 0.40 else 0.0,
            consensus,
            fnum(row["consensus_iou_min"]) if not math.isnan(fnum(row["consensus_iou_min"])) else 0.0,
            uncertainty,
            float(row["area_delta_to_median"]),
            float(row["area_delta_to_auto"]),
        ]
    )


def fit_ridge(train_rows: list[dict[str, Any]], alpha: float = 0.9) -> np.ndarray:
    x = np.asarray([feature_vector(row) for row in train_rows], dtype=np.float64)
    y = np.asarray([fnum(row["dice"]) for row in train_rows], dtype=np.float64)
    reg = alpha * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def protocol_prior(train_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in train_rows:
        buckets[(row["dataset"], row["model"], row["protocol"])].append(fnum(row["dice"]))
    return {key: mean(values) for key, values in buckets.items()}


def predict_reliability(row: dict[str, Any], weights: np.ndarray, priors: dict[tuple[str, str, str], float], global_mean: float) -> float:
    linear = float(np.asarray(feature_vector(row), dtype=np.float64) @ weights)
    prior = priors.get((row["dataset"], row["model"], row["protocol"]), global_mean)
    uncertainty = fnum(row["prompt_uncertainty"])
    stability_bonus = 0.025 * (1.0 - uncertainty) if not math.isnan(uncertainty) else 0.0
    area_penalty = 0.018 * min(float(row["area_delta_to_median"]), 1.0)
    return 0.54 * prior + 0.42 * linear + stability_bonus - area_penalty


def evaluate_selector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_model[(row["case_id"], row["model"])].append(row)

    selected: list[dict[str, Any]] = []
    all_case_ids = sorted({row["case_id"] for row in rows})
    for case_id in all_case_ids:
        train_rows = [row for row in rows if row["case_id"] != case_id]
        weights = fit_ridge(train_rows)
        priors = protocol_prior(train_rows)
        global_mean = mean([fnum(row["dice"]) for row in train_rows])
        for (candidate_case, model), candidates in sorted(by_case_model.items()):
            if candidate_case != case_id:
                continue
            scored = [(predict_reliability(row, weights, priors, global_mean), row) for row in candidates]
            predicted, best_row = max(scored, key=lambda item: item[0])
            auto_box = next((row for row in candidates if row["protocol"] == "auto_box"), None)
            auto_box_dice = fnum(auto_box["dice"]) if auto_box is not None else math.nan
            selected.append(
                {
                    "case_id": case_id,
                    "dataset": best_row["dataset"],
                    "model": model,
                    "policy": "uncertainty_reliability_selector_v3",
                    "selected_protocol": best_row["protocol"],
                    "predicted_reliability": round(predicted, 6),
                    "dice": round(fnum(best_row["dice"]), 6),
                    "delta_vs_auto_box": round(fnum(best_row["dice"]) - auto_box_dice, 6) if not math.isnan(auto_box_dice) else "nan",
                    "score": round(float(best_row["score_value"]), 6),
                    "pred_area_ratio": round(float(best_row["pred_area_ratio"]), 6),
                    "consensus_iou_mean": round(fnum(best_row["consensus_iou_mean"]), 6),
                    "prompt_uncertainty": round(fnum(best_row["prompt_uncertainty"]), 6),
                    "mask_path": best_row.get("mask_path", ""),
                }
            )
    return selected


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[k]) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        dice = [fnum(row["dice"]) for row in group_rows]
        delta = [fnum(row["delta_vs_auto_box"]) for row in group_rows]
        record = {k: v for k, v in zip(keys, group_key)}
        record.update(
            {
                "rows": len(group_rows),
                "mean_dice": round(mean(dice), 6),
                "mean_delta_vs_auto_box": round(mean(delta), 6),
                "wins_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and v > 0),
                "ties_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and abs(v) < 1e-9),
                "losses_vs_auto_box": sum(1 for v in delta if not math.isnan(v) and v < 0),
                "low_dice_lt_0_1": sum(1 for v in dice if not math.isnan(v) and v < 0.1),
                "high_dice_ge_0_5": sum(1 for v in dice if not math.isnan(v) and v >= 0.5),
            }
        )
        out.append(record)
    return out


def write_doc(path: Path, summary: list[dict[str, Any]], v1_summary: list[dict[str, str]], v2_summary: list[dict[str, str]]) -> None:
    base = {row["policy"]: row for row in v1_summary}
    v2 = v2_summary[0]
    v3 = summary[0]
    lines = [
        "# Prompt Selector v3 Validation",
        "",
        "## Component",
        "",
        "`prompt_selector_v3` adds prompt-induced uncertainty and mask stability to the v2 reliability scorer.",
        "",
        "The component is inspired by uncertainty-guided multi-prompt adaptation and dynamic prompting ideas in the reviewed medical SAM papers, but is implemented locally with the current benchmark artifacts.",
        "",
        "It uses no GT on the held-out case. For each case/model, it compares auto-family masks and derives:",
        "",
        "- mean mask agreement between candidate prompts;",
        "- minimum mask agreement;",
        "- prompt uncertainty = 1 - agreement;",
        "- area deviation from median candidate mask;",
        "- area deviation from `auto_box` mask;",
        "- SAM/SAM2 raw score when available.",
        "",
        "## Results",
        "",
        "| method | mean Dice | delta vs auto_box | low Dice < 0.1 | high Dice >= 0.5 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| single_auto_box | {float(base['single_auto_box']['mean_dice']):.4f} | {float(base['single_auto_box']['mean_delta_vs_auto_box']):+.4f} | {base['single_auto_box']['low_dice_lt_0_1']} | {base['single_auto_box']['high_dice_ge_0_5']} |",
        f"| data_aware_selector_v1 | {float(base['data_aware_selector_v1']['mean_dice']):.4f} | {float(base['data_aware_selector_v1']['mean_delta_vs_auto_box']):+.4f} | {base['data_aware_selector_v1']['low_dice_lt_0_1']} | {base['data_aware_selector_v1']['high_dice_ge_0_5']} |",
        f"| ridge_reliability_selector_v2 | {float(v2['mean_dice']):.4f} | {float(v2['mean_delta_vs_auto_box']):+.4f} | {v2['low_dice_lt_0_1']} | {v2['high_dice_ge_0_5']} |",
        f"| uncertainty_reliability_selector_v3 | {float(v3['mean_dice']):.4f} | {float(v3['mean_delta_vs_auto_box']):+.4f} | {v3['low_dice_lt_0_1']} | {v3['high_dice_ge_0_5']} |",
        f"| oracle_auto_family | {float(base['oracle_auto_family']['mean_dice']):.4f} | {float(base['oracle_auto_family']['mean_delta_vs_auto_box']):+.4f} | {base['oracle_auto_family']['low_dice_lt_0_1']} | {base['oracle_auto_family']['high_dice_ge_0_5']} |",
        "",
        "## Interpretation",
        "",
        "If v3 exceeds v2, prompt-induced uncertainty is useful and should remain part of Phase3.",
        "If v3 is close to v2, the stability features are not enough on 40 cases but still provide an interpretable reliability signal for the paper.",
        "If v3 drops below v2, the current uncertainty proxy is too noisy and should be replaced by perturbation-based sampling.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt_selector_v3 with prompt-induced uncertainty features.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    analysis = run_root / "analysis"
    metrics = read_csv(run_root / "metrics" / "phase2_lite_metrics.csv")
    rows = enrich_auto_rows(metrics)
    selected = evaluate_selector(rows)
    summary = summarize(selected, ["policy"])
    summary_dataset = summarize(selected, ["policy", "dataset"])

    write_csv(analysis / "prompt_selector_v3_results.csv", selected)
    write_csv(analysis / "prompt_selector_v3_summary_by_policy.csv", summary)
    write_csv(analysis / "prompt_selector_v3_summary_by_policy_dataset.csv", summary_dataset)

    v1_summary = read_csv(analysis / "prompt_selector_v1_summary_by_policy.csv")
    v2_summary = read_csv(analysis / "prompt_selector_v2_summary_by_policy.csv")
    write_doc(WORKSPACE_ROOT / "docs" / "prompt_selector_v3_validation.md", summary, v1_summary, v2_summary)
    print(f"Wrote {analysis / 'prompt_selector_v3_results.csv'}", flush=True)
    print(f"Wrote {WORKSPACE_ROOT / 'docs' / 'prompt_selector_v3_validation.md'}", flush=True)


if __name__ == "__main__":
    main()
