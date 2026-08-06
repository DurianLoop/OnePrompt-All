from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def fnum(value: str | float | int) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def supported_auto_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        rows.append(enriched)
    return rows


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def feature_vector(row: dict[str, Any]) -> list[float]:
    pred_area_ratio = float(row["pred_area_ratio"])
    score = float(row["score_value"])
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
        ]
    )


def fit_ridge(train_rows: list[dict[str, Any]], alpha: float = 0.6) -> np.ndarray:
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
    # Blend a learned scorer with a stable protocol prior. On 40 cases this is
    # intentionally conservative; the next step is to replace it with a stronger
    # reliability model once more cases are available.
    return 0.62 * prior + 0.38 * linear


def evaluate_selector(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_model[(row["case_id"], row["model"])].append(row)

    selected: list[dict[str, Any]] = []
    all_case_ids = sorted({row["case_id"] for row in rows})
    for case_id in all_case_ids:
        train_rows = [row for row in rows if row["case_id"] != case_id]
        if not train_rows:
            continue
        weights = fit_ridge(train_rows)
        priors = protocol_prior(train_rows)
        global_mean = mean([fnum(row["dice"]) for row in train_rows])
        for (candidate_case, model), candidates in sorted(by_case_model.items()):
            if candidate_case != case_id:
                continue
            scored = [
                (predict_reliability(row, weights, priors, global_mean), row)
                for row in candidates
            ]
            predicted, best_row = max(scored, key=lambda item: item[0])
            auto_box = next((row for row in candidates if row["protocol"] == "auto_box"), None)
            auto_box_dice = fnum(auto_box["dice"]) if auto_box is not None else math.nan
            selected.append(
                {
                    "case_id": case_id,
                    "dataset": best_row["dataset"],
                    "model": model,
                    "policy": "ridge_reliability_selector_v2",
                    "selected_protocol": best_row["protocol"],
                    "predicted_reliability": round(predicted, 6),
                    "dice": round(fnum(best_row["dice"]), 6),
                    "delta_vs_auto_box": round(fnum(best_row["dice"]) - auto_box_dice, 6) if not math.isnan(auto_box_dice) else "nan",
                    "score": round(float(best_row["score_value"]), 6),
                    "pred_area_ratio": round(float(best_row["pred_area_ratio"]), 6),
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


def write_doc(path: Path, selector_summary: list[dict[str, Any]], policy_summary: list[dict[str, str]]) -> None:
    base = {row["policy"]: row for row in policy_summary}
    v2 = selector_summary[0]
    auto = base["single_auto_box"]
    data_v1 = base["data_aware_selector_v1"]
    oracle = base["oracle_auto_family"]
    lines = [
        "# Prompt Selector v2 Validation",
        "",
        "## Component",
        "",
        "`prompt_selector_v2` is a lightweight reliability scorer inspired by prompt reranking, uncertainty-aware selection, and dynamic prompting components in the reviewed medical SAM papers.",
        "",
        "It uses leave-one-case-out validation. GT Dice is used only on the training cases to fit the scorer; the held-out case is selected without GT.",
        "",
        "Features:",
        "",
        "- dataset one-hot",
        "- model one-hot",
        "- prompt protocol one-hot",
        "- SAM/SAM2 raw score when available",
        "- score missing flag",
        "- predicted mask area ratio",
        "- small/large mask flags",
        "",
        "## Results",
        "",
        "| method | mean Dice | delta vs auto_box | low Dice < 0.1 | high Dice >= 0.5 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| single_auto_box | {float(auto['mean_dice']):.4f} | {float(auto['mean_delta_vs_auto_box']):+.4f} | {auto['low_dice_lt_0_1']} | {auto['high_dice_ge_0_5']} |",
        f"| data_aware_selector_v1 | {float(data_v1['mean_dice']):.4f} | {float(data_v1['mean_delta_vs_auto_box']):+.4f} | {data_v1['low_dice_lt_0_1']} | {data_v1['high_dice_ge_0_5']} |",
        f"| ridge_reliability_selector_v2 | {float(v2['mean_dice']):.4f} | {float(v2['mean_delta_vs_auto_box']):+.4f} | {v2['low_dice_lt_0_1']} | {v2['high_dice_ge_0_5']} |",
        f"| oracle_auto_family | {float(oracle['mean_dice']):.4f} | {float(oracle['mean_delta_vs_auto_box']):+.4f} | {oracle['low_dice_lt_0_1']} | {oracle['high_dice_ge_0_5']} |",
        "",
        "## Interpretation",
        "",
        "- If v2 is above `data_aware_selector_v1`, the learned reliability route is already useful on 40 cases.",
        "- If v2 is below `data_aware_selector_v1`, the feature set is too weak and the next component should be prompt-induced uncertainty or mask stability, not a larger backbone.",
        "- The gap to `oracle_auto_family` is the remaining Phase3 research space.",
        "",
        "## Phase3 Completion Criterion",
        "",
        "Phase3 is considered complete for the small-project milestone when:",
        "",
        "1. selector is evaluated without GT on held-out cases;",
        "2. it is compared against fixed auto prompt and data-aware rules;",
        "3. the oracle gap is reported honestly;",
        "4. the next component is chosen based on failure evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate leave-one-case-out prompt_selector_v2.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    analysis = run_root / "analysis"
    metrics = read_csv(run_root / "metrics" / "phase2_lite_metrics.csv")
    rows = supported_auto_rows(metrics)
    selected = evaluate_selector(rows)
    summary = summarize(selected, ["policy"])
    summary_dataset = summarize(selected, ["policy", "dataset"])

    write_csv(analysis / "prompt_selector_v2_results.csv", selected)
    write_csv(analysis / "prompt_selector_v2_summary_by_policy.csv", summary)
    write_csv(analysis / "prompt_selector_v2_summary_by_policy_dataset.csv", summary_dataset)

    policy_summary = read_csv(analysis / "prompt_selector_v1_summary_by_policy.csv")
    write_doc(WORKSPACE_ROOT / "docs" / "prompt_selector_v2_validation.md", summary, policy_summary)
    print(f"Wrote {analysis / 'prompt_selector_v2_results.csv'}", flush=True)
    print(f"Wrote {WORKSPACE_ROOT / 'docs' / 'prompt_selector_v2_validation.md'}", flush=True)


if __name__ == "__main__":
    main()
