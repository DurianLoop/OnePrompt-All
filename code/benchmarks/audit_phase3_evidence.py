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
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def bootstrap_ci(values: np.ndarray, seed: int = 20260618, n: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values.astype(np.float64)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalue(values: np.ndarray, seed: int = 20260618, n: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(values)))
    null_means = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.sum(null_means >= observed) + 1) / (n + 1))


def paired_selector_tests(selector_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[float]] = defaultdict(list)
    for row in selector_rows:
        by_policy[row["policy"]].append(fnum(row["delta_vs_auto_box"]))
    out: list[dict[str, Any]] = []
    for policy, deltas in sorted(by_policy.items()):
        arr = np.asarray([v for v in deltas if not math.isnan(v)], dtype=np.float64)
        if len(arr) == 0:
            continue
        lo, hi = bootstrap_ci(arr)
        out.append(
            {
                "policy": policy,
                "n_pairs": len(arr),
                "mean_delta_vs_auto_box": round(float(arr.mean()), 6),
                "median_delta_vs_auto_box": round(float(np.median(arr)), 6),
                "bootstrap_ci95_low": round(lo, 6),
                "bootstrap_ci95_high": round(hi, 6),
                "sign_flip_pvalue": round(sign_flip_pvalue(arr), 6),
                "wins": int(np.sum(arr > 0)),
                "ties": int(np.sum(np.abs(arr) < 1e-9)),
                "losses": int(np.sum(arr < 0)),
                "practical_note": practical_note(float(arr.mean()), lo, hi),
            }
        )
    return out


def practical_note(mean_delta: float, lo: float, hi: float) -> str:
    if lo > 0.0 and mean_delta >= 0.02:
        return "statistically_positive_and_practically_visible"
    if lo > 0.0:
        return "statistically_positive_but_small"
    if hi < 0.0:
        return "negative"
    return "uncertain_or_small"


def index_metrics(metrics: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in metrics:
        if row["status"] == "ok":
            out[(row["case_id"], row["model"], row["protocol"])] = row
    return out


def model_portfolio_rows(metrics: list[dict[str, str]], selector_v3: list[dict[str, str]]) -> list[dict[str, Any]]:
    metric_index = index_metrics(metrics)
    by_case_selector: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selector_v3:
        by_case_selector[row["case_id"]].append(row)
    case_ids = sorted(by_case_selector)
    models = ("sam_vit_b", "medsam_vit_b", "sam2_tiny")
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        selector_candidates = by_case_selector[case_id]
        dataset = selector_candidates[0]["dataset"]
        for model in models:
            auto = metric_index.get((case_id, model, "auto_box"))
            if auto:
                rows.append(
                    {
                        "case_id": case_id,
                        "dataset": dataset,
                        "policy": f"fixed_{model}_auto_box",
                        "selected_model": model,
                        "selected_protocol": "auto_box",
                        "dice": round(fnum(auto["dice"]), 6),
                    }
                )
        # Deployable-ish portfolio: choose the model/prompt with highest v3
        # predicted reliability. This tests whether scores are calibrated across
        # models, which is harder than within-model prompt selection.
        best_pred = max(selector_candidates, key=lambda r: fnum(r["predicted_reliability"]))
        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "policy": "portfolio_v3_predicted_reliability",
                "selected_model": best_pred["model"],
                "selected_protocol": best_pred["selected_protocol"],
                "dice": round(fnum(best_pred["dice"]), 6),
            }
        )
        # Upper bounds for model complementarity.
        auto_rows = [metric_index[(case_id, model, "auto_box")] for model in models if (case_id, model, "auto_box") in metric_index]
        best_auto = max(auto_rows, key=lambda r: fnum(r["dice"]))
        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "policy": "oracle_best_model_auto_box",
                "selected_model": best_auto["model"],
                "selected_protocol": "auto_box",
                "dice": round(fnum(best_auto["dice"]), 6),
            }
        )
        best_v3 = max(selector_candidates, key=lambda r: fnum(r["dice"]))
        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "policy": "oracle_best_model_v3_selector",
                "selected_model": best_v3["model"],
                "selected_protocol": best_v3["selected_protocol"],
                "dice": round(fnum(best_v3["dice"]), 6),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row[k]) for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(buckets.items()):
        dice = [fnum(row["dice"]) for row in group_rows]
        record = {k: v for k, v in zip(keys, group_key)}
        record.update(
            {
                "rows": len(group_rows),
                "mean_dice": round(mean(dice), 6),
                "low_dice_lt_0_1": sum(1 for v in dice if not math.isnan(v) and v < 0.1),
                "high_dice_ge_0_5": sum(1 for v in dice if not math.isnan(v) and v >= 0.5),
            }
        )
        out.append(record)
    return out


def portfolio_significance(portfolio_rows: list[dict[str, Any]], baseline_policy: str = "fixed_sam_vit_b_auto_box") -> list[dict[str, Any]]:
    by_case_policy: dict[tuple[str, str], dict[str, Any]] = {}
    for row in portfolio_rows:
        by_case_policy[(row["case_id"], row["policy"])] = row
    policies = sorted({row["policy"] for row in portfolio_rows if row["policy"] != baseline_policy})
    out: list[dict[str, Any]] = []
    for policy in policies:
        deltas: list[float] = []
        for (case_id, row_policy), row in by_case_policy.items():
            if row_policy != policy:
                continue
            baseline = by_case_policy.get((case_id, baseline_policy))
            if baseline is None:
                continue
            deltas.append(fnum(row["dice"]) - fnum(baseline["dice"]))
        arr = np.asarray([v for v in deltas if not math.isnan(v)], dtype=np.float64)
        if len(arr) == 0:
            continue
        lo, hi = bootstrap_ci(arr, seed=20260619)
        out.append(
            {
                "policy": policy,
                "baseline_policy": baseline_policy,
                "n_pairs": len(arr),
                "mean_delta": round(float(arr.mean()), 6),
                "median_delta": round(float(np.median(arr)), 6),
                "bootstrap_ci95_low": round(lo, 6),
                "bootstrap_ci95_high": round(hi, 6),
                "sign_flip_pvalue": round(sign_flip_pvalue(arr, seed=20260619), 6),
                "wins": int(np.sum(arr > 0)),
                "ties": int(np.sum(np.abs(arr) < 1e-9)),
                "losses": int(np.sum(arr < 0)),
                "practical_note": practical_note(float(arr.mean()), lo, hi),
            }
        )
    return out


def write_doc(
    path: Path,
    significance: list[dict[str, Any]],
    portfolio_summary: list[dict[str, Any]],
    portfolio_sig: list[dict[str, Any]],
) -> None:
    sig = {row["policy"]: row for row in significance}
    port = {row["policy"]: row for row in portfolio_summary}
    lines = [
        "# Phase3 Evidence Audit: Significance, Model Combinations, and Strong Baselines",
        "",
        "## Why this audit exists",
        "",
        "A small Dice improvement is not automatically meaningful. This audit checks three questions:",
        "",
        "1. Is the selector improvement statistically reliable on paired case/model samples?",
        "2. Does the method combine meaningfully with different SAM-family backends?",
        "3. Can we honestly claim competitiveness with recent strong medical segmentation baselines?",
        "",
        "## 1. Paired significance",
        "",
        "The comparison is paired against fixed `auto_box` over the same case/model rows.",
        "",
        "| policy | mean delta | 95% bootstrap CI | p-value | wins/ties/losses | note |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in significance:
        lines.append(
            f"| {row['policy']} | {float(row['mean_delta_vs_auto_box']):+.4f} | "
            f"[{float(row['bootstrap_ci95_low']):+.4f}, {float(row['bootstrap_ci95_high']):+.4f}] | "
            f"{float(row['sign_flip_pvalue']):.4f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['practical_note']} |"
        )
    v3 = sig.get("uncertainty_reliability_selector_v3")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            f"- `prompt_selector_v3` has mean delta {float(v3['mean_delta_vs_auto_box']):+.4f} if available.",
            "- If the confidence interval includes zero, the current 40-case benchmark is not enough for a strong statistical claim.",
            "- Even when positive, a delta around 0.01-0.02 Dice should be described as a reliability/selection gain, not a new SOTA segmentation model.",
            "",
            "## 2. Model combination audit",
            "",
            "| policy | rows | mean Dice | low Dice < 0.1 | high Dice >= 0.5 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in portfolio_summary:
        lines.append(
            f"| {row['policy']} | {row['rows']} | {float(row['mean_dice']):.4f} | "
            f"{row['low_dice_lt_0_1']} | {row['high_dice_ge_0_5']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The oracle model portfolios estimate complementarity between SAM, MedSAM, and SAM2.",
            "- A predicted-reliability portfolio is harder than within-model prompt selection because reliability scores must be calibrated across models.",
            "- If the deployable portfolio is below a fixed model, the next step is calibration, not another prompt heuristic.",
            "",
            "### Portfolio paired test",
            "",
            "Baseline: `fixed_sam_vit_b_auto_box`, the strongest fixed single-model auto-box baseline in this run.",
            "",
            "| policy | mean delta | 95% bootstrap CI | p-value | wins/ties/losses | note |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in portfolio_sig:
        lines.append(
            f"| {row['policy']} | {float(row['mean_delta']):+.4f} | "
            f"[{float(row['bootstrap_ci95_low']):+.4f}, {float(row['bootstrap_ci95_high']):+.4f}] | "
            f"{float(row['sign_flip_pvalue']):.4f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['practical_note']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Strong baseline positioning",
            "",
            "Current OCSAM is a prompt selection and reranking method over SAM-family outputs. It should not yet be claimed to beat recent fully trained medical segmentation models.",
            "",
            "Representative recent/stronger baselines include:",
            "",
            "- Medical SAM Adapter / Med-SA: light adaptation over many medical tasks.",
            "- SAM-Med2D: large-scale medical 2D fine-tuned SAM over millions of medical images and masks.",
            "- MedSAM / MedSAM2 style medical adaptation.",
            "- Task-specific U-Net/nnU-Net/TransUNet-style models on standard medical datasets.",
            "",
            "Our current claim should be narrower:",
            "",
            "```text",
            "OCSAM improves automatic prompt selection and reliability over fixed automatic prompts for SAM-family medical segmentation.",
            "```",
            "",
            "To claim competitiveness with strong baselines, the project must add:",
            "",
            "1. larger validation, at least 100-200 cases across multiple medical domains;",
            "2. at least one trained medical SAM baseline such as SAM-Med2D or Med-SA if weights are usable;",
            "3. a task-specific supervised baseline where possible;",
            "4. statistical tests against each baseline, not only within-SAM prompt ablations.",
            "",
            "## Recommendation",
            "",
            "Proceed with OCSAM as a prompt reliability paper, not as a universal SOTA medical segmentation paper yet.",
            "",
            "Next technical step:",
            "",
            "```text",
            "prompt_selector_v4 = perturbation-based uncertainty + cross-model calibration.",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase3 evidence for significance, model combinations, and baseline positioning.")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    analysis = run_root / "analysis"
    metrics = read_csv(run_root / "metrics" / "phase2_lite_metrics.csv")
    selector_v1 = read_csv(analysis / "prompt_selector_v1_results.csv")
    selector_v2 = read_csv(analysis / "prompt_selector_v2_results.csv")
    selector_v3 = read_csv(analysis / "prompt_selector_v3_results.csv")
    selector_all = selector_v1 + selector_v2 + selector_v3

    significance = paired_selector_tests(selector_all)
    portfolio = model_portfolio_rows(metrics, selector_v3)
    portfolio_summary = summarize(portfolio, ["policy"])
    portfolio_dataset_summary = summarize(portfolio, ["policy", "dataset"])
    portfolio_sig = portfolio_significance(portfolio)

    write_csv(analysis / "phase3_selector_significance.csv", significance)
    write_csv(analysis / "phase3_model_portfolio_results.csv", portfolio)
    write_csv(analysis / "phase3_model_portfolio_summary.csv", portfolio_summary)
    write_csv(analysis / "phase3_model_portfolio_summary_by_dataset.csv", portfolio_dataset_summary)
    write_csv(analysis / "phase3_model_portfolio_significance.csv", portfolio_sig)
    write_doc(WORKSPACE_ROOT / "docs" / "phase3_evidence_audit.md", significance, portfolio_summary, portfolio_sig)
    print(f"Wrote {analysis / 'phase3_selector_significance.csv'}", flush=True)
    print(f"Wrote {WORKSPACE_ROOT / 'docs' / 'phase3_evidence_audit.md'}", flush=True)


if __name__ == "__main__":
    main()
