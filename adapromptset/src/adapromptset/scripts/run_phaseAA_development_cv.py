#!/usr/bin/env python3
"""Phase AA Gate AA2: Development Cohort 80-query grouped 5-fold CV.

Evaluates Prediction Consistency + baselines with proper grouped cross-validation.
Query-level grouping, reference seeds kept with query, learning baselines
trained on inner folds, evaluated on outer folds.

Metrics: AUROC, AUPRC, AURC, excess AURC, CFR@coverage, Dice@coverage.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Pipeline data paths ──────────────────────────────────────────────
PIPELINE_DIRS = {
    "L+SAM-B":  REPORTS_DIR / "phaseU_wp1" / "sam_b",
    "L+SAM-L":  REPORTS_DIR / "phaseU_wp1" / "sam_l",
    "L+SAM2-T": REPORTS_DIR / "phaseU_wp1" / "sam2_t",
    "L+SAM2-S": REPORTS_DIR / "phaseU_wp1" / "sam2_s",
    "B+SAM-B":  REPORTS_DIR / "phaseU_wp2",
}

FEATURE_COLS = [
    "matching_map_disagreement",
    "candidate_jaccard_mean_k1",
    "prompt_overlap_mean_k1",
    "prediction_jaccard_k1_within",
]

COVERAGE_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
N_FOLDS = 5
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_pipeline_data(pipeline_name: str) -> pd.DataFrame:
    """Load episode instability data for a pipeline."""
    path = PIPELINE_DIRS[pipeline_name] / "phase2_episode_instability.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path)
    df["pipeline"] = pipeline_name
    return df


def load_all_pipelines() -> dict[str, pd.DataFrame]:
    """Load all pipeline data."""
    return {name: load_pipeline_data(name) for name in PIPELINE_DIRS}


def aggregate_to_query_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-episode data to query level (mean over seeds and configs)."""
    query_data = df.groupby("query_case_id").agg({
        "matching_map_disagreement": "mean",
        "candidate_jaccard_mean_k1": "mean",
        "prompt_overlap_mean_k1": "mean",
        "prediction_jaccard_k1_within": "mean",
        "mean_dice_k1": "mean",
        "std_dice_k1": "mean",
        "best_worst_gap_k1": "mean",
        "catastrophic_failure_rate_k1": "mean",
        "n_references_k1": "first",
    }).reset_index()
    return query_data


# ══════════════════════════════════════════════════════════════════════
# Risk Score Computation
# ══════════════════════════════════════════════════════════════════════

def compute_risk_scores(query_data: pd.DataFrame, train_idx=None, test_idx=None) -> dict[str, np.ndarray]:
    """Compute all risk scores. If train_idx is provided, train learning models on train set."""
    n = len(query_data)
    risk_scores = {}

    # B0: Random
    np.random.seed(RANDOM_SEED)
    risk_scores["B0_Random"] = np.random.rand(n)

    # B1: Matching-map entropy (higher = riskier)
    risk_scores["B1_Entropy"] = query_data["matching_map_disagreement"].values

    # B2: Candidate Jaccard (lower = riskier)
    risk_scores["B2_CandidateJaccard"] = 1.0 - query_data["candidate_jaccard_mean_k1"].values

    # B3: Prompt overlap (lower = riskier)
    risk_scores["B3_PromptOverlap"] = 1.0 - query_data["prompt_overlap_mean_k1"].values

    # B4: Prediction consistency (lower = riskier) — PRIMARY
    risk_scores["B4_PredictionConsistency"] = 1.0 - query_data["prediction_jaccard_k1_within"].values

    # B5: M2-Instability (trained)
    features = query_data[FEATURE_COLS].values
    inst_labels = (query_data["best_worst_gap_k1"] > 0.3).astype(int).values

    if train_idx is not None and len(np.unique(inst_labels[train_idx])) > 1:
        inst_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED
        )
        inst_model.fit(features[train_idx], inst_labels[train_idx])
        b5 = inst_model.predict_proba(features)[:, 1]
    elif len(np.unique(inst_labels)) > 1:
        inst_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED
        )
        inst_model.fit(features, inst_labels)
        b5 = inst_model.predict_proba(features)[:, 1]
    else:
        b5 = inst_labels.astype(float)
    risk_scores["B5_M2Instability"] = b5

    # B6: M2-Catastrophic (trained)
    cat_labels = (query_data["mean_dice_k1"] < 0.1).astype(int).values

    if train_idx is not None and len(np.unique(cat_labels[train_idx])) > 1:
        cat_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED
        )
        cat_model.fit(features[train_idx], cat_labels[train_idx])
        b6 = cat_model.predict_proba(features)[:, 1]
    elif len(np.unique(cat_labels)) > 1:
        cat_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED
        )
        cat_model.fit(features, cat_labels)
        b6 = cat_model.predict_proba(features)[:, 1]
    else:
        b6 = cat_labels.astype(float)
    risk_scores["B6_M2Catastrophic"] = b6

    # B7: Dual-head M2
    risk_scores["B7_DualHeadM2"] = (b5 + b6) / 2.0

    return risk_scores


# ══════════════════════════════════════════════════════════════════════
# Metrics Computation
# ══════════════════════════════════════════════════════════════════════

def compute_auc_metrics(dice: np.ndarray, cfr: np.ndarray, risk: np.ndarray) -> dict:
    """Compute AUROC and AUPRC for catastrophic failure prediction."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    cat_label = (dice < 0.1).astype(int)
    if len(np.unique(cat_label)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}

    try:
        auroc = float(roc_auc_score(cat_label, risk))
    except Exception:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(cat_label, risk))
    except Exception:
        auprc = float("nan")

    return {"auroc": auroc, "auprc": auprc}


def compute_aurc(dice: np.ndarray, risk: np.ndarray, cfr: np.ndarray) -> dict:
    """Compute AURC (Area Under Risk-Coverage curve) and excess AURC.

    AURC: mean CFR of cumulative top-k as k varies (matching PhaseW definition).
    Lower AURC = better risk ranking (fewer catastrophic failures in accepted set).
    excess_aurc = AURC - random_aurc (negative = better than random).
    """
    n = len(risk)
    order = np.argsort(risk)[::-1]  # Highest risk first
    sorted_cfr = cfr[order]

    # AURC: average CFR of top-k as k varies from 1 to n
    aurc = 0.0
    for i in range(1, n + 1):
        aurc += sorted_cfr[:i].mean()
    aurc /= n

    # Random baseline AURC
    random_order = np.random.RandomState(RANDOM_SEED).permutation(n)
    random_cfr = sorted_cfr[random_order]
    random_aurc = 0.0
    for i in range(1, n + 1):
        random_aurc += random_cfr[:i].mean()
    random_aurc /= n

    excess_aurc = float(aurc - random_aurc)

    return {"aurc": float(aurc), "excess_aurc": excess_aurc, "random_aurc": float(random_aurc)}


def compute_coverage_metrics(
    dice: np.ndarray, risk: np.ndarray, cfr: np.ndarray, coverage_grid: list[float]
) -> dict:
    """Compute CFR and Dice at each coverage level."""
    n = len(risk)
    order = np.argsort(risk)[::-1]  # Highest risk first
    sorted_dice = dice[order]
    sorted_cfr = cfr[order]

    cfr_all = float(np.mean(cfr))
    dice_all = float(np.mean(dice))

    result = {
        "cfr_no_abstention": cfr_all,
        "dice_no_abstention": dice_all,
    }

    for cov in coverage_grid:
        k = max(1, int(n * cov))
        accepted_cfr = float(np.mean(sorted_cfr[:k]))
        accepted_dice = float(np.mean(sorted_dice[:k]))
        cfr_red = float((cfr_all - accepted_cfr) / cfr_all * 100) if cfr_all > 0 else 0.0

        result[f"cfr_at_{int(cov*100)}"] = accepted_cfr
        result[f"dice_at_{int(cov*100)}"] = accepted_dice
        result[f"cfr_reduction_at_{int(cov*100)}"] = cfr_red

    return result


def compute_risk_coverage_curve(
    dice: np.ndarray, risk: np.ndarray, cfr: np.ndarray, coverage_grid: list[float]
) -> list[dict]:
    """Compute full risk-coverage curve points."""
    n = len(risk)
    order = np.argsort(risk)[::-1]
    sorted_dice = dice[order]
    sorted_cfr = cfr[order]

    curve = []
    for cov in coverage_grid:
        k = max(1, int(n * cov))
        curve.append({
            "coverage": cov,
            "num_queries": k,
            "dice": float(np.mean(sorted_dice[:k])),
            "cfr": float(np.mean(sorted_cfr[:k])),
        })

    return curve


def evaluate_baseline(
    dice: np.ndarray, risk: np.ndarray, cfr: np.ndarray, name: str
) -> dict:
    """Evaluate all metrics for a single risk baseline."""
    auc = compute_auc_metrics(dice, cfr, risk)
    aurc = compute_aurc(dice, risk, cfr)
    cov = compute_coverage_metrics(dice, risk, cfr, COVERAGE_GRID)

    return {**auc, **aurc, **cov, "name": name}


# ══════════════════════════════════════════════════════════════════════
# Grouped CV
# ══════════════════════════════════════════════════════════════════════

def run_grouped_cv(query_data: pd.DataFrame) -> dict:
    """Run grouped 5-fold CV with query-level grouping."""
    query_ids = query_data["query_case_id"].values
    n_queries = len(query_ids)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = query_ids  # Each query is its own group

    fold_results = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(n_queries), groups=groups)):
        test_queries = query_ids[test_idx]
        for q in test_queries:
            fold_assignments[q] = fold_idx

        # Compute risk scores with training on inner folds
        risk_scores = compute_risk_scores(query_data, train_idx, test_idx)

        # Evaluate on test fold
        test_data = query_data.iloc[test_idx]
        test_dice = test_data["mean_dice_k1"].values
        test_cfr = test_data["catastrophic_failure_rate_k1"].values

        fold_eval = {}
        for name, risk in risk_scores.items():
            test_risk = risk[test_idx]
            fold_eval[name] = evaluate_baseline(test_dice, test_risk, test_cfr, name)

        fold_results.append({
            "fold": fold_idx,
            "n_test": len(test_idx),
            "test_queries": test_queries.tolist(),
            "baselines": fold_eval,
        })

    # Aggregate across folds
    aggregated = _aggregate_folds(fold_results)

    return {
        "n_queries": n_queries,
        "n_folds": N_FOLDS,
        "fold_assignments": {str(k): int(v) for k, v in fold_assignments.items()},
        "fold_results": fold_results,
        "aggregated": aggregated,
    }


def _aggregate_folds(fold_results: list[dict]) -> dict:
    """Aggregate metrics across folds (mean and std)."""
    baseline_names = list(fold_results[0]["baselines"].keys())
    aggregated = {}

    for name in baseline_names:
        metrics = {}
        for fold in fold_results:
            for metric, value in fold["baselines"][name].items():
                if metric == "name":
                    continue
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(value)

        agg = {}
        for metric, values in metrics.items():
            valid = [v for v in values if not np.isnan(v)]
            if valid:
                agg[metric] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                }
            else:
                agg[metric] = {"mean": float("nan"), "std": float("nan")}

        aggregated[name] = agg

    return aggregated


# ══════════════════════════════════════════════════════════════════════
# In-Sample Evaluation (for comparison)
# ══════════════════════════════════════════════════════════════════════

def run_in_sample_eval(query_data: pd.DataFrame) -> dict:
    """Run in-sample evaluation (no CV) for comparison."""
    risk_scores = compute_risk_scores(query_data)
    dice = query_data["mean_dice_k1"].values
    cfr = query_data["catastrophic_failure_rate_k1"].values

    results = {}
    for name, risk in risk_scores.items():
        results[name] = evaluate_baseline(dice, risk, cfr, name)

    return results


# ══════════════════════════════════════════════════════════════════════
# Gate AA2 Decision
# ══════════════════════════════════════════════════════════════════════

def check_gate_criteria(pipeline_results: dict) -> dict:
    """Check Gate AA2 success criteria for each pipeline."""
    criteria = {}

    for pipeline_name, result in pipeline_results.items():
        agg = result["aggregated"]
        pc = agg.get("B4_PredictionConsistency", {})
        random = agg.get("B0_Random", {})
        entropy = agg.get("B1_Entropy", {})

        checks = {}

        # C1: AURC lower than Random (lower AURC = better risk ranking)
        if "aurc" in pc and "aurc" in random:
            pc_aurc = pc["aurc"]["mean"]
            random_aurc = random["aurc"]["mean"]
            checks["c1_aurc_vs_random"] = {
                "pass": pc_aurc < random_aurc,
                "pc_aurc": pc_aurc,
                "random_aurc": random_aurc,
                "delta": pc_aurc - random_aurc,
            }
        else:
            checks["c1_aurc_vs_random"] = {"pass": None, "error": "missing data"}

        # C2: AURC improvement >= 0.01 vs Entropy (lower AURC is better), or CFR@80 reduction >= 20%
        if "aurc" in pc and "aurc" in entropy:
            aurc_delta = entropy["aurc"]["mean"] - pc["aurc"]["mean"]  # Positive = PC better
            cfr_red_pc = pc.get("cfr_reduction_at_80", {}).get("mean", 0)
            checks["c2_aurc_vs_entropy"] = {
                "pass": aurc_delta >= 0.01 or cfr_red_pc >= 20,
                "aurc_delta": aurc_delta,
                "cfr_reduction_at_80": cfr_red_pc,
            }
        else:
            checks["c2_aurc_vs_entropy"] = {"pass": None, "error": "missing data"}

        # C3: AUROC >= 0.75 in primary pipeline
        if "auroc" in pc:
            auroc_val = pc["auroc"]["mean"]
            checks["c3_auroc"] = {
                "pass": auroc_val >= 0.75 if not np.isnan(auroc_val) else False,
                "pc_auroc": auroc_val,
            }

        # C4: CFR relative reduction >= 30% at 80% coverage
        if "cfr_reduction_at_80" in pc:
            cfr_red = pc["cfr_reduction_at_80"]["mean"]
            checks["c4_cfr_reduction"] = {
                "pass": cfr_red >= 30,
                "cfr_reduction_at_80": cfr_red,
            }

        # C5: Selective Dice not lower than random
        if "dice_at_80" in pc and "dice_at_80" in random:
            pc_dice = pc["dice_at_80"]["mean"]
            random_dice = random["dice_at_80"]["mean"]
            checks["c5_dice_vs_random"] = {
                "pass": pc_dice >= random_dice,
                "pc_dice": pc_dice,
                "random_dice": random_dice,
            }

        criteria[pipeline_name] = checks

    return criteria


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    output_dir = REPORTS_DIR / "phaseAA_development_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase AA Gate AA2: Development Cohort 80-query Grouped CV")
    print("=" * 70)

    # Load all pipeline data
    all_data = load_all_pipelines()
    print(f"\nLoaded {len(all_data)} pipelines")

    pipeline_results = {}
    pipeline_in_sample = {}

    for pipeline_name, df in all_data.items():
        print(f"\n{'─' * 70}")
        print(f"Pipeline: {pipeline_name}")

        # Aggregate to query level
        query_data = aggregate_to_query_level(df)
        n_queries = len(query_data)
        print(f"  Queries: {n_queries}")
        print(f"  Mean Dice: {query_data['mean_dice_k1'].mean():.4f}")
        print(f"  CFR: {query_data['catastrophic_failure_rate_k1'].mean():.4f}")

        # Run grouped CV
        print(f"  Running {N_FOLDS}-fold grouped CV...")
        cv_result = run_grouped_cv(query_data)
        pipeline_results[pipeline_name] = cv_result

        # Run in-sample evaluation
        in_sample = run_in_sample_eval(query_data)
        pipeline_in_sample[pipeline_name] = in_sample

        # Print key results
        agg = cv_result["aggregated"]
        pc = agg.get("B4_PredictionConsistency", {})
        if pc:
            print(f"  B4 PC AURC: {pc.get('aurc', {}).get('mean', 'N/A'):.4f} ± {pc.get('aurc', {}).get('std', 'N/A'):.4f}")
            print(f"  B4 PC AUROC: {pc.get('auroc', {}).get('mean', 'N/A'):.4f} ± {pc.get('auroc', {}).get('std', 'N/A'):.4f}")
            print(f"  B4 PC CFR@80: {pc.get('cfr_at_80', {}).get('mean', 'N/A'):.4f}")
            print(f"  B4 PC CFR Red@80: {pc.get('cfr_reduction_at_80', {}).get('mean', 'N/A'):.1f}%")

    # Check gate criteria
    print(f"\n{'=' * 70}")
    print("Gate AA2 Criteria Check")
    print("=" * 70)

    criteria = check_gate_criteria(pipeline_results)

    all_passed = []
    for pipeline_name, checks in criteria.items():
        print(f"\n{pipeline_name}:")
        passed = 0
        total = 0
        for check_name, check in checks.items():
            status = "PASS" if check.get("pass") else ("FAIL" if check.get("pass") is False else "N/A")
            if status == "PASS":
                passed += 1
            if status != "N/A":
                total += 1
            if check_name == "c1_aurc_vs_random":
                print(f"  {check_name}: {status} (PC AURC={check.get('pc_aurc', 'N/A'):.4f}, Random={check.get('random_aurc', 'N/A'):.4f})")
            elif check_name == "c2_aurc_vs_entropy":
                print(f"  {check_name}: {status} (ΔAURC={check.get('aurc_delta', 'N/A'):.4f}, CFR Red={check.get('cfr_reduction_at_80', 'N/A'):.1f}%)")
            elif check_name == "c3_auroc":
                print(f"  {check_name}: {status} (AUROC={check.get('pc_auroc', 'N/A'):.4f})")
            elif check_name == "c4_cfr_reduction":
                print(f"  {check_name}: {status} (CFR Red={check.get('cfr_reduction_at_80', 'N/A'):.1f}%)")
            elif check_name == "c5_dice_vs_random":
                print(f"  {check_name}: {status} (PC Dice={check.get('pc_dice', 'N/A'):.4f}, Random={check.get('random_dice', 'N/A'):.4f})")
        all_passed.append(f"{pipeline_name}: {passed}/{total}")

    # Determine overall gate status
    primary = criteria.get("L+SAM-B", {})
    n_strong = sum(1 for p, c in criteria.items() 
                   if c.get("c1_aurc_vs_random", {}).get("pass") 
                   and c.get("c4_cfr_reduction", {}).get("pass"))

    primary_pass = all(
        primary.get(c, {}).get("pass") for c in [
            "c1_aurc_vs_random", "c2_aurc_vs_entropy", "c3_auroc",
            "c4_cfr_reduction", "c5_dice_vs_random"
        ]
    )

    if primary_pass and n_strong >= 3:
        gate_status = "PHASEAA_PC_DEVELOPMENT_REPLICATED"
    elif n_strong >= 3:
        gate_status = "PHASEAA_PC_DEVELOPMENT_PARTIAL_PASS"
    else:
        gate_status = "PHASEAA_PC_NOT_STABLE_ACROSS_PIPELINES"

    print(f"\nPassed per pipeline: {', '.join(all_passed)}")
    print(f"Strong pipelines: {n_strong}/5")
    print(f"Gate AA2: {gate_status}")

    # Save results
    output = {
        "gate": "AA2",
        "date": "2026-07-27",
        "status": gate_status,
        "n_folds": N_FOLDS,
        "pipeline_results": {
            p: {
                "n_queries": r["n_queries"],
                "aggregated": r["aggregated"],
            }
            for p, r in pipeline_results.items()
        },
        "in_sample_results": pipeline_in_sample,
        "gate_criteria": {p: {
            k: {kk: (vv if not (isinstance(vv, float) and np.isnan(vv)) else None) for kk, vv in v.items()}
            for k, v in c.items()
        } for p, c in criteria.items()},
        "fold_assignments": {
            p: r["fold_assignments"] for p, r in pipeline_results.items()
        },
    }

    output_path = output_dir / "development_cv_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Save detailed fold results
    detailed_path = output_dir / "development_cv_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {detailed_path}")

    return output


if __name__ == "__main__":
    main()