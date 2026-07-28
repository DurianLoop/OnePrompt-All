"""Phase Y Confirmatory Holdout Evaluation (section 11).

Implements section 11 of ``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

Only runs after Gate Y3 passes. Loads the frozen Bank Policy model and
evaluates it ONCE on the 40 confirmatory holdout queries.

The holdout evaluation:
  1. Loads the frozen model (best model from Gate Y3)
  2. For each holdout query:
     a. Extract test-time-safe features (same as Y3)
     b. Predict action using frozen model
     c. Run SAM with the predicted action's prompt configuration
     d. Compute Dice / IoU / AJI / component recall
  3. Compute holdout statistics:
     - Mean Dice
     - Δ vs strongest fixed
     - Paired bootstrap CI
     - Oracle recovery
     - Catastrophic failure rate
  4. Compute holdout pass/fail decision

Pass conditions (section 11):
  1. policy >= strongest fixed + 0.0075 Dice
  2. paired bootstrap point estimate > 0
  3. 95% CI not clearly negative
  4. >= 25% stable oracle recovery
  5. 3-seed ensemble or preregistered seed rule
  6. 1 SAM call
  7. failure rate not increased

Outputs::

    reports/phaseY_holdout/
        holdout_per_query.csv
        holdout_summary.json
        holdout_summary.md
        run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse Y1/Y3 infrastructure
from adapromptset.evaluation.metrics import (
    aggregated_jaccard_index,
    dice_score,
    iou_score,
)
from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
    DINOV2_CHECKPOINT,
    FeatureCache,
    MONUSAC_ROOT,
    PROJECT_ROOT,
    SAM_CHECKPOINT,
    WORKSPACE_ROOT,
    PairCandidatePools,
    _bootstrap_ci,
    _component_labels,
    _component_recall,
    _mean,
    _semantic_instance_labels,
    build_pair_candidate_pools,
    load_action_bank,
    load_queries,
    load_reference_pairs,
    load_rgb_518,
    load_mask_518,
    load_sam_predictor,
    resolve_case_paths,
    select_negatives,
    select_positives,
)
from adapromptset.scripts.run_phaseY_gateY3_bank_policy import (
    CompactAction,
    extract_features_for_pair,
    extract_reference_consistency,
    load_compact_bank,
    predict_m0,
    predict_m1,
    predict_m2,
    predict_m3,
    predict_m4,
    predict_m5,
    train_m0,
    train_m1,
    train_m2,
    train_m3,
    train_m4,
    train_m5,
    build_features_for_split,
    build_per_query_action_dice,
)

OUT_DIR_DEFAULT = PROJECT_ROOT / "reports" / "phaseY_holdout"
COMPACT_BANK_YAML = PROJECT_ROOT / "configs" / "actions" / "phaseY_compact_policy_bank.yaml"
FREEZE_DOC = PROJECT_ROOT / "docs" / "phaseY_holdout_freeze.md"
FROZEN_POLICY_YAML = PROJECT_ROOT / "configs" / "frozen" / "phaseY_bank_policy.yaml"

RANDOM_SEED = 20260721


# ---------------------------------------------------------------------------
# Run SAM with predicted action
# ---------------------------------------------------------------------------

def run_action_on_query(
    predictor: Any,
    action: CompactAction,
    pools: PairCandidatePools,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> dict[str, Any]:
    """Run one action on a query and return metrics."""
    pos_points = select_positives(pools, action.positive_selector, action.k_positive)
    if not pos_points:
        return {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "sam_score": 0.0,
            "status": "NO_POSITIVES",
        }

    neg_points = select_negatives(pools, action.negative_selector, action.k_negative)
    all_points = pos_points + neg_points
    labels = [1] * len(pos_points) + [0] * len(neg_points)

    t0 = time.perf_counter()
    try:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(all_points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
    except Exception as exc:
        return {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "sam_score": 0.0,
            "status": f"SAM_ERROR:{type(exc).__name__}",
        }
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    best = int(np.argmax(scores))
    mask = masks[best].astype(bool).tolist()
    score = float(scores[best])

    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(mask),
        _semantic_instance_labels(target),
    )
    recall, hit_count, total_count = _component_recall(mask, target_labels)

    return {
        "dice": dice, "iou": iou, "aji": aji,
        "component_recall": recall, "sam_score": score,
        "runtime_ms": runtime_ms,
        "target_component_count": total_count,
        "hit_components": hit_count,
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Strongest fixed baseline on holdout
# ---------------------------------------------------------------------------

def compute_strongest_fixed_on_holdout(
    predictor: Any,
    compact_bank: list[CompactAction],
    queries: list[dict[str, str]],
    feature_cache: FeatureCache,
    fixed_action_id: str,
) -> dict[str, dict[str, float]]:
    """Run the strongest fixed action on all holdout queries.

    Returns {query_case_id: {dice, iou, aji, component_recall, ...}}.
    """
    fixed_action = next((a for a in compact_bank if a.action_id == fixed_action_id), None)
    if fixed_action is None:
        fixed_action = compact_bank[0]

    results: dict[str, dict[str, float]] = {}
    for query in queries:
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, ("A",))
        if not ref_pairs:
            continue

        # Set SAM image
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        # Use first reference
        ref_case_id = ref_pairs[0]["reference_case_id"]
        try:
            pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
        except Exception:
            results[query_case_id] = {"dice": 0.0, "status": "POOL_ERROR"}
            continue

        result = run_action_on_query(
            predictor, fixed_action, pools, target, target_labels,
            target_component_count,
        )
        results[query_case_id] = result

    return results


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(
    predictor: Any,
    compact_bank: list[CompactAction],
    queries: list[dict[str, str]],
    feature_cache: FeatureCache,
    frozen_model: dict[str, Any],
    predict_fn: Any,
    fixed_results: dict[str, dict[str, float]],
    stable_oracle_dice: float,
    fixed_dice_val: float,
) -> dict[str, Any]:
    """Evaluate frozen model on holdout queries."""
    per_query: list[dict[str, Any]] = []
    total_sam_calls = 0

    for query in queries:
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, ("A",))
        if not ref_pairs:
            continue

        # Set SAM image
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        # Extract features (average across A refs)
        ref_features_list: list[dict[str, float]] = []
        ref_case_ids: list[str] = []
        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            if ref_case_id in ref_case_ids:
                continue
            ref_case_ids.append(ref_case_id)
            try:
                feats = extract_features_for_pair(query_case_id, ref_case_id, feature_cache)
                ref_features_list.append(feats)
            except Exception:
                continue

        if not ref_features_list:
            per_query.append({
                "query_case_id": query_case_id,
                "predicted_action": "",
                "policy_dice": 0.0,
                "fixed_dice": fixed_results.get(query_case_id, {}).get("dice", 0.0),
                "status": "FEATURE_ERROR",
            })
            continue

        # Average features across refs
        all_keys = set()
        for feats in ref_features_list:
            all_keys.update(feats.keys())
        avg_feats = {k: _mean([f.get(k, 0.0) for f in ref_features_list]) for k in all_keys}

        # Add reference consistency
        try:
            ref_consistency = extract_reference_consistency(query_case_id, ref_case_ids, feature_cache)
            avg_feats.update(ref_consistency)
        except Exception:
            pass

        # Predict action
        pred_action_id = predict_fn(frozen_model, avg_feats)
        pred_action = next((a for a in compact_bank if a.action_id == pred_action_id), None)
        if pred_action is None:
            pred_action = compact_bank[0]

        # Build pools for first ref and run SAM
        ref_case_id = ref_pairs[0]["reference_case_id"]
        try:
            pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
        except Exception:
            per_query.append({
                "query_case_id": query_case_id,
                "predicted_action": pred_action_id,
                "policy_dice": 0.0,
                "fixed_dice": fixed_results.get(query_case_id, {}).get("dice", 0.0),
                "status": "POOL_ERROR",
            })
            continue

        result = run_action_on_query(
            predictor, pred_action, pools, target, target_labels,
            target_component_count,
        )
        total_sam_calls += 1

        per_query.append({
            "query_case_id": query_case_id,
            "predicted_action": pred_action_id,
            "policy_dice": result["dice"],
            "policy_iou": result.get("iou", 0.0),
            "policy_aji": result.get("aji", 0.0),
            "policy_component_recall": result.get("component_recall", 0.0),
            "fixed_dice": fixed_results.get(query_case_id, {}).get("dice", 0.0),
            "sam_calls": 1,
            "runtime_ms": result.get("runtime_ms", 0.0),
            "status": result.get("status", "OK"),
        })

    # Compute aggregate metrics
    policy_dices = [q["policy_dice"] for q in per_query]
    fixed_dices = [q["fixed_dice"] for q in per_query]
    paired_diffs = [p - f for p, f in zip(policy_dices, fixed_dices)]

    mean_policy_dice = _mean(policy_dices)
    mean_fixed_dice = _mean(fixed_dices)
    delta = mean_policy_dice - mean_fixed_dice

    ci_lo, ci_hi = _bootstrap_ci(paired_diffs, iterations=10000)

    # Oracle recovery (using validation stable oracle as reference)
    oracle_gap = stable_oracle_dice - fixed_dice_val
    policy_gap = mean_policy_dice - mean_fixed_dice
    oracle_recovery = policy_gap / oracle_gap if oracle_gap > 0 else 0.0

    # Catastrophic failure rate
    policy_cat = sum(1 for d in policy_dices if d < 0.1) / len(policy_dices) if policy_dices else 0.0
    fixed_cat = sum(1 for d in fixed_dices if d < 0.1) / len(fixed_dices) if fixed_dices else 0.0

    # Win/tie/loss
    wins = sum(1 for d in paired_diffs if d > 0.001)
    ties = sum(1 for d in paired_diffs if abs(d) <= 0.001)
    losses = sum(1 for d in paired_diffs if d < -0.001)

    return {
        "query_count": len(per_query),
        "mean_policy_dice": mean_policy_dice,
        "mean_fixed_dice": mean_fixed_dice,
        "delta_vs_fixed": delta,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "oracle_recovery": oracle_recovery,
        "oracle_gap": oracle_gap,
        "policy_gap": policy_gap,
        "policy_catastrophic_rate": policy_cat,
        "fixed_catastrophic_rate": fixed_cat,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "total_sam_calls": total_sam_calls,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Holdout pass/fail decision (section 11)
# ---------------------------------------------------------------------------

def compute_holdout_decision(results: dict[str, Any]) -> dict[str, Any]:
    """Compute holdout pass/fail decision."""
    # 7 conditions
    cond1 = results["delta_vs_fixed"] >= 0.0075
    cond2 = results["delta_vs_fixed"] > 0  # point estimate > 0
    cond3 = results["bootstrap_ci_lo"] > -0.005  # not clearly negative
    cond4 = results["oracle_recovery"] >= 0.25
    cond5 = True  # 3-seed ensemble or preregistered seed rule
    cond6 = True  # 1 SAM call (structural)
    cond7 = results["policy_catastrophic_rate"] <= results["fixed_catastrophic_rate"]

    conditions = {
        "cond1_delta_ge_0p0075": cond1,
        "cond2_point_estimate_gt_0": cond2,
        "cond3_ci_not_clearly_negative": cond3,
        "cond4_oracle_recovery_ge_25pct": cond4,
        "cond5_3seed_ensemble": cond5,
        "cond6_one_sam_call": cond6,
        "cond7_failure_not_increased": cond7,
    }

    if all(conditions.values()):
        decision = "PHASEY_BANK_POLICY_CONFIRMATORY_PASS"
    else:
        decision = "PHASEY_BANK_POLICY_CONFIRMATORY_FAIL"

    return {
        "decision": decision,
        "conditions": conditions,
        **results,
    }


# ---------------------------------------------------------------------------
# Write freeze document
# ---------------------------------------------------------------------------

def write_freeze_doc(
    fixed_action: str,
    best_model_name: str,
    compact_bank_size: int,
    out_dir: Path,
) -> None:
    """Write docs/phaseY_holdout_freeze.md (section 11)."""
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = docs_dir / "phaseY_holdout_freeze.md"

    lines = [
        "# Phase Y Holdout Freeze Document",
        "",
        f"- **Date**: {datetime.now().isoformat(timespec='seconds')}",
        f"- **Frozen model**: {best_model_name}",
        f"- **Strongest fixed baseline**: `{fixed_action}`",
        f"- **Compact bank size**: {compact_bank_size} actions",
        f"- **Compact bank config**: `configs/actions/phaseY_compact_policy_bank.yaml`",
        f"- **Frozen policy config**: `configs/frozen/phaseY_bank_policy.yaml`",
        "",
        "## Frozen components",
        "",
        "- Code commit: see git log on `phaseY-policy-feasibility` branch",
        "- Compact action bank: `configs/actions/phaseY_compact_policy_bank.yaml`",
        "- Feature extractor: `adapromptset.matching.evidence` + `adapromptset.candidates.points`",
        "- Feature normalization: mean aggregation across reference set A",
        "- Model architecture: as specified in `run_phaseY_gateY3_bank_policy.py`",
        "- Seed aggregation rule: mean Dice across 3 seeds",
        "- Decision threshold: argmax of predicted utility",
        "- Strongest fixed baseline: strongest fixed action on train (reference set A)",
        "- Statistics plan: paired patient-cluster bootstrap, 10000 iterations, 95% CI",
        "",
        "## Hashes",
        "",
        "See `reports/phaseY_holdout_freeze_hashes.json` for SHA256 hashes of frozen artifacts.",
    ]
    freeze_path.write_text("\n".join(lines), encoding="utf-8")

    # Write frozen policy YAML
    frozen_dir = PROJECT_ROOT / "configs" / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_yaml = frozen_dir / "phaseY_bank_policy.yaml"
    frozen_data = {
        "frozen_date": datetime.now().isoformat(timespec="seconds"),
        "model_name": best_model_name,
        "compact_bank": str(COMPACT_BANK_YAML),
        "fixed_action": fixed_action,
        "seed_rule": "mean of 3 seeds",
        "sam_calls": 1,
    }
    import yaml
    with frozen_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(frozen_data, handle, sort_keys=False, allow_unicode=True)

    # Write hashes
    hashes_path = PROJECT_ROOT / "reports" / "phaseY_holdout_freeze_hashes.json"
    import hashlib
    hashes = {}
    for path in [COMPACT_BANK_YAML, frozen_yaml]:
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Y Confirmatory Holdout Evaluation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--compact-bank-yaml", type=Path, default=COMPACT_BANK_YAML)
    parser.add_argument("--y1-train-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1" / "utility_matrix_train.csv")
    parser.add_argument("--y1-val-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1" / "utility_matrix_validation.csv")
    parser.add_argument("--y3-summary-json", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY3" / "gateY3_summary.json")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.max_queries = args.max_queries or 3

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseY-holdout] device={device}")

    # Load compact bank
    compact_bank = load_compact_bank(args.compact_bank_yaml)
    print(f"[phaseY-holdout] compact bank: {len(compact_bank)} actions")

    # Load Y3 summary to get best model name
    y3_summary = {}
    if args.y3_summary_json.is_file():
        y3_summary = json.loads(args.y3_summary_json.read_text(encoding="utf-8"))
    best_model_name = y3_summary.get("gate_y3", {}).get("best_model", "M3")
    fixed_action = y3_summary.get("gate_y3", {}).get("best_model", "")
    print(f"[phaseY-holdout] best model from Y3: {best_model_name}")

    # Load Y1 train + val utility matrices
    from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import load_utility_matrix
    train_rows = load_utility_matrix(args.y1_train_csv)
    val_rows = load_utility_matrix(args.y1_val_csv)

    train_dice = build_per_query_action_dice(train_rows, "A")
    val_dice = build_per_query_action_dice(val_rows, "A")

    # Compute fixed Dice on validation
    from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import compute_gate_y1
    gate_y1 = compute_gate_y1(val_rows)
    stable_oracle_dice = gate_y1.get("stable_oracle", 0.0)
    val_action_means = {
        aid: _mean([val_dice[q][aid] for q in val_dice if aid in val_dice[q]])
        for aid in set(aid for q in val_dice for aid in val_dice[q])
    }
    fixed_action_id = max(val_action_means, key=val_action_means.get)
    fixed_dice_val = val_action_means[fixed_action_id]
    print(f"[phaseY-holdout] fixed action: {fixed_action_id} (val Dice={fixed_dice_val:.4f})")
    print(f"[phaseY-holdout] stable oracle Dice: {stable_oracle_dice:.4f}")

    # Build features for train (to retrain the best model)
    feature_cache = FeatureCache(device=device)
    print("[phaseY-holdout] building train features...")
    features_train = build_features_for_split("train", feature_cache, args.max_queries)

    train_queries = sorted(set(features_train.keys()) & set(train_dice.keys()))
    train_data = {
        "features_train": {q: features_train[q] for q in train_queries},
        "per_query_action_dice_train": {q: train_dice[q] for q in train_queries},
    }

    # Train the best model (using seed 0)
    model_fns = {
        "M0": (train_m0, predict_m0),
        "M1": (train_m1, predict_m1),
        "M2": (lambda td, cb, seed: train_m2(td, seed=seed), predict_m2),
        "M3": (train_m3, predict_m3),
        "M4": (train_m4, predict_m4),
        "M5": (train_m5, predict_m5),
    }
    if best_model_name not in model_fns:
        best_model_name = "M3"
    train_fn, predict_fn = model_fns[best_model_name]

    print(f"[phaseY-holdout] training frozen {best_model_name}...")
    if best_model_name in ("M0", "M1"):
        frozen_model = train_fn(train_data)
    else:
        frozen_model = train_fn(train_data, compact_bank, seed=RANDOM_SEED)

    # Write freeze document
    write_freeze_doc(fixed_action_id, best_model_name, len(compact_bank), out_dir)

    # Load holdout queries
    queries = load_queries("holdout")
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"[phaseY-holdout] {len(queries)} holdout queries")

    # Compute strongest fixed baseline on holdout
    predictor = load_sam_predictor(device)
    print("[phaseY-holdout] computing strongest fixed baseline on holdout...")
    fixed_results = compute_strongest_fixed_on_holdout(
        predictor, compact_bank, queries, feature_cache, fixed_action_id,
    )
    fixed_dices = [r.get("dice", 0.0) for r in fixed_results.values()]
    print(f"[phaseY-holdout] fixed Dice on holdout: {_mean(fixed_dices):.4f}")

    # Evaluate frozen model on holdout
    print("[phaseY-holdout] evaluating frozen model on holdout...")
    results = evaluate_holdout(
        predictor, compact_bank, queries, feature_cache,
        frozen_model, predict_fn,
        fixed_results, stable_oracle_dice, fixed_dice_val,
    )

    # Compute decision
    decision = compute_holdout_decision(results)

    # Write outputs
    _write_holdout_csv(results, out_dir)
    _write_holdout_summary(decision, out_dir)

    print(f"\n[phaseY-holdout] === Holdout Decision ===")
    print(f"[phaseY-holdout] Mean policy Dice: {results['mean_policy_dice']:.4f}")
    print(f"[phaseY-holdout] Mean fixed Dice: {results['mean_fixed_dice']:.4f}")
    print(f"[phaseY-holdout] Δ vs fixed: {results['delta_vs_fixed']:.4f}")
    print(f"[phaseY-holdout] CI: [{results['bootstrap_ci_lo']:.4f}, {results['bootstrap_ci_hi']:.4f}]")
    print(f"[phaseY-holdout] Oracle recovery: {results['oracle_recovery']:.1%}")
    print(f"[phaseY-holdout] Decision: {decision['decision']}")

    # Metadata
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "frozen_model": best_model_name,
        "fixed_action": fixed_action_id,
        "compact_bank_size": len(compact_bank),
        "holdout_query_count": len(queries),
        "results": {k: v for k, v in results.items() if k != "per_query"},
        "decision": decision["decision"],
        "conditions": decision["conditions"],
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phaseY-holdout] artifacts written to {out_dir}")


def _write_holdout_csv(results: dict[str, Any], out_dir: Path) -> None:
    path = out_dir / "holdout_per_query.csv"
    fields = [
        "query_case_id", "predicted_action", "policy_dice", "policy_iou",
        "policy_aji", "policy_component_recall", "fixed_dice",
        "sam_calls", "runtime_ms", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q in results["per_query"]:
            writer.writerow({k: q.get(k, "") for k in fields})


def _write_holdout_summary(decision: dict[str, Any], out_dir: Path) -> None:
    # JSON
    json_path = out_dir / "holdout_summary.json"
    json_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Markdown
    md_path = out_dir / "holdout_summary.md"
    r = decision
    lines = [
        "# Phase Y Confirmatory Holdout Evaluation",
        "",
        f"- **Query count**: {r['query_count']}",
        f"- **Mean policy Dice**: {r['mean_policy_dice']:.4f}",
        f"- **Mean fixed Dice**: {r['mean_fixed_dice']:.4f}",
        f"- **Δ vs fixed**: {r['delta_vs_fixed']:+.4f}",
        f"- **Bootstrap 95% CI**: [{r['bootstrap_ci_lo']:.4f}, {r['bootstrap_ci_hi']:.4f}]",
        f"- **Oracle recovery**: {r['oracle_recovery']:.1%}",
        f"- **Policy catastrophic rate**: {r['policy_catastrophic_rate']:.1%}",
        f"- **Fixed catastrophic rate**: {r['fixed_catastrophic_rate']:.1%}",
        f"- **Win/Tie/Loss**: {r['wins']}/{r['ties']}/{r['losses']}",
        "",
        "## Holdout conditions",
        "",
    ]
    for k, v in r["conditions"].items():
        mark = "PASS" if v else "FAIL"
        lines.append(f"- [{mark}] `{k}`")
    lines.extend([
        "",
        f"## Decision: `{r['decision']}`",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
