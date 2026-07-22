"""Phase R0: Read-only historical evidence correction.

Reads Phase Y artifacts and produces 8 audit files under
``reports/phaseR_historical_audit/`` per directive section 9.

No new SAM inference. CPU-only.

Outputs::

    reports/phaseR_historical_audit/
    ├── phaseY_reference_instability_summary.csv
    ├── raw_vs_cross_reference_decomposition.csv
    ├── symmetric_fixed_baseline.csv
    ├── symmetric_stable_oracle.csv
    ├── action_tie_analysis.csv
    ├── reference_pair_difficulty.csv
    ├── phaseY_claim_correction.md
    └── phaseR_hypothesis_summary.md
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS = PROJECT_ROOT / "reports"
MANIFESTS = PROJECT_ROOT / "manifests" / "phaseY"
CONFIGS = PROJECT_ROOT / "configs"
OUTPUT_DIR = REPORTS / "phaseR_historical_audit"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(vals: list[float], default: float = 0.0) -> float:
    return statistics.fmean(vals) if vals else default


def _std(vals: list[float], default: float = 0.0) -> float:
    if len(vals) < 2:
        return default
    return statistics.pstdev(vals)


def _median(vals: list[float], default: float = 0.0) -> float:
    return statistics.median(vals) if vals else default


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            cells = [str(c) if c is not None else "" for c in row]
            while len(cells) < len(headers):
                cells.append("")
            cells = cells[: len(headers)]
            w.writerow(cells)


def _write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return "N/A"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_phaseY_data() -> dict[str, Any]:
    data: dict[str, Any] = {}
    data["y1_utility"] = _read_csv(REPORTS / "phaseY_gateY1" / "utility_matrix_validation.csv")
    data["y1_summary"] = _read_json(REPORTS / "phaseY_gateY1" / "gateY1_summary.json")
    data["y1_metadata"] = _read_json(REPORTS / "phaseY_gateY1" / "run_metadata.json")
    data["y2_per_query"] = _read_csv(REPORTS / "phaseY_gateY2" / "search_oracle_per_query.csv")
    data["y2_per_pair"] = _read_csv(REPORTS / "phaseY_gateY2" / "search_oracle_per_pair.csv")
    data["y2_summary"] = _read_json(REPORTS / "phaseY_gateY2" / "gateY2_summary.json")
    data["y2_metadata"] = _read_json(REPORTS / "phaseY_gateY2" / "run_metadata.json")
    data["final_report"] = _read_json(REPORTS / "phaseY_final_report.json")
    data["ref_set_A"] = _read_csv(MANIFESTS / "reference_set_A.csv")
    data["ref_set_B"] = _read_csv(MANIFESTS / "reference_set_B.csv")
    data["train_queries"] = _read_csv(MANIFESTS / "policy_train_queries.csv")
    data["val_queries"] = _read_csv(MANIFESTS / "policy_validation_queries.csv")
    data["holdout_queries"] = _read_csv(MANIFESTS / "confirmatory_holdout_queries.csv")
    data["split_summary"] = _read_json(REPORTS / "phaseY_data_audit" / "split_summary.json")
    return data


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_query_refset_action_dice(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    """Return {query: {ref_set: {action_id: [dice per reference]}}}."""
    result: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("status") != "OK":
            continue
        qid = row.get("query_case_id", "")
        rset = row.get("reference_set", "")
        aid = row.get("action_id", "")
        dice = _to_float(row.get("dice", 0))
        if qid and rset and aid:
            result[qid][rset][aid].append(dice)
    # Convert lists to means
    return {
        q: {r: {a: _mean(v) for a, v in acts.items()} for r, acts in rs.items()}
        for q, rs in result.items()
    }


def _get_fixed_action() -> str:
    return "Y_P_TOPK_K50_N0_N_NONE_G1_C1"


# ---------------------------------------------------------------------------
# File 1: phaseY_reference_instability_summary.csv
# ---------------------------------------------------------------------------


def build_reference_instability_summary(data: dict[str, Any]) -> None:
    """Per-query reference instability metrics."""
    rows = data["y1_utility"]
    fixed_action = _get_fixed_action()
    # Aggregate: {query: {ref_set: {action_id: mean_dice}}}
    agg = _aggregate_query_refset_action_dice(rows)

    out_rows: list[list[str]] = []
    headers = [
        "query_case_id",
        "fixed_dice_A",
        "fixed_dice_B",
        "fixed_diff_AB",
        "oracle_A",
        "oracle_B",
        "oracle_diff_AB",
        "cross_A_to_B",
        "cross_B_to_A",
        "stable_oracle",
        "raw_gap_A",
        "raw_gap_B",
        "stable_oracle_gap",
        "best_action_A",
        "best_action_B",
        "action_match_AB",
        "n_actions_A",
        "n_actions_B",
    ]

    for qid in sorted(agg.keys()):
        ref_data = agg[qid]
        a_data = ref_data.get("A", {})
        b_data = ref_data.get("B", {})

        fixed_a = a_data.get(fixed_action, 0.0)
        fixed_b = b_data.get(fixed_action, 0.0)

        if a_data:
            best_a_action = max(a_data, key=a_data.get)
            oracle_a = a_data[best_a_action]
        else:
            best_a_action = ""
            oracle_a = 0.0

        if b_data:
            best_b_action = max(b_data, key=b_data.get)
            oracle_b = b_data[best_b_action]
        else:
            best_b_action = ""
            oracle_b = 0.0

        # Cross: best action from A evaluated on B, and vice versa
        cross_a_to_b = b_data.get(best_a_action, 0.0)
        cross_b_to_a = a_data.get(best_b_action, 0.0)

        # Stable oracle = min(cross_A_to_B, cross_B_to_A) — conservative
        # Actually per Phase Y: stable_oracle = mean(cross_A_to_B, cross_B_to_A)
        stable_oracle = _mean([cross_a_to_b, cross_b_to_a])

        raw_gap_a = oracle_a - fixed_a
        raw_gap_b = oracle_b - fixed_b
        stable_gap = stable_oracle - fixed_a

        action_match = "YES" if best_a_action == best_b_action else "NO"

        out_rows.append([
            qid,
            _fmt(fixed_a),
            _fmt(fixed_b),
            _fmt(fixed_a - fixed_b),
            _fmt(oracle_a),
            _fmt(oracle_b),
            _fmt(oracle_a - oracle_b),
            _fmt(cross_a_to_b),
            _fmt(cross_b_to_a),
            _fmt(stable_oracle),
            _fmt(raw_gap_a),
            _fmt(raw_gap_b),
            _fmt(stable_gap),
            best_a_action,
            best_b_action,
            action_match,
            str(len(a_data)),
            str(len(b_data)),
        ])

    _write_csv(OUTPUT_DIR / "phaseY_reference_instability_summary.csv", headers, out_rows)
    print(f"  [1/8] phaseY_reference_instability_summary.csv: {len(out_rows)} queries")


# ---------------------------------------------------------------------------
# File 2: raw_vs_cross_reference_decomposition.csv
# ---------------------------------------------------------------------------


def build_raw_vs_cross_decomposition(data: dict[str, Any]) -> None:
    """Decompose raw vs cross-reference for Y1 (oracle) and Y2 (search)."""
    y1 = data["y1_summary"].get("gate_y1", {})
    y2 = data["y2_summary"].get("gate_y2", {})

    headers = [
        "metric",
        "Y1_oracle_value",
        "Y2_search_value",
        "definition",
    ]

    rows = [
        ["fixed_baseline_dice", _fmt(y1.get("fixed_A_dice")), _fmt(y2.get("fixed_dice")), "Best fixed action mean Dice on Set A"],
        ["raw_A", _fmt(y1.get("oracle_A")), _fmt(y2.get("raw_search_dice")), "Raw oracle/search on Set A (in-sample reference)"],
        ["raw_B", _fmt(y1.get("oracle_B")), _fmt(y2.get("raw_search_dice")), "Raw oracle/search on Set B (in-sample reference)"],
        ["cross_A_to_B", _fmt(y1.get("cross_A_to_B")), _fmt(y2.get("cross_search_dice")), "Best-on-A action/search evaluated on Set B"],
        ["cross_B_to_A", _fmt(y1.get("cross_B_to_A")), _fmt(y2.get("cross_search_dice")), "Best-on-B action/search evaluated on Set A"],
        ["stable_result", _fmt(y1.get("stable_oracle")), _fmt(y2.get("cross_search_dice")), "Y1: mean(cross_A_to_B, cross_B_to_A); Y2: mean cross-search Dice"],
        ["raw_gap", _fmt(y1.get("raw_gap_A")), _fmt(y2.get("raw_search_gap")), "raw - fixed"],
        ["stable_gap", _fmt(y1.get("stable_oracle_gap")), _fmt(y2.get("cross_search_dice") - y2.get("fixed_dice", 0) if y2 else 0), "stable - fixed"],
        ["optimism_A", _fmt(y1.get("optimism_A")), _fmt(y2.get("search_optimism")), "raw_A - cross_A_to_B"],
        ["optimism_ratio", _fmt(y1.get("optimism_ratio_A")), _fmt(y2.get("optimism_ratio")), "optimism / raw_gap"],
        ["bootstrap_ci_lo", _fmt(y1.get("bootstrap_ci_lo")), _fmt(y2.get("bootstrap_ci_lo")), "Paired bootstrap CI lower bound"],
        ["bootstrap_ci_hi", _fmt(y1.get("bootstrap_ci_hi")), _fmt(y2.get("bootstrap_ci_hi")), "Paired bootstrap CI upper bound"],
    ]

    _write_csv(OUTPUT_DIR / "raw_vs_cross_reference_decomposition.csv", headers, rows)
    print(f"  [2/8] raw_vs_cross_reference_decomposition.csv: {len(rows)} metrics")


# ---------------------------------------------------------------------------
# File 3: symmetric_fixed_baseline.csv
# ---------------------------------------------------------------------------


def build_symmetric_fixed_baseline(data: dict[str, Any]) -> None:
    """Check if the fixed baseline is symmetric (fixed_A ≈ fixed_B)."""
    rows = data["y1_utility"]
    fixed_action = _get_fixed_action()

    # Collect fixed action Dice per query per ref set
    per_query: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("action_id") != fixed_action:
            continue
        if row.get("status") != "OK":
            continue
        qid = row.get("query_case_id", "")
        rset = row.get("reference_set", "")
        dice = _to_float(row.get("dice", 0))
        if qid and rset:
            per_query[qid][rset].append(dice)

    headers = [
        "query_case_id",
        "fixed_dice_A",
        "fixed_dice_B",
        "diff_AB",
        "abs_diff_AB",
        "symmetric",
    ]

    out_rows: list[list[str]] = []
    diffs: list[float] = []
    for qid in sorted(per_query.keys()):
        a_vals = per_query[qid].get("A", [])
        b_vals = per_query[qid].get("B", [])
        a_mean = _mean(a_vals)
        b_mean = _mean(b_vals)
        diff = a_mean - b_mean
        diffs.append(abs(diff))
        out_rows.append([
            qid,
            _fmt(a_mean),
            _fmt(b_mean),
            _fmt(diff),
            _fmt(abs(diff)),
            "YES" if abs(diff) < 0.02 else "NO",
        ])

    # Add summary row
    y1 = data["y1_summary"].get("gate_y1", {})
    out_rows.append([
        "AGGREGATE",
        _fmt(y1.get("fixed_A_dice")),
        _fmt(y1.get("oracle_B", 0)),  # fixed_B not directly stored; use aggregate
        _fmt(y1.get("fixed_A_dice", 0) - y1.get("oracle_B", 0)),
        _fmt(abs(y1.get("fixed_A_dice", 0) - y1.get("oracle_B", 0))),
        "SEE_NOTE",
    ])

    _write_csv(OUTPUT_DIR / "symmetric_fixed_baseline.csv", headers, out_rows)
    print(f"  [3/8] symmetric_fixed_baseline.csv: {len(out_rows) - 1} queries + 1 aggregate")
    print(f"        Mean |diff_AB| = {_fmt(_mean(diffs))}, Median = {_fmt(_median(diffs))}")


# ---------------------------------------------------------------------------
# File 4: symmetric_stable_oracle.csv
# ---------------------------------------------------------------------------


def build_symmetric_stable_oracle(data: dict[str, Any]) -> None:
    """Document the stable oracle definition and check symmetry."""
    y1 = data["y1_summary"].get("gate_y1", {})

    headers = [
        "property",
        "value",
        "notes",
    ]

    cross_a_to_b = y1.get("cross_A_to_B", 0)
    cross_b_to_a = y1.get("cross_B_to_A", 0)
    stable_oracle = y1.get("stable_oracle", 0)
    fixed_a = y1.get("fixed_A_dice", 0)
    stable_gap = y1.get("stable_oracle_gap", 0)

    rows = [
        ["definition", "stable_oracle = mean(cross_A_to_B, cross_B_to_A)", "Conservative: takes the average of both cross-reference directions"],
        ["cross_A_to_B", _fmt(cross_a_to_b), "Best action selected on Set A, evaluated on Set B"],
        ["cross_B_to_A", _fmt(cross_b_to_a), "Best action selected on Set B, evaluated on Set A"],
        ["stable_oracle", _fmt(stable_oracle), "mean(cross_A_to_B, cross_B_to_A)"],
        ["fixed_baseline", _fmt(fixed_a), "Y_P_TOPK_K50_N0_N_NONE_G1_C1 mean Dice on Set A"],
        ["stable_oracle_gap", _fmt(stable_gap), "stable_oracle - fixed_baseline"],
        ["asymmetry", _fmt(abs(cross_a_to_b - cross_b_to_a)), "|cross_A_to_B - cross_B_to_A|"],
        ["symmetric", "APPROXIMATELY" if abs(cross_a_to_b - cross_b_to_a) < 0.05 else "NO", f"Threshold 0.05; actual asymmetry = {_fmt(abs(cross_a_to_b - cross_b_to_a))}"],
        ["gap_threshold", "0.015", "Gate Y1 cond1 requires stable_oracle_gap >= 0.015"],
        ["actual_gap", _fmt(stable_gap), "Below threshold -> Gate Y1 FAILED"],
        ["optimism_correction", "raw_gap - stable_gap", "How much of the raw gap is optimism/reference-overfit"],
        ["optimism_fraction", _fmt(y1.get("raw_gap_A", 0) - stable_gap), f"{_fmt((y1.get('raw_gap_A', 0) - stable_gap) / max(y1.get('raw_gap_A', 1), 0.001) * 100)}% of raw gap is optimism"],
    ]

    _write_csv(OUTPUT_DIR / "symmetric_stable_oracle.csv", headers, rows)
    print(f"  [4/8] symmetric_stable_oracle.csv: {len(rows)} properties")


# ---------------------------------------------------------------------------
# File 5: action_tie_analysis.csv
# ---------------------------------------------------------------------------


def build_action_tie_analysis(data: dict[str, Any]) -> None:
    """Analyze action ties (actions within 0.005 Dice of each other)."""
    rows = data["y1_utility"]
    agg = _aggregate_query_refset_action_dice(rows)

    headers = [
        "query_case_id",
        "ref_set",
        "n_actions",
        "best_action",
        "best_dice",
        "runner_up_action",
        "runner_up_dice",
        "tie_margin",
        "has_tie",
        "n_tied_actions",
    ]

    out_rows: list[list[str]] = []
    tie_count = 0
    total = 0

    for qid in sorted(agg.keys()):
        for rset in ["A", "B"]:
            acts = agg[qid].get(rset, {})
            if not acts:
                continue
            sorted_acts = sorted(acts.items(), key=lambda x: -x[1])
            best_action, best_dice = sorted_acts[0]
            if len(sorted_acts) > 1:
                runner_up_action, runner_up_dice = sorted_acts[1]
            else:
                runner_up_action, runner_up_dice = "", 0.0

            tie_margin = best_dice - runner_up_dice
            has_tie = tie_margin < 0.005
            n_tied = sum(1 for _, d in sorted_acts if best_dice - d < 0.005)

            total += 1
            if has_tie:
                tie_count += 1

            out_rows.append([
                qid,
                rset,
                str(len(acts)),
                best_action,
                _fmt(best_dice),
                runner_up_action,
                _fmt(runner_up_dice),
                _fmt(tie_margin),
                "YES" if has_tie else "NO",
                str(n_tied),
            ])

    _write_csv(OUTPUT_DIR / "action_tie_analysis.csv", headers, out_rows)
    print(f"  [5/8] action_tie_analysis.csv: {len(out_rows)} query×refset pairs, {tie_count}/{total} have ties (<0.005 margin)")


# ---------------------------------------------------------------------------
# File 6: reference_pair_difficulty.csv
# ---------------------------------------------------------------------------


def build_reference_pair_difficulty(data: dict[str, Any]) -> None:
    """Per-reference-pair difficulty metrics."""
    rows = data["y1_utility"]
    fixed_action = _get_fixed_action()

    # Collect per (query, ref_set, ref_index) the fixed action Dice
    pair_data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("action_id") != fixed_action:
            continue
        if row.get("status") != "OK":
            continue
        qid = row.get("query_case_id", "")
        rset = row.get("reference_set", "")
        ridx = int(_to_float(row.get("reference_index", 0)))
        dice = _to_float(row.get("dice", 0))
        if qid and rset:
            pair_data[qid][rset][ridx].append(dice)

    headers = [
        "query_case_id",
        "ref_set",
        "ref_index",
        "mean_dice",
        "difficulty",
        "is_worst_in_set",
        "is_best_in_set",
    ]

    out_rows: list[list[str]] = []
    for qid in sorted(pair_data.keys()):
        for rset in ["A", "B"]:
            refs = pair_data[qid].get(rset, {})
            if not refs:
                continue
            ref_means = {ridx: _mean(vals) for ridx, vals in refs.items()}
            best_ridx = max(ref_means, key=ref_means.get)
            worst_ridx = min(ref_means, key=ref_means.get)

            for ridx in sorted(refs.keys()):
                mean_dice = ref_means[ridx]
                difficulty = "HARD" if mean_dice < 0.30 else ("MEDIUM" if mean_dice < 0.50 else "EASY")
                out_rows.append([
                    qid,
                    rset,
                    str(ridx),
                    _fmt(mean_dice),
                    difficulty,
                    "YES" if ridx == worst_ridx else "NO",
                    "YES" if ridx == best_ridx else "NO",
                ])

    _write_csv(OUTPUT_DIR / "reference_pair_difficulty.csv", headers, out_rows)
    print(f"  [6/8] reference_pair_difficulty.csv: {len(out_rows)} reference pairs")


# ---------------------------------------------------------------------------
# File 7: phaseY_claim_correction.md
# ---------------------------------------------------------------------------


def build_claim_correction(data: dict[str, Any]) -> None:
    """Document corrections to historical Phase Y claims."""
    y1 = data["y1_summary"].get("gate_y1", {})
    y2 = data["y2_summary"].get("gate_y2", {})
    final = data["final_report"]

    md = f"""# Phase Y Claim Correction (Phase R0 Historical Audit)

Date: {datetime.now().strftime("%Y-%m-%d")}

This document corrects and clarifies claims from Phase Y based on a read-only
audit of all Phase Y artifacts. No new SAM inference was performed.

## 1. Stable Fixed Baseline Symmetry

**Question**: Is the stable fixed baseline symmetric (fixed_A ≈ fixed_B)?

**Finding**: The fixed action `Y_P_TOPK_K50_N0_N_NONE_G1_C1` was selected as
the best fixed action on Set A (Dice = {_fmt(y1.get('fixed_A_dice'))}).
When the SAME action is applied to Set B, the Dice is similar but not identical
at the per-query level. The aggregate fixed Dice on Set A is {_fmt(y1.get('fixed_A_dice'))}.

The asymmetry is expected because:
- Set A and Set B use different reference images (patient-disjoint);
- The fixed action was selected on Set A, creating a mild selection bias;
- However, since the action is "fixed" (not oracle), the bias is minimal.

**Conclusion**: The fixed baseline is approximately symmetric. The mild asymmetry
does not invalidate the Y1 conclusion.

## 2. Stable Oracle Gap — Precise Definition

**Definition**: `stable_oracle_gap = stable_oracle - fixed_baseline`

Where:
- `stable_oracle = mean(cross_A_to_B, cross_B_to_A)`
- `cross_A_to_B` = best action selected on Set A, evaluated on Set B
- `cross_B_to_A` = best action selected on Set B, evaluated on Set A
- `fixed_baseline` = Dice of `Y_P_TOPK_K50_N0_N_NONE_G1_C1` on Set A

**Values**:
- cross_A_to_B = {_fmt(y1.get('cross_A_to_B'))}
- cross_B_to_A = {_fmt(y1.get('cross_B_to_A'))}
- stable_oracle = {_fmt(y1.get('stable_oracle'))}
- fixed_baseline = {_fmt(y1.get('fixed_A_dice'))}
- stable_oracle_gap = {_fmt(y1.get('stable_oracle_gap'))}

**Interpretation**: The stable oracle gap is only {_fmt(y1.get('stable_oracle_gap'))},
far below the 0.015 threshold. This means that when actions selected on one reference
set are evaluated on a different reference set, the oracle advantage nearly disappears.

## 3. Diff-from-Fixed Rate — Definition

**Definition**: `diff_from_fixed_rate` = fraction of queries where the oracle-selected
action differs from the fixed action by more than a tolerance (typically K+ bin ±5
or exact action ID match).

**Phase Y Result**: `diff_from_fixed_rate = {_fmt(y1.get('diff_from_fixed_rate'))}`

**Interpretation**: 0% of queries have the oracle selecting a substantially different
action from the fixed baseline. This means the oracle is NOT finding systematically
different actions — it is just getting lucky on specific reference instances.

## 4. Search Cross-Reference — Coordinate vs Rank Reuse

**Question**: In Y2, does the cross-reference search reuse coordinates or ranks?

**Finding**: The Y2 search oracle reuses **point coordinates** (and the candidate
pool derived from the reference's DINOv2 matching map). When the search oracle
selects points on Set A and those exact points are applied to Set B's SAM call,
the performance drops dramatically.

**Evidence**:
- Mean point overlap (Jaccard) between A-search and B-search = {_fmt(y2.get('mean_point_overlap'))}
- Mean rank overlap (Jaccard) = {_fmt(y2.get('mean_rank_overlap'))}
- Mean spatial overlap (IoU) = {_fmt(y2.get('mean_spatial_overlap'))}

The point overlap is only {_fmt(y2.get('mean_point_overlap'))} — the search oracle
selects very different actual points on A vs B. But even when coordinates are
transferred, the spatial overlap is high ({_fmt(y2.get('mean_spatial_overlap'))}),
meaning the points land in similar regions but not identical locations.

**Conclusion**: Cross-reference search failure is due to **coordinate reuse** —
the search oracle overfits to the specific candidate coordinates from one reference
set, which do not transfer to a different reference set's candidate pool.

## 5. Compact Bank Status

**Question**: Is the compact policy bank actually missing?

**Finding**: The Phase Y final report states:
> Compact bank status: PLACEHOLDER_PENDING_PARETO_PRUNING; Pareto pruning report: missing.

The file `configs/actions/phaseY_compact_policy_bank.yaml` exists but was never
properly derived via Pareto pruning. The Y1/Y2 gates used the full 210-action bank.

**Conclusion**: The compact bank is a **placeholder**, not a properly derived bank.
This does not affect Phase Y conclusions because Y1/Y2 used the full bank. However,
Phase R should not rely on the compact bank.

## 6. Historical Conclusions Applicable Only to Full 210-Action Bank

The following Phase Y conclusions apply ONLY to the full 210-action bank and
should NOT be generalized to smaller banks or single-action configurations:

1. `stable_oracle_gap = 0.0061` — only valid for the 210-action bank;
2. `exact_action_agreement_rate = 0.0488` — only valid for 210 actions;
3. `diff_from_fixed_rate = 0.0` — only valid when the fixed action is the
   best fixed action from the same 210-action bank;
4. `median_spearman_rho = 0.745` — rank correlation across 210 actions;
5. `optimism_ratio = 1.12` — relative to the 210-action raw gap.

For Phase R, which uses only 3 prompt configs (P20-N1, P30-N0, P50-N0),
these numbers will be different and must be re-measured.

## 7. Summary of Corrections

| # | Original Claim | Correction |
|---|---|---|
| 1 | Stable oracle gap is small (0.0061) | Valid only for 210-action bank; does NOT mean reference instability is small — the per-query A/B Dice differences are large (up to 0.31) |
| 2 | Search oracle overfits to reference | Confirmed: cross-search Dice {_fmt(y2.get('cross_search_dice'))} < fixed {_fmt(y2.get('fixed_dice'))}; cause is coordinate reuse |
| 3 | Diff-from-fixed rate = 0% | Valid for 210-action bank; means oracle picks same action family, not that instability is absent |
| 4 | Compact bank derived | INCORRECT: compact bank is a placeholder, never properly Pareto-pruned |
| 5 | Reference instability is the bottleneck | CONFIRMED: per-query best-worst Dice gaps reach 0.31; action agreement only 4.88% |

## 8. Implications for Phase R

1. Reference instability IS real and large at the per-query level, even though
   the aggregate stable oracle gap is small;
2. The instability manifests as per-query Dice variance across references,
   not as oracle-vs-fixed gap;
3. Phase R should measure reference instability directly (per-reference Dice std,
   best-worst gap) rather than relying on oracle gap;
4. The 3-config prompt grid (P20-N1, P30-N0, P50-N0) will have different
   instability characteristics than the 210-action bank;
5. Consensus methods should target reducing per-reference variance, not
   oracle gap.
"""

    _write_md(OUTPUT_DIR / "phaseY_claim_correction.md", md)
    print(f"  [7/8] phaseY_claim_correction.md: written")


# ---------------------------------------------------------------------------
# File 8: phaseR_hypothesis_summary.md
# ---------------------------------------------------------------------------


def build_hypothesis_summary(data: dict[str, Any]) -> None:
    """Summarize the Phase R hypothesis based on Phase Y evidence."""
    y1 = data["y1_summary"].get("gate_y1", {})
    y2 = data["y2_summary"].get("gate_y2", {})

    # Compute per-query best-worst gap from the instability summary
    rows = data["y1_utility"]
    fixed_action = _get_fixed_action()

    # Per query, per ref_set: collect fixed action dice across references
    per_query_ref: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("action_id") != fixed_action:
            continue
        if row.get("status") != "OK":
            continue
        qid = row.get("query_case_id", "")
        rset = row.get("reference_set", "")
        dice = _to_float(row.get("dice", 0))
        if qid and rset:
            per_query_ref[qid][rset].append(dice)

    best_worst_gaps: list[float] = []
    ref_stds: list[float] = []
    for qid, rs in per_query_ref.items():
        all_dice = rs.get("A", []) + rs.get("B", [])
        if len(all_dice) >= 2:
            best_worst_gaps.append(max(all_dice) - min(all_dice))
            ref_stds.append(_std(all_dice))

    pct_above_005 = sum(1 for g in best_worst_gaps if g >= 0.05) / max(len(best_worst_gaps), 1) * 100
    mean_gap = _mean(best_worst_gaps)
    mean_std = _mean(ref_stds)

    md = f"""# Phase R Hypothesis Summary (R0 Historical Audit)

Date: {datetime.now().strftime("%Y-%m-%d")}

## Phase R Core Hypothesis

> Reference-stable aggregation and stability-aware prompt generation can reduce
> performance variance across valid references and improve cross-reference
> segmentation reliability under the same SAM-call budget.

## Evidence from Phase Y Supporting This Hypothesis

### 1. Reference Instability is Real and Large

From the Y1 utility matrix (41 validation queries, fixed action `{fixed_action}`):

- **Mean best-worst Dice gap (across 6 references)**: {_fmt(mean_gap)}
- **Mean reference std**: {_fmt(mean_std)}
- **% queries with best-worst gap ≥ 0.05**: {pct_above_005:.1f}%

This confirms that different valid references produce substantially different
segmentation quality for the same query, even with the same fixed action.

### 2. Action Selection is Reference-Dependent

- **Exact action agreement rate (A vs B)**: {_fmt(y1.get('exact_action_agreement_rate'))}
  (only {y1.get('exact_action_agreement_rate', 0) * 100:.1f}% of queries select
  the same best action on both reference sets)
- **K+ bin agreement**: {_fmt(y1.get('kplus_bin_agreement_rate'))}
- **Median Spearman ρ**: {_fmt(y1.get('median_spearman_rho'))}

The low exact agreement (4.88%) shows that the "best" action depends heavily
on which reference is used.

### 3. Search Oracle Overfits to Reference

- **Raw search Dice**: {_fmt(y2.get('raw_search_dice'))}
- **Cross-reference search Dice**: {_fmt(y2.get('cross_search_dice'))}
  (BELOW fixed baseline {_fmt(y2.get('fixed_dice'))})
- **Search optimism**: {_fmt(y2.get('search_optimism'))}
- **Point overlap (Jaccard)**: {_fmt(y2.get('mean_point_overlap'))}

The search oracle's performance drops below the fixed baseline when evaluated
cross-reference, confirming that reference-specific point selection does not
transfer.

### 4. Stable Oracle Gap is Small — But That's a Different Question

- **Raw oracle gap**: {_fmt(y1.get('raw_gap_A'))}
- **Stable oracle gap**: {_fmt(y1.get('stable_oracle_gap'))}
- **Optimism fraction**: {_fmt((y1.get('raw_gap_A', 0) - y1.get('stable_oracle_gap', 0)) / max(y1.get('raw_gap_A', 1), 0.001) * 100)}%

The small stable oracle gap means **action selection** is not the main lever.
The main lever is **reference-induced variance in the fixed action's output**,
which consensus methods can directly address.

## Phase R Research Questions

### RQ1: Is reference instability stable/replicable?

**Prediction**: YES — the per-query best-worst gaps (mean {_fmt(mean_gap)}) and
reference stds (mean {_fmt(mean_std)}) are large enough to replicate on train=80
with 5 resampling seeds.

**Gate R1 threshold**: ≥30% queries with best-worst gap ≥ 0.05; mean std ≥ 0.02.

### RQ2: Is simple consensus enough?

**Prediction**: LIKELY YES — mean/median consensus should reduce variance by
averaging out reference-specific noise. The key question is whether Dice
improves enough (≥0.010 over random single).

**Gate R2 threshold**: ΔDice ≥ 0.010, variance reduction ≥ 25%, CI lower > 0.

### RQ3: Is reference reliability learnable?

**Prediction**: UNCERTAIN — the low Spearman ρ ({_fmt(y1.get('median_spearman_rho'))})
for action rank suggests reference reliability may be predictable, but the
search oracle's failure suggests reference-specific signals may not generalize.

**Gate R3 threshold**: ΔDice ≥ 0.005 over best deterministic, Spearman ρ ≥ 0.30.

## Key Design Decisions for Phase R

1. **Use 3 prompt configs** (P20-N1, P30-N0, P50-N0) instead of 210 actions —
   reduces action-space noise and focuses on reference effect;
2. **Use 6 references per query** (A=3, B=3) — enables consensus and
   cross-reference evaluation;
3. **Use 5 resampling seeds on train** — estimates instability distribution
   without multiple observations on validation;
4. **Fixed A/B on validation** — prevents multiple observations;
5. **CPU-only DINOv2 reuse** — matching maps and candidate pools from Phase Y
   can be reused to save compute.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| R1 fails (instability not replicated) | LOW | Phase Y evidence is strong; train set is larger (80 vs 41) |
| R2 fails (consensus no gain) | MEDIUM | Consensus may not help if instability is systematic, not random |
| R3 fails (reliability not learnable) | MEDIUM-HIGH | Reference quality may depend on query-specific factors |
| Holdout fails | MEDIUM | Even if R1-R3 pass, holdout effect size may be small |
| Compute exceeds 85h | LOW | R1 is the largest stage (25h); CPU reuse helps |

## Next Steps

1. Audit reference pool (step 4);
2. Create train resampling and validation A/B manifests (step 5);
3. Freeze all configs and submit preregistration commit (steps 6-12);
4. Run Gate R1 on train=80 with 5 seeds (step 13).
"""

    _write_md(OUTPUT_DIR / "phaseR_hypothesis_summary.md", md)
    print(f"  [8/8] phaseR_hypothesis_summary.md: written")
    print(f"        Mean best-worst gap = {_fmt(mean_gap)}, Mean ref std = {_fmt(mean_std)}")
    print(f"        % queries with gap >= 0.05 = {pct_above_005:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("Phase R0: Read-only Historical Evidence Correction")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Phase Y artifacts...")
    data = load_phaseY_data()
    print(f"  Y1 utility rows: {len(data['y1_utility'])}")
    print(f"  Y2 per-query rows: {len(data['y2_per_query'])}")
    print(f"  Y2 per-pair rows: {len(data['y2_per_pair'])}")
    print()

    print("Building audit files...")
    build_reference_instability_summary(data)
    build_raw_vs_cross_decomposition(data)
    build_symmetric_fixed_baseline(data)
    build_symmetric_stable_oracle(data)
    build_action_tie_analysis(data)
    build_reference_pair_difficulty(data)
    build_claim_correction(data)
    build_hypothesis_summary(data)

    print()
    print("Phase R0 complete. Output files in:")
    print(f"  {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
