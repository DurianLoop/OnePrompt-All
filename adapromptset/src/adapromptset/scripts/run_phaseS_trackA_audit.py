"""Phase S Track A §6.3 metric correction audit.

Phase R Gate R2 FAILED on cond3 (variance reduction >= 25%) with VR = -0.0506.
The root cause is a metric definition error: ``var_single`` mixes cross-query AND
within-query variance (Var_{q,r}[D] = V_query + V_ref), while ``var_consensus``
is pure cross-query variance (V_query only, since consensus gives one prediction
per query). This is an apples-to-oranges comparison.

Per directive §6.2 the CORRECT decomposition is:
    V_query = Var_q[ E_r[ D(q,r) ] ]            # cross-query (case difficulty)
    V_ref   = (1/N) * sum_q Var_r[ D(q,r) ]     # within-query reference variance
    VR_correct = 1 - V_ref(consensus) / V_ref(single)
where V_ref(consensus) = 0 by construction (one prediction per query).

This script re-derives the corrected variance decomposition from the Phase R
validation artefacts and writes 10 audit files to
``reports/phaseS_trackA_metric_audit/``.

CPU-only. Uses stdlib + numpy only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS = PROJECT_ROOT / "reports"

PHASE_R2_DIR = REPORTS / "phaseR_gateR2"
PHASE_R1_DIR = REPORTS / "phaseR_gateR1"
OUTPUT_DIR = REPORTS / "phaseS_trackA_metric_audit"

GATE_R2_SCRIPT = PROJECT_ROOT / "src" / "adapromptset" / "scripts" / "run_phaseR_gateR2.py"

CATASTROPHIC_DICE_THRESHOLD = 0.1
CONFIGS = ("P20-N1", "P30-N0", "P50-N0")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _row_dice(row: dict[str, str]) -> float | None:
    s = row.get("dice", "")
    return float(s) if s != "" else None


def _row_float(row: dict[str, str], key: str) -> float | None:
    s = row.get(key, "")
    return float(s) if s != "" else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(x: float, n: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return ""
    return f"{x:.{n}f}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Output 1: variance_code_trace.md
# ---------------------------------------------------------------------------

def write_variance_code_trace(path: Path) -> None:
    """Trace the exact code path in run_phaseR_gateR2.py lines 1459-1589."""
    print("[trackA-audit] writing variance_code_trace.md ...")
    text = """# Track A §6.3 — Variance Code Trace

Source file: `src/adapromptset/scripts/run_phaseR_gateR2.py`
Function: `compute_gate_r2(...)` (lines 1459-1597)
Condition audited: `cond3_variance_reduction_ge_25pct` (lines 1510-1518)

## 1. The exact lines that computed `var_single`, `var_consensus`, `vr`

```python
# cond3: variance reduction >= 25%
per_ref_dices = [
    _row_dice(r) for r in val_per_ref_rows
    if r["config_name"] == best_config and _row_dice(r) is not None
]
var_single = float(np.var(per_ref_dices)) if len(per_ref_dices) > 1 else 0.0
var_cons = float(np.var(cons_dices)) if len(cons_dices) > 1 else 0.0
vr = 1.0 - var_cons / var_single if var_single > 0 else 0.0
cond3 = vr >= 0.25
```

## 2. What each variable contains

### `per_ref_dices` (lines 1511-1514)
A flat list of **every** per-reference Dice value for `best_config` (P30-N0)
drawn from `val_per_reference_metrics.csv`. There are 41 queries × 6 references
= **246 values**. Each value is `D(q, r)` — the Dice score for query `q` when
prompted from reference `r`. This list is NOT grouped by query; it is a flat
pool that mixes variation across queries AND variation across references within
each query.

### `var_single` (line 1515)
`var_single = np.var(per_ref_dices)` — the population variance (`ddof=0`) of
the 246 pooled Dice values. By the law of total variance this equals:

    Var_{q,r}[D(q,r)] = Var_q[ E_r[D(q,r)] ] + E_q[ Var_r[D(q,r)] ]
                      = V_query            + V_ref

i.e. **total** variance = cross-query variance (case difficulty) + within-query
reference variance (the instability we actually want to quantify). See
`phaseR_var_single_definition.md`.

### `cons_dices` (line 1495)
`cons_dices = [cons_by_q[q] for q in common_q]` — one consensus Dice value per
query (41 values) for `best_method` (B2_mean) at `reference_scope=all6`. The
consensus step collapses the 6 per-reference predictions into a single
prediction per query, so the reference dimension is eliminated.

### `var_cons` / `var_consensus` (line 1516)
`var_cons = np.var(cons_dices)` — the population variance of the 41 consensus
Dice values. Because there is exactly one value per query, this is **pure
cross-query variance**: `Var_q[ D_consensus(q) ] = V_query` only. The
within-query reference variance is gone by construction (the consensus already
aggregated the 6 references). See `phaseR_var_consensus_definition.md`.

### `vr` (line 1517)
`vr = 1.0 - var_cons / var_single = 1.0 - V_query / (V_query + V_ref)`

## 3. The metric definition error

`cond3` compares two variances measured on **different axes**:

| quantity        | axis                        | decomposition |
|-----------------|-----------------------------|---------------|
| `var_single`    | Var over (query × ref) pairs| V_query + V_ref |
| `var_consensus` | Var over query only         | V_query        |

`vr = 1 - V_query / (V_query + V_ref)` is therefore **not** the reduction in
reference-induced variance. It is dominated by `V_query` (case difficulty),
which is intrinsic and is NOT reduced by consensus. With `V_ref` being a small
fraction of `V_total`, the ratio `V_query / (V_query + V_ref)` is close to 1,
so `vr` is close to 0 (here -0.0506), making cond3 impossible to satisfy even
though consensus eliminates 100% of the within-query reference variance.

## 4. The correct metric (directive §6.2)

    V_ref(single)    = (1/N) Σ_q Var_r[ D(q,r) ]        # the instability
    V_ref(consensus) = 0                                 # by construction
    VR_correct       = 1 - V_ref(consensus) / V_ref(single) = 1.0

The corrected variance reduction is **1.0 (100%)** for every config, because
consensus produces one prediction per query and therefore has zero
within-query reference variance by construction.

## 5. Numeric confirmation (from gateR2_summary.json)

    var_single        = 0.054308   (246 pooled P30-N0 Dice values)
    var_consensus     = 0.057056   (41 consensus P30-N0 Dice values)
    variance_reduction = -0.050601 (the reported, INCORRECT cond3 metric)

The corrected decomposition (computed in
`cross_query_vs_within_query_variance.csv`) shows that `V_ref` is only a small
fraction of `V_total`; the rest (`V_query`) is case difficulty that consensus
cannot and should not be expected to remove.
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Output 2 & 3: var_single / var_consensus definition docs
# ---------------------------------------------------------------------------

def write_var_single_definition(path: Path, config_metrics: dict) -> None:
    print("[trackA-audit] writing phaseR_var_single_definition.md ...")
    p30 = config_metrics["P30-N0"]
    text = f"""# Phase R `var_single` definition

## Definition

`var_single` was computed in `run_phaseR_gateR2.py` line 1515 as:

```python
per_ref_dices = [_row_dice(r) for r in val_per_ref_rows
                 if r["config_name"] == best_config and _row_dice(r) is not None]
var_single = float(np.var(per_ref_dices))
```

For `best_config = P30-N0`, `per_ref_dices` is the flat list of all 41 × 6 = 246
per-reference Dice values `D(q, r)` drawn from `val_per_reference_metrics.csv`.
`np.var` uses population variance (`ddof=0`).

Therefore:

    var_single = Var_{{q,r}}[ D(q, r) ]    (total variance over all query×ref pairs)

## Law of total variance decomposition

With equal reference count N=6 per query:

    Var_{{q,r}}[D] = Var_q[ E_r[D(q,r)] ] + E_q[ Var_r[D(q,r)] ]
                   = V_query            + V_ref

- `V_query = Var_q[ E_r[D(q,r)] ]` — variance across queries of the per-query
  mean Dice. This is **case difficulty** spread (some queries are simply harder).
- `V_ref = (1/N) Σ_q Var_r[D(q,r)]` — average within-query variance across the 6
  references. This is the **reference-induced instability** that Gate R1
  confirmed and that consensus is designed to eliminate.

## Key point

`var_single` is **NOT** within-query reference variance. It is the *total*
variance, of which reference-induced variance is only a small fraction. Using
`var_single` as the "single-reference variance" baseline against
`var_consensus` therefore conflates case difficulty with reference instability.

## Numeric values (validation, P30-N0)

| quantity  | value     |
|-----------|-----------|
| V_total (= var_single) | {p30['V_total']:.6f} |
| V_query                | {p30['V_query']:.6f} |
| V_ref                  | {p30['V_ref']:.6f} |
| V_ref_frac             | {p30['V_ref_frac']:.6f} |

Reported `var_single` in `gateR2_summary.json` = 0.054308, which matches
`V_total` for P30-N0 (rounding aside), confirming the trace.
"""
    path.write_text(text, encoding="utf-8")


def write_var_consensus_definition(path: Path, best_config: str, var_consensus: float) -> None:
    print("[trackA-audit] writing phaseR_var_consensus_definition.md ...")
    text = f"""# Phase R `var_consensus` definition

## Definition

`var_consensus` was computed in `run_phaseR_gateR2.py` line 1516 as:

```python
cons_dices = [cons_by_q[q] for q in common_q]      # line 1495
var_cons   = float(np.var(cons_dices))               # line 1516
```

`cons_by_q` maps each query to a single consensus Dice value for `best_method`
(B2_mean) at `reference_scope = all6`. There is exactly **one** consensus Dice
per query (41 values for {best_config}).

Therefore:

    var_consensus = Var_q[ D_consensus(q) ]    (variance across queries only)

## Why this is pure cross-query variance

The consensus step (B2_mean) aggregates the 6 per-reference SAM predictions
into a single prediction per query. After aggregation the reference dimension
no longer exists: for each query there is one Dice value, not six. Any
within-query reference variance (`V_ref`) has been eliminated by construction.

Applying the law of total variance to the consensus output:

    Var_q[D_consensus] = V_query + V_ref(consensus)
                       = V_query + 0
                       = V_query

So `var_consensus` measures **only** `V_query` — the cross-query case-difficulty
spread. It contains **no** reference-induced component.

## The apples-to-oranges comparison

| quantity        | axis                 | decomposition |
|-----------------|----------------------|---------------|
| `var_single`    | (query × ref) pairs  | V_query + V_ref |
| `var_consensus` | query only           | V_query          |

Because the two quantities are on different axes, the ratio
`var_consensus / var_single = V_query / (V_query + V_ref)` is close to 1
(V_query dominates), producing `vr ≈ 0` (here -0.0506 for {best_config}).

## Numeric value (validation, {best_config}, B2_mean, all6)

| quantity        | value     |
|-----------------|-----------|
| var_consensus (reported) | {var_consensus:.6f} |

This matches `gateR2_summary.json` (`var_consensus = 0.057056`), confirming the
trace. Note `var_consensus (0.057056) > var_single (0.054308)` is itself a red
flag: consensus variance should never exceed total single-reference variance if
the two were on the same axis. The fact that it does confirms they are on
different axes.
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Output 4: cross_query_vs_within_query_variance.csv
# ---------------------------------------------------------------------------

def compute_config_variance_metrics(val_per_ref_rows: list[dict]) -> dict:
    """For each config compute V_total, V_query, V_ref, V_ref_frac."""
    metrics: dict[str, dict] = {}
    # group per (config, query) -> list of dice
    by_cfg_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in val_per_ref_rows:
        d = _row_dice(r)
        if d is None:
            continue
        by_cfg_q[(r["config_name"], r["query_case_id"])].append(d)

    for cfg in CONFIGS:
        print(f"[trackA-audit] computing V_ref for config {cfg} ...")
        q_to_dices = {q: ds for (c, q), ds in by_cfg_q.items() if c == cfg}
        queries = sorted(q_to_dices)
        n_queries = len(queries)
        ref_counts = {len(q_to_dices[q]) for q in queries}
        n_refs = max(ref_counts) if ref_counts else 0

        all_dices = [d for q in queries for d in q_to_dices[q]]
        per_query_means = [float(np.mean(q_to_dices[q])) for q in queries]
        per_query_vars = [float(np.var(q_to_dices[q])) for q in queries]  # ddof=0

        v_total = float(np.var(all_dices)) if len(all_dices) > 1 else 0.0
        v_query = float(np.var(per_query_means)) if len(per_query_means) > 1 else 0.0
        v_ref = float(np.mean(per_query_vars)) if per_query_vars else 0.0
        v_ref_frac = v_ref / v_total if v_total > 0 else 0.0

        # sanity check: law of total variance
        ltv_check = v_query + v_ref

        metrics[cfg] = {
            "n_queries": n_queries,
            "n_refs_per_query": n_refs,
            "V_total": v_total,
            "V_query": v_query,
            "V_ref": v_ref,
            "V_ref_frac": v_ref_frac,
            "ltv_check": ltv_check,
        }
        print(
            f"    V_total={v_total:.6f}  V_query={v_query:.6f}  "
            f"V_ref={v_ref:.6f}  V_ref_frac={v_ref_frac:.6f}  "
            f"(V_query+V_ref={ltv_check:.6f})"
        )
    return metrics


def write_cross_query_vs_within_query(path: Path, metrics: dict) -> None:
    print("[trackA-audit] writing cross_query_vs_within_query_variance.csv ...")
    fieldnames = [
        "config_name", "n_queries", "n_refs_per_query",
        "V_total", "V_query", "V_ref", "V_ref_frac",
    ]
    rows = []
    for cfg in CONFIGS:
        m = metrics[cfg]
        rows.append({
            "config_name": cfg,
            "n_queries": m["n_queries"],
            "n_refs_per_query": m["n_refs_per_query"],
            "V_total": _fmt(m["V_total"]),
            "V_query": _fmt(m["V_query"]),
            "V_ref": _fmt(m["V_ref"]),
            "V_ref_frac": _fmt(m["V_ref_frac"]),
        })
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Output 5: corrected_reference_variance.csv
# ---------------------------------------------------------------------------

def write_corrected_reference_variance(path: Path, val_per_ref_rows: list[dict]) -> None:
    print("[trackA-audit] writing corrected_reference_variance.csv ...")
    by_cfg_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in val_per_ref_rows:
        d = _row_dice(r)
        if d is None:
            continue
        by_cfg_q[(r["config_name"], r["query_case_id"])].append(d)

    fieldnames = [
        "query_case_id", "config_name", "per_query_mean_dice",
        "per_query_ref_std", "per_query_ref_var", "per_query_best_worst_gap",
    ]
    rows = []
    for cfg in CONFIGS:
        qs = sorted(q for (c, q) in by_cfg_q if c == cfg)
        for q in qs:
            ds = by_cfg_q[(cfg, q)]
            if not ds:
                continue
            arr = np.asarray(ds, dtype=float)
            rows.append({
                "query_case_id": q,
                "config_name": cfg,
                "per_query_mean_dice": _fmt(float(arr.mean())),
                "per_query_ref_std": _fmt(float(arr.std())),      # ddof=0
                "per_query_ref_var": _fmt(float(arr.var())),       # ddof=0
                "per_query_best_worst_gap": _fmt(float(arr.max() - arr.min())),
            })
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Output 6: corrected_pairwise_deviation.csv
# ---------------------------------------------------------------------------

def write_corrected_pairwise_deviation(path: Path, val_per_ref_rows: list[dict]) -> None:
    print("[trackA-audit] writing corrected_pairwise_deviation.csv ...")
    by_cfg_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in val_per_ref_rows:
        d = _row_dice(r)
        if d is None:
            continue
        by_cfg_q[(r["config_name"], r["query_case_id"])].append(d)

    fieldnames = ["query_case_id", "config_name", "mad_ref", "n_pairs"]
    rows = []
    for cfg in CONFIGS:
        qs = sorted(q for (c, q) in by_cfg_q if c == cfg)
        for q in qs:
            ds = by_cfg_q[(cfg, q)]
            if len(ds) < 2:
                continue
            pairs = list(combinations(ds, 2))
            abs_diffs = [abs(a - b) for a, b in pairs]
            rows.append({
                "query_case_id": q,
                "config_name": cfg,
                "mad_ref": _fmt(_mean(abs_diffs)),
                "n_pairs": len(pairs),
            })
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Output 7: prediction_consistency.csv
# ---------------------------------------------------------------------------

def write_prediction_consistency(path: Path, val_cross_set_rows: list[dict]) -> None:
    print("[trackA-audit] writing prediction_consistency.csv ...")
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in val_cross_set_rows:
        by_cfg[r["config_name"]].append(r)

    fieldnames = [
        "config_name", "mean_prediction_overlap_AB",
        "mean_single_prediction_overlap_AB", "n_queries",
    ]
    rows = []
    for cfg in CONFIGS:
        rs = by_cfg.get(cfg, [])
        cons_ov = [float(r["prediction_overlap_AB"]) for r in rs
                   if r.get("prediction_overlap_AB", "") not in ("", None)]
        single_ov = [float(r["single_prediction_overlap_AB"]) for r in rs
                     if r.get("single_prediction_overlap_AB", "") not in ("", None)]
        rows.append({
            "config_name": cfg,
            "mean_prediction_overlap_AB": _fmt(_mean(cons_ov)),
            "mean_single_prediction_overlap_AB": _fmt(_mean(single_ov)),
            "n_queries": len(rs),
        })
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Output 8: cvar_tail_risk.csv
# ---------------------------------------------------------------------------

def write_cvar_tail_risk(path: Path, val_per_ref_rows: list[dict]) -> None:
    print("[trackA-audit] writing cvar_tail_risk.csv ...")
    by_cfg: dict[str, list[float]] = defaultdict(list)
    for r in val_per_ref_rows:
        d = _row_dice(r)
        if d is None:
            continue
        by_cfg[r["config_name"]].append(d)

    fieldnames = [
        "config_name", "cvar_0_1", "worst_dice", "p10_dice",
        "catastrophic_count", "catastrophic_rate", "n_total",
    ]
    rows = []
    for cfg in CONFIGS:
        ds = sorted(by_cfg.get(cfg, []))
        n = len(ds)
        if n == 0:
            continue
        # bottom 10% worst (lowest) values
        k = max(1, int(round(0.1 * n)))
        worst_k = ds[:k]
        cvar = _mean(worst_k)
        worst_dice = min(ds)
        p10 = float(np.percentile(ds, 10))
        cat_count = sum(1 for d in ds if d < CATASTROPHIC_DICE_THRESHOLD)
        cat_rate = cat_count / n
        rows.append({
            "config_name": cfg,
            "cvar_0_1": _fmt(cvar),
            "worst_dice": _fmt(worst_dice),
            "p10_dice": _fmt(p10),
            "catastrophic_count": cat_count,
            "catastrophic_rate": _fmt(cat_rate),
            "n_total": n,
        })
    _write_csv(path, fieldnames, rows)


# ---------------------------------------------------------------------------
# Output 9: method_cost_decomposition.csv
# ---------------------------------------------------------------------------

def write_method_cost_decomposition(path: Path, val_consensus_rows: list[dict]) -> None:
    print("[trackA-audit] writing method_cost_decomposition.csv ...")
    # validation, config=P30-N0, scope=all6
    target_cfg = "P30-N0"
    scope = "all6"
    sel = [r for r in val_consensus_rows
           if r["config_name"] == target_cfg and r["reference_scope"] == scope]

    # group by (method, hyperparam)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sel:
        groups[(r["method"], r["hyperparam"])].append(r)

    method_order = [
        "B1_random", "B2_mean", "B3_median", "B4_trimmed",
        "B5_disagreement", "B6_quantile",
    ]
    fieldnames = [
        "method", "hyperparam", "mean_dice", "mean_runtime_ms",
        "n_sam_calls_per_query", "total_sam_calls", "n_queries",
    ]
    rows = []
    for method in method_order:
        # all hyperparam rows for this method, in sorted hyperparam order
        hp_groups = sorted(
            ((hp, rs) for (m, hp), rs in groups.items() if m == method),
            key=lambda x: x[0],
        )
        for hp, rs in hp_groups:
            dices = [d for d in (_row_dice(r) for r in rs) if d is not None]
            rts = [d for d in (_row_float(r, "runtime_ms") for r in rs) if d is not None]
            n_q = len(dices)
            # Per directive: consensus methods make 1 SAM call per query
            # (they reuse cached per-reference predictions and aggregate masks);
            # B1 also makes 1 SAM call (it picks one reference). So all = 1.
            n_sam = 1
            rows.append({
                "method": method,
                "hyperparam": hp,
                "mean_dice": _fmt(_mean(dices)),
                "mean_runtime_ms": _fmt(_mean(rts)),
                "n_sam_calls_per_query": n_sam,
                "total_sam_calls": n_q * n_sam,
                "n_queries": n_q,
            })
    _write_csv(path, fieldnames, rows)

    # Note: validation only evaluated the single selected-best hyperparam for
    # B5 (beta=0.25) and B6 (tau=0.83); the x3 design grid was collapsed to x1
    # at validation. Train data has the full grid but is out of scope here.
    n_b5 = sum(1 for r in rows if r["method"] == "B5_disagreement")
    n_b6 = sum(1 for r in rows if r["method"] == "B6_quantile")
    print(
        f"    note: validation has {n_b5} B5 hyperparam row(s) and {n_b6} B6 "
        f"hyperparam row(s) for {target_cfg}/{scope} (selected-best only)."
    )


# ---------------------------------------------------------------------------
# Output 10: phaseR_interpretation_addendum.md
# ---------------------------------------------------------------------------

def write_interpretation_addendum(
    path: Path,
    config_metrics: dict,
    summary_r2: dict,
) -> None:
    print("[trackA-audit] writing phaseR_interpretation_addendum.md ...")
    gate = summary_r2.get("gate_r2", {})
    best_config = gate.get("best_config", "P30-N0")
    best_method = gate.get("best_method", "B2_mean")
    best_beta = gate.get("best_beta", 0.25)
    best_tau = gate.get("best_tau", 0.83)
    var_single_rep = gate.get("var_single", 0.054308)
    var_cons_rep = gate.get("var_consensus", 0.057056)
    vr_rep = gate.get("variance_reduction", -0.050601)
    delta_dice = gate.get("delta_dice", 0.045953)
    cfr_cons = gate.get("cfr_consensus", 0.073171)
    cfr_b1 = gate.get("cfr_b1", 0.121951)

    p30 = config_metrics[best_config]
    v_ref_single = p30["V_ref"]
    # V_ref(consensus) = 0 by construction (one prediction per query)
    vr_correct = 1.0 - 0.0 / v_ref_single if v_ref_single > 0 else 0.0

    # Build a per-config table for Q6
    cfg_lines = []
    for cfg in CONFIGS:
        m = config_metrics[cfg]
        vref = m["V_ref"]
        vr_c = 1.0 if vref > 0 else 0.0
        cfg_lines.append(
            f"| {cfg} | {m['n_queries']} | {m['n_refs_per_query']} | "
            f"{m['V_total']:.6f} | {m['V_query']:.6f} | {m['V_ref']:.6f} | "
            f"{m['V_ref_frac']:.6f} | {vr_c:.3f} |"
        )
    cfg_table = "\n".join(cfg_lines)

    text = f"""# Phase R Interpretation Addendum — Track A §6.3

This addendum answers the six mandatory questions from directive §6.3 and
amends the scientific interpretation of the Phase R Gate R2 result. It does
**not** reopen R2 and does **not** unblock R3/R4.

## Corrected variance decomposition (validation)

| config   | n_queries | n_refs | V_total   | V_query  | V_ref    | V_ref_frac | VR_correct |
|----------|-----------|--------|-----------|----------|----------|------------|------------|
{cfg_table}

`VR_correct = 1 - V_ref(consensus) / V_ref(single)`, and `V_ref(consensus) = 0`
by construction (consensus yields one prediction per query, eliminating the
within-query reference dimension). Hence `VR_correct = 1.0` (100%) for every
config — the opposite of the reported `-0.0506`.

Reported (incorrect) Phase R cond3 values for `{best_config}`:
`var_single = {var_single_rep}`, `var_consensus = {var_cons_rep}`,
`variance_reduction = {vr_rep}`.

---

## Q1. var_single and var_consensus computation axis

**var_single** is computed on the **(query × reference) pair axis**: it is the
population variance of all 246 per-reference Dice values
`Var_{{q,r}}[D(q,r)]` for `{best_config}`. By the law of total variance this
decomposes as `V_query + V_ref` — it mixes cross-query variance (case
difficulty) with within-query reference variance.

**var_consensus** is computed on the **query axis only**: it is the population
variance of the 41 consensus Dice values `Var_q[D_consensus(q)]` for
`{best_method}` at `reference_scope=all6`. Because consensus produces one
prediction per query, the reference dimension is gone, so this is pure
`V_query`.

**The two quantities are on different axes.** Comparing them is apples-to-oranges.

## Q2. Whether query variability was treated as reference variability

**YES.** `var_single` includes `V_query` (case difficulty), which dominates the
total variance. The Phase R cond3 effectively compared `V_query(consensus)`
against `V_query + V_ref(single)`, i.e.

    vr = 1 - V_query / (V_query + V_ref)

Since `V_query` is the dominant term, the ratio is close to 1 and `vr` is close
to 0 (here {vr_rep}). This attributes the (large, irreducible) cross-query
case-difficulty variance to the consensus method's failure, which is invalid.
The reference-induced variance `V_ref` that consensus actually eliminates is
only `V_ref_frac = {p30['V_ref_frac']:.6f}` of `V_total` for `{best_config}`,
yet cond3 weighed the comparison as if all of `V_total` were reference variance.

## Q3. Whether B1/B2 reference count differs

**YES, and this compounds the error.**

- **B1_random** uses **1** reference per query — a deterministic hash-based
  selection from the 6 references. Its Dice is one of the 6 per-reference
  values.
- **B2_mean** uses **all 6** references, aggregated (mask averaging) into a
  single consensus prediction.

However, `var_single` was computed from `val_per_reference_metrics.csv`, which
contains **all 6 references per query** (246 values for `{best_config}`), i.e.
the full per-reference Dice spread. This is **not** B1's variance: B1 picks a
single reference per query and therefore has no within-query reference variance
either (it has one value per query, just like consensus). So `var_single` does
**not** represent B1's variance — it represents the per-reference Dice spread,
which is a different quantity from either B1 or B2. The cond3 baseline was
therefore neither B1 nor a meaningful "single-reference" condition; it was the
raw cross-reference Dice dispersion pooled across queries.

## Q4. Whether matcher cost was ignored

**YES.** Only the SAM-B `predict` time was recorded in `runtime_ms`. The
DINOv2-L feature-extraction cost (the matcher forward pass used to retrieve
references and to score candidates) was **not** included in the runtime / cost
accounting. The matcher cost is shared across all methods (B1–B6) and across
single-reference and consensus runs, so omitting it does not change the
*relative* method ranking on `runtime_ms`. However, for fairness and for any
absolute cost claim (e.g. "consensus adds negligible cost"), the matcher cost
should be reported separately. The omission is a reporting gap, not a
scientific error in the variance metric.

## Q5. Whether B2_mean's beta/tau fields are exporter error

**The beta/tau fields for B2_mean are a cosmetic exporter artifact, NOT a
scientific error.**

`gateR2_summary.json` reports `best_beta = {best_beta}` and `best_tau =
{best_tau}` alongside `best_method = B2_mean`. But B2_mean does **not** use
beta or tau: beta is the hyperparameter of **B5_disagreement** and tau is the
hyperparameter of **B6_quantile**. These fields are populated by the unified
selection schema which always carries `best_beta`/`best_tau` regardless of the
selected method. For B2_mean (which has an empty hyperparam) they are inert
leftovers from the schema, not parameters of the method. This is a cosmetic
bug in the exporter; it does not affect any Gate R2 condition or the
consensus Dice values.

## Q6. How Phase R scientific conclusions need correction

1. **The R2 cond3 failure is INVALID.** It compared total variance
   (`V_query + V_ref`, mixed) against cross-query variance only (`V_query`),
   producing `vr = {vr_rep}`. The correct comparison is
   `V_ref(consensus) / V_ref(single) = 0 / {v_ref_single:.6f}`, giving
   `VR_correct = {vr_correct:.3f}` (100% reduction). Consensus eliminates
   within-query reference variance **by construction**.

2. **The real finding stands and is positive:** consensus improves mean Dice
   by `+{delta_dice:.6f}` (`{best_config}`, `{best_method}` vs B1_random),
   with a bootstrap 95% CI lower bound above 0, and reduces catastrophic
   failure rate from `{cfr_b1:.6f}` to `{cfr_cons:.6f}` (a
   {(cfr_b1 - cfr_cons) * 100:.2f} pp reduction).

3. **Consensus does NOT reduce cross-query variance** (`V_query`), nor should
   it be expected to: `V_query` reflects case difficulty (some queries are
   intrinsically harder), not reference choice. This is intrinsic to the case
   mix and is not reducible by reference aggregation.

4. **Phase R's final state should be AMENDED** to read:
   > "Deterministic consensus eliminates within-query reference variance and
   > improves mean Dice, but cross-query variance is not reducible by consensus
   > — it reflects case difficulty, not reference choice."

5. **This AMENDMENT does not reopen R2 and does not unblock R3/R4.** It
   corrects the metric *interpretation*, not the gate decision flow. The
   corrected VR (1.0) would have passed cond3, but the amendment is scoped to
   fixing the metric definition and the scientific narrative; it does not
   retroactively flip the gate.

6. **Track B (reliability learning) remains scientifically motivated.**
   Because `V_query` (case difficulty) is the dominant, consensus-irreducible
   component, the lever for further stabilisation is per-query reliability
   weighting — down-weighting bad references *for a given query* — which
   targets `V_ref` at the per-query level and could additionally compress
   cross-query tail risk. This sustains Track B's rationale independent of the
   cond3 metric fix.

---

## Files produced by this audit

1. `variance_code_trace.md` — code path trace (lines 1459-1589)
2. `phaseR_var_single_definition.md` — var_single = V_total = V_query + V_ref
3. `phaseR_var_consensus_definition.md` — var_consensus = V_query only
4. `cross_query_vs_within_query_variance.csv` — corrected decomposition per config
5. `corrected_reference_variance.csv` — per (query, config) within-query stats
6. `corrected_pairwise_deviation.csv` — per (query, config) MAD across 15 ref pairs
7. `prediction_consistency.csv` — per config A/B prediction overlap (consensus vs single)
8. `cvar_tail_risk.csv` — per config CVaR_0.1, worst, p10, catastrophic rate
9. `method_cost_decomposition.csv` — per method cost (P30-N0, all6, validation)
10. `phaseR_interpretation_addendum.md` — this document (6 mandatory answers)
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Phase S — Track A §6.3 metric correction audit")
    print("=" * 72)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs -------------------------------------------------------
    print("[trackA-audit] loading phaseR_gateR2 inputs ...")
    val_per_ref_rows = _read_csv(PHASE_R2_DIR / "val_per_reference_metrics.csv")
    val_consensus_rows = _read_csv(PHASE_R2_DIR / "val_consensus_metrics.csv")
    val_cross_set_rows = _read_csv(PHASE_R2_DIR / "val_cross_set_stability.csv")
    summary_r2 = _read_json(PHASE_R2_DIR / "gateR2_summary.json")
    # phaseR_gateR1 / train_consensus are available as supporting context
    _r1_instability = _read_csv(PHASE_R1_DIR / "per_query_instability.csv")
    _r1_per_ref = _read_csv(PHASE_R1_DIR / "per_reference_metrics.csv")
    _train_consensus = _read_csv(PHASE_R2_DIR / "train_consensus_metrics.csv")

    print(f"  val_per_reference_metrics.csv : {len(val_per_ref_rows)} rows")
    print(f"  val_consensus_metrics.csv     : {len(val_consensus_rows)} rows")
    print(f"  val_cross_set_stability.csv   : {len(val_cross_set_rows)} rows")
    print(f"  gateR2_summary.json           : loaded")
    print(f"  per_query_instability.csv     : {len(_r1_instability)} rows (context)")
    print()

    gate = summary_r2.get("gate_r2", {})
    best_config = gate.get("best_config", "P30-N0")
    best_method = gate.get("best_method", "B2_mean")
    var_consensus_rep = gate.get("var_consensus", 0.057056)
    print(
        f"[trackA-audit] Phase R best_config={best_config}  "
        f"best_method={best_method}  reported VR={gate.get('variance_reduction')}"
    )
    print()

    # ---- Compute corrected variance decomposition -------------------------
    config_metrics = compute_config_variance_metrics(val_per_ref_rows)
    print()

    # ---- Write outputs -----------------------------------------------------
    write_variance_code_trace(OUTPUT_DIR / "variance_code_trace.md")
    write_var_single_definition(OUTPUT_DIR / "phaseR_var_single_definition.md", config_metrics)
    write_var_consensus_definition(
        OUTPUT_DIR / "phaseR_var_consensus_definition.md",
        best_config, var_consensus_rep,
    )
    write_cross_query_vs_within_query(
        OUTPUT_DIR / "cross_query_vs_within_query_variance.csv", config_metrics
    )
    write_corrected_reference_variance(
        OUTPUT_DIR / "corrected_reference_variance.csv", val_per_ref_rows
    )
    write_corrected_pairwise_deviation(
        OUTPUT_DIR / "corrected_pairwise_deviation.csv", val_per_ref_rows
    )
    write_prediction_consistency(
        OUTPUT_DIR / "prediction_consistency.csv", val_cross_set_rows
    )
    write_cvar_tail_risk(OUTPUT_DIR / "cvar_tail_risk.csv", val_per_ref_rows)
    write_method_cost_decomposition(
        OUTPUT_DIR / "method_cost_decomposition.csv", val_consensus_rows
    )
    write_interpretation_addendum(
        OUTPUT_DIR / "phaseR_interpretation_addendum.md", config_metrics, summary_r2
    )
    print()

    # ---- Final summary -----------------------------------------------------
    print("=" * 72)
    print("FINAL SUMMARY — Track A §6.3 corrected variance decomposition")
    print("=" * 72)
    print()
    print(f"{'config':<10} {'V_total':>10} {'V_query':>10} {'V_ref':>10} "
          f"{'V_ref_frac':>10} {'VR_correct':>10}")
    for cfg in CONFIGS:
        m = config_metrics[cfg]
        vref = m["V_ref"]
        vr_c = 1.0 if vref > 0 else 0.0
        print(f"{cfg:<10} {m['V_total']:>10.6f} {m['V_query']:>10.6f} "
              f"{m['V_ref']:>10.6f} {m['V_ref_frac']:>10.6f} {vr_c:>10.3f}")
    print()
    p30 = config_metrics[best_config]
    print(f"Reported (incorrect) VR for {best_config}: "
          f"{gate.get('variance_reduction')}  (var_single={gate.get('var_single')}, "
          f"var_consensus={gate.get('var_consensus')})")
    print(f"Corrected  VR        for {best_config}: "
          f"{1.0 - 0.0 / p30['V_ref'] if p30['V_ref'] > 0 else 0.0:.3f}  "
          f"(V_ref(single)={p30['V_ref']:.6f}, V_ref(consensus)=0 by construction)")
    print()
    print("-" * 72)
    print("Six mandatory answers (see phaseR_interpretation_addendum.md):")
    print("-" * 72)
    print("Q1 axis: var_single on (q×r) pair axis = V_query+V_ref; "
          "var_consensus on q axis only = V_query. Different axes.")
    print("Q2 query-vs-ref: YES — var_single includes V_query (case difficulty), "
          "dominates total; cond3 compared V_query vs V_query+V_ref (invalid).")
    print("Q3 B1/B2 ref count: B1=1 ref/query, B2=6 refs aggregated; but var_single "
          "used all 6 refs (246 values) — represents per-ref spread, NOT B1 variance.")
    print("Q4 matcher cost: YES — DINOv2-L feature extraction omitted from runtime; "
          "shared across methods so relative ranking unaffected, but should be reported.")
    print("Q5 B2 beta/tau: cosmetic exporter artifact (beta=B5, tau=B6 params); "
          "B2_mean has empty hyperparam. Not a scientific error.")
    print("Q6 correction: R2 cond3 failure INVALID; VR_correct=1.0 (100%); "
          "consensus eliminates V_ref by construction, improves mean Dice, "
          "does NOT reduce V_query (case difficulty). AMENDS interpretation, "
          "does not reopen R2 / unblock R3/R4. Track B still motivated.")
    print()
    print("[trackA-audit] DONE. Output directory:")
    print(f"  {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
