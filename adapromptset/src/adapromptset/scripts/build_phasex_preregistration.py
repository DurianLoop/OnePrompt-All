"""Phase X preregistration builder.

Implements the pre-registration phase of
``AdaPromptSet_ActionSpace_Oracle_Retry.md`` (sections 0-6 + section 25 steps
1-12). This is CPU-only: it reads historical action-bank and oracle artifacts,
writes the preregistration docs/reports, builds the historical bank audit,
appends to the live logs/checklist, and freezes input/protocol hashes.

No new SAM/matcher inference is performed. No formal test is accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Protocol constants (frozen at preregistration time per sections 7-13).
# ---------------------------------------------------------------------------

PHASEX_NAME = "Phase X — Action-Space Adequacy, Dense Prompt Scaling, and Search-Oracle Audit"
PHASEX_BRANCH = "phaseX-action-space-oracle-audit"
PHASEX_COMMIT_MESSAGE = "phaseX: preregister action-space adequacy and search-oracle audit"

# Section 7 candidate scaling grid.
POSITIVE_K_GRID: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 50, 75, 100)
NEGATIVE_K_GRID: tuple[int, ...] = (0, 1, 3, 5, 10, 20, 30, 50)

# Section 8 dense prompt grids.
POSITIVE_SELECTORS: tuple[str, ...] = (
    "P_TOPK",
    "P_LOCAL_PEAKS",
    "P_SPATIAL_NMS",
    "P_SPREAD",
    "P_COMPONENT_BALANCED",
    "P_DIVERSITY_GREEDY",
)
NEGATIVE_SELECTORS: tuple[str, ...] = (
    "N_NONE",
    "N_BOTTOMK",
    "N_HIGH_BACKGROUND",
    "N_BOUNDARY_HARD",
    "N_LOCAL_CONFUSION",
    "N_DIVERSITY",
)
GROUPINGS: tuple[str, ...] = (
    "G_ONE_GROUP",
    "G_ONE_POINT_PER_GROUP",
    "G_FIXED_CHUNK_5",
    "G_FIXED_CHUNK_10",
    "G_FIXED_CHUNK_20",
    "G_KMEANS_3",
    "G_KMEANS_5",
    "G_KMEANS_10",
    "G_KMEANS_20",
    "G_SPATIAL_GRID",
    "G_GRAPH",
    "G_CANDIDATE_COMPONENT",
)
CALL_BUDGETS: tuple[int, ...] = (1, 3, 5, 10, 20)
POINT_BUDGETS: tuple[int, ...] = (5, 10, 20, 50, 100, 150)

# Section 10 nested action banks.
A_SMALL_LIMITS = {"max_positive": 5, "max_negative": 4, "max_calls": 3}
A_MEDIUM_LIMITS = {"max_positive": 20, "max_negative": 10, "max_calls": 5}
A_LARGE_LIMITS = {"max_positive": 100, "max_negative": 30, "max_calls": 20}

# Section 13 search budgets.
SEARCH_BEAM_WIDTHS: tuple[int, ...] = (4, 8)
SEARCH_DEPTHS: tuple[int, ...] = (10, 20, 50)
SEARCH_CALL_BUDGETS: tuple[int, ...] = (1, 3, 5, 10, 20)
SEARCH_POINT_BUDGETS: tuple[int, ...] = (10, 20, 50, 100)

# Section 16 reported budget buckets.
REPORTED_CALL_BUDGETS: tuple[int, ...] = (1, 3, 5, 10, 20)

# Section 17 adaptive value threshold.
ADAPTIVE_VALUE_DICE_THRESHOLD = 0.010
ADAPTIVE_VALUE_MIN_QUERY_FRACTION = 0.25

# Section 21 compute caps (GPU-hours).
COMPUTE_CAPS = {
    "historical_audit": 0.0,
    "candidate_scaling": 4.0,
    "single_variable_k_scaling": 12.0,
    "grouping_scaling": 16.0,
    "negative_scaling": 12.0,
    "coarse_joint_grid": 24.0,
    "successive_halving_validation": 24.0,
    "greedy_beam_search_oracle": 40.0,
    "statistics_and_figures": 0.0,
    "interim_report_threshold": 40.0,
    "reevaluate_search_value_threshold": 90.0,
    "total_initial_cap": 132.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_md(path: Path, text: str) -> None:
    """Append a markdown section to an existing doc, preserving history."""
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    separator = "\n\n" if existing and not existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + text + "\n", encoding="utf-8")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Historical artifact loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalArtifacts:
    bank_v1: dict[str, Any]
    bank_v2: dict[str, Any]
    oracle_v2_summary: dict[str, Any]
    fixed_oracle_summary: dict[str, Any]
    phase5_decision: dict[str, Any]
    gate_f_report: dict[str, Any]
    action_utility_rows: list[dict[str, str]]
    fixed_action_summary_rows: list[dict[str, str]]
    budget_pareto_rows: list[dict[str, str]]
    oracle_heatmap_rows: list[dict[str, str]]
    episodes_val_rows: list[dict[str, str]]
    candidate_pilot_rows: list[dict[str, str]]
    bank_v1_path: Path
    bank_v2_path: Path
    oracle_v2_summary_path: Path
    fixed_oracle_summary_path: Path
    phase5_decision_path: Path
    gate_f_report_path: Path
    action_utility_path: Path
    fixed_action_summary_path: Path
    budget_pareto_path: Path
    oracle_heatmap_path: Path
    episodes_val_path: Path
    candidate_pilot_path: Path


def load_historical_artifacts(project_root: Path) -> HistoricalArtifacts:
    paths = {
        "bank_v1_path": project_root / "configs" / "actions" / "prompt_set_bank_v1.json",
        "bank_v2_path": project_root / "configs" / "actions" / "prompt_set_bank_v2.json",
        "oracle_v2_summary_path": project_root / "reports" / "oracle_gap_pilot_v2" / "oracle_gap_pilot_summary.json",
        "fixed_oracle_summary_path": project_root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json",
        "phase5_decision_path": project_root / "reports" / "decision_checkpoint" / "phase5_decision.json",
        "gate_f_report_path": project_root / "reports" / "gates" / "gate_f_oracle_gap.json",
        "action_utility_path": project_root / "reports" / "oracle_gap_pilot_v2" / "action_utility_pilot.csv",
        "fixed_action_summary_path": project_root / "reports" / "fixed_oracle_analysis" / "action_summary.csv",
        "budget_pareto_path": project_root / "reports" / "fixed_oracle_analysis" / "budget_pareto.csv",
        "oracle_heatmap_path": project_root / "reports" / "decision_checkpoint" / "oracle_action_heatmap.csv",
        "episodes_val_path": project_root / "manifests" / "episodes" / "monusac" / "episodes_val.csv",
        "candidate_pilot_path": project_root / "reports" / "candidate_pilot" / "candidate_pilot_candidates.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "PHASEX_BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE: missing files: " + ", ".join(missing)
        )
    return HistoricalArtifacts(
        bank_v1=_read_json(paths["bank_v1_path"]),
        bank_v2=_read_json(paths["bank_v2_path"]),
        oracle_v2_summary=_read_json(paths["oracle_v2_summary_path"]),
        fixed_oracle_summary=_read_json(paths["fixed_oracle_summary_path"]),
        phase5_decision=_read_json(paths["phase5_decision_path"]),
        gate_f_report=_read_json(paths["gate_f_report_path"]),
        action_utility_rows=_read_csv(paths["action_utility_path"]),
        fixed_action_summary_rows=_read_csv(paths["fixed_action_summary_path"]),
        budget_pareto_rows=_read_csv(paths["budget_pareto_path"]),
        oracle_heatmap_rows=_read_csv(paths["oracle_heatmap_path"]),
        episodes_val_rows=_read_csv(paths["episodes_val_path"]),
        candidate_pilot_rows=_read_csv(paths["candidate_pilot_path"]),
        **paths,
    )


# ---------------------------------------------------------------------------
# Historical bank audit (section 5)
# ---------------------------------------------------------------------------


def _action_definition_rows(bank: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in bank["actions"]:
        positive_count = sum(len(g["positive_indices"]) for g in action["groups"])
        negative_count = sum(len(g["negative_indices"]) for g in action["groups"])
        rows.append(
            {
                "action_id": action["action_id"],
                "family": action["family"],
                "valid": action["valid"],
                "invalid_reason": action.get("invalid_reason", ""),
                "num_groups": len(action["groups"]),
                "num_positive": positive_count,
                "num_negative": negative_count,
                "sam_calls": action["sam_calls"],
                "prompt_points": action["prompt_points"],
                "positive_selection_rule": "topk_by_contrast_score"
                if action["family"]
                in {"single_group_positive", "single_group_positive_negative", "independent_positive_groups"}
                else "topk_split_or_local_neg",
                "negative_selection_rule": "bottomk_by_contrast_score"
                if action["family"] in {"single_group_positive_negative", "rescue_group_specific_negatives"}
                else "none_or_nearest_to_positive",
                "grouping": _family_grouping(action["family"]),
                "cluster_count": _family_cluster_count(action["family"], len(action["groups"])),
                "call_budget": action["sam_calls"],
                "point_budget": action["prompt_points"],
                "fusion": "mvp_area_filter_iou_nms_semantic_union",
                "content_hash": action["content_hash"],
            }
        )
    return rows


def _family_grouping(family: str) -> str:
    return {
        "single_group_positive": "G_ONE_GROUP",
        "single_group_positive_negative": "G_ONE_GROUP",
        "independent_positive_groups": "G_ONE_POINT_PER_GROUP",
        "rescue_split_groups": "G_FIXED_CHUNK_3_2",
        "rescue_group_specific_negatives": "G_FIXED_CHUNK_OR_ONE_PER_GROUP_WITH_LOCAL_NEG",
    }.get(family, family)


def _family_cluster_count(family: str, num_groups: int) -> int:
    if family in {"single_group_positive", "single_group_positive_negative"}:
        return 1
    return num_groups


def _fixed_action_metrics(
    fixed_action_summary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fixed_action_summary_rows:
        rows.append(
            {
                "action_id": row["action_id"],
                "action_family": row["action_family"],
                "mean_dice": float(row["mean_dice"]),
                "mean_iou": float(row["mean_iou"]),
                "mean_aji": float(row["mean_aji"]),
                "mean_sam_calls": float(row["mean_sam_calls"]),
                "mean_runtime_ms": float(row["mean_runtime_ms"]),
                "oracle_count": int(row["oracle_count"]),
                "episode_count": int(row["episode_count"]),
            }
        )
    rows.sort(key=lambda r: r["mean_dice"], reverse=True)
    return rows


def _oracle_query_metrics(
    oracle_heatmap_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in oracle_heatmap_rows:
        rows.append(
            {
                "episode_id": row["episode_id"],
                "action_id": row["action_id"],
                "dice": float(row["dice"]),
                "is_oracle": bool(int(row["is_oracle"])),
            }
        )
    return rows


def _cost_distribution(action_utility_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, str]]] = {}
    for row in action_utility_rows:
        by_action.setdefault(row["action_id"], []).append(row)
    rows: list[dict[str, Any]] = []
    for action_id, items in sorted(by_action.items()):
        sam_calls_values = [int(it["sam_calls"]) for it in items]
        decoder_calls_values = [int(it["decoder_calls"]) for it in items]
        cache_hits_values = [int(it["cache_hits"]) for it in items]
        runtime_values = [float(it["runtime_ms"]) for it in items]
        n = len(items)
        rows.append(
            {
                "action_id": action_id,
                "action_family": items[0]["action_family"],
                "episode_count": n,
                "mean_sam_calls": sum(sam_calls_values) / n,
                "mean_decoder_calls": sum(decoder_calls_values) / n,
                "mean_cache_hits": sum(cache_hits_values) / n,
                "mean_runtime_ms": sum(runtime_values) / n,
                "max_runtime_ms": max(runtime_values),
                "min_runtime_ms": min(runtime_values),
            }
        )
    rows.sort(key=lambda r: r["mean_runtime_ms"])
    return rows


def _absolute_metrics_block(fixed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    best = fixed_metrics[0]
    return {
        "strongest_fixed_action_id": best["action_id"],
        "strongest_fixed_family": best["action_family"],
        "strongest_fixed_mean_dice": best["mean_dice"],
        "strongest_fixed_mean_iou": best["mean_iou"],
        "strongest_fixed_mean_aji": best["mean_aji"],
        "strongest_fixed_mean_sam_calls": best["mean_sam_calls"],
        "strongest_fixed_mean_runtime_ms": best["mean_runtime_ms"],
        "top1_mean_dice": next(
            (r["mean_dice"] for r in fixed_metrics if r["action_id"] == "A_top1_one_group"),
            None,
        ),
        "top3_mean_dice": next(
            (r["mean_dice"] for r in fixed_metrics if r["action_id"] == "A_top3_one_group"),
            None,
        ),
        "top5_mean_dice": next(
            (r["mean_dice"] for r in fixed_metrics if r["action_id"] == "A_top5_one_group"),
            None,
        ),
        "top10_status": "INVALID_REQUIRES_10_POSITIVES",
    }


# ---------------------------------------------------------------------------
# Documentation builders
# ---------------------------------------------------------------------------


def _build_goal_read_receipt(art: HistoricalArtifacts) -> str:
    return f"""# Phase X Goal Read Receipt

Date: {_today()}

## Source directive

`D:\\SAM\\adapromptset\\AdaPromptSet_ActionSpace_Oracle_Retry.md`

## Goal

{PHASEX_NAME}.

## Core scientific question (section 3)

Given the frozen matcher, candidate generator, segmenter, and reference-query
protocol, is the historical 12-action bank sufficient to approximate the prompt
search space? Concretely: for dense, spatially-dispersed segmentation queries,
does optimal prompt performance keep rising as positive count, negative count,
spatial grouping, and SAM-call budget scale up? And in an already-saturated
action space, does query-specific Oracle still exceed the strongest fixed action
by enough to justify policy learning?

## Mandatory corrections acknowledged (section 0)

The historical "Oracle gap = 0.005" claim is renamed to
`FINITE_BANK_ORACLE_GAP_SMALL`. It is *not* a Global Prompt Oracle and *not* a
saturation proof. It only states that within the historical 12-action bank, the
per-query best action beats the strongest fixed action by a small margin.

## Strict ordering acknowledged (section 25)

1. Read historical action-bank/oracle artifacts.
2. Create historical Oracle correction doc.
3. Export old 12-action exact definitions.
4. Export per-action absolute Dice.
5. Pinpoint old strongest fixed exact config.
6. Freeze candidate scaling protocol.
7. Freeze dense prompt scaling protocol.
8. Freeze search oracle protocol.
9. Freeze K/negative/grouping/call grids.
10. Generate config/input hashes.
11. Run focused tests.
12. Preregistration commit.

No new SAM inference runs before the preregistration commit. No formal test
access. No test-driven Top-K, cluster count, or budget changes. MoNuSAC
development/validation scope only.

## Frozen historical anchors

- Historical v2 bank: `configs/actions/prompt_set_bank_v2.json`
  - bank_hash: `{art.bank_v2["bank_hash"]}`
  - action_count: {art.bank_v2["action_count"]}
  - valid_action_count: {art.bank_v2["valid_action_count"]}
- Historical best fixed action: `{art.fixed_oracle_summary["best_fixed_action_id"]}`
- Historical finite-bank oracle mean Dice: {art.fixed_oracle_summary["mean_oracle_dice"]:.6f}
- Historical strongest fixed mean Dice: {art.fixed_oracle_summary["mean_best_fixed_dice"]:.6f}
- Historical finite-bank oracle gap: {art.fixed_oracle_summary["mean_regret_to_oracle"]:.6f}
- Historical Phase 5 status: `{art.phase5_decision["phase5_status"]}`
- MoNuSAC validation episodes used historically: {art.oracle_v2_summary["episode_count"]}
"""


def _build_action_space_problem_statement(art: HistoricalArtifacts) -> str:
    return f"""# Phase X Action-Space Problem Statement

## Problem

The historical MoNuSAC experiment concluded `PIVOT_ORACLE_ANALYSIS` based on a
finite-bank oracle gap of {art.fixed_oracle_summary["mean_regret_to_oracle"]:.6f} Dice.
This conclusion is only valid if the 12-action bank spans the actionable prompt
space. The bank's coverage is:

- max positive K = 5 (Top-10 was *invalid*: required 10 candidates but only 5
  were available from the 5-positive candidate pilot).
- max negative K = 3.
- max SAM calls = 5 (only `C_top5_independent` and `R_top5_independent_localneg`).
- groupings tested: one-group, one-point-per-group, and a single 3+2 split.
- cluster-based groupings (kmeans / spatial grid / graph / candidate-component):
  *never tested*.

The absolute performance of every historical action is low:

| Action | Mean Dice | Notes |
|---|---|---|
| {art.fixed_oracle_summary["best_fixed_action_id"]} (strongest fixed) | {art.fixed_oracle_summary["mean_best_fixed_dice"]:.4f} | 5 positives + 3 negatives, 1 call |
| Finite-bank oracle | {art.fixed_oracle_summary["mean_oracle_dice"]:.4f} | per-query best of 12 |
| A_top1_one_group | ~0.03 | single positive |
| A_top3_one_group | ~0.26 | three positives |
| A_top5_one_group | ~0.33 | five positives |

Two low-performance configurations differing by 0.005 Dice cannot establish
"Oracle is near-optimal". They can only establish "no historical action is much
better than another historical action".

## Hypotheses to test

H1 (Action-space inadequacy). Top-K scaling to 20/50/75/100 positives
significantly improves absolute Dice and component recall.

H2 (Grouping matters at high K). Dense positives benefit from spatial clustering
(kmeans / grid / graph / candidate-component) over a single group or
one-per-group.

H3 (Negative scaling matters at high K). Hard-negative counts of 10/20/30/50
further improve Dice at large positive K, beyond what 0/1/3 negatives achieve.

H4 (Action-space saturation). Beyond some K/calls budget, fixed Dice saturates
and bank-oracle gap saturates; only in this regime can a small gap rule out
adaptive value.

H5 (Search oracle reveals hidden capacity). GT-guided greedy/beam search over
the legal candidate pool and budget finds prompts whose mean Dice exceeds the
finite-bank oracle by >= 0.01.

H6 (Adaptive value). After saturation is established, the matched-budget search
oracle beats the strongest fixed action by >= 0.01 Dice in >= 25% of queries
with morphology-predictable structure.

## Out-of-scope

- Formal test (MoNuSAC test, GlaS testA/testB, IDRID test, DDR test).
- Tuning K, cluster count, or budget on test.
- Promoting GT-guided search outputs into automatic actions without a separate
  train-only abstraction step.
- PPO/DQN policy training (Section 18 explicitly forbids these even when policy
  reopens; only hierarchical classifiers / regressors / routers are allowed).
"""


def _build_historical_oracle_correction(art: HistoricalArtifacts) -> str:
    return f"""# Historical Oracle Correction

Date: {_today()}

## What the historical "Oracle" actually was

The MoNuSAC Phase 5 / Gate F "Oracle" was a **Finite-Bank Oracle**: for each
query, the highest-Dice action among the 12 valid actions in
`configs/actions/prompt_set_bank_v2.json` was selected.

It was *not*:

- A Global Prompt Oracle over all possible positive/negative counts, subsets,
  groupings, and clustering algorithms.
- A search oracle.
- A privileged-coordinate oracle.

## Renaming

The historical claim

> Oracle gap ≈ 0.005

is renamed to

```text
FINITE_BANK_ORACLE_GAP_SMALL
```

It is prohibited to continue abbreviating this as "Oracle gap proves adaptive
prompting is ineffective".

## What the small gap actually implies

Given the historical anchors:

- mean oracle Dice = {art.fixed_oracle_summary["mean_oracle_dice"]:.6f}
- mean best-fixed Dice = {art.fixed_oracle_summary["mean_best_fixed_dice"]:.6f}
- gap = {art.fixed_oracle_summary["mean_regret_to_oracle"]:.6f}
- strongest fixed action = `{art.fixed_oracle_summary["best_fixed_action_id"]}`
- bank max K+ = 5, max K- = 3, max calls = 5
- bank never tested K+ in {{10, 20, 50, 75, 100}}
- bank never tested K- in {{10, 20, 30, 50}}
- bank never tested dense kmeans/grid/graph/component groupings
- bank never tested call budgets above 5

the small gap only implies:

> Within the current 12 historical actions, per-query action selection beats the
> strongest fixed historical action by a small margin.

It does *not* imply:

- Larger positive counts are ineffective.
- More negatives are ineffective.
- Denser spatial coverage is ineffective.
- Top-50 / Top-100 actions are ineffective.
- Other groupings, clusterings, or multi-call strategies are ineffective.
- Adaptive prompt allocation has no value in a broader action space.

## Cross-references

- `reports/phaseX_historical_bank_audit/historical_claim_correction.md`
- `reports/phaseX_historical_bank_audit/old_bank_oracle_summary.md`
- `docs/action_space_problem_statement.md`
"""


def _build_dense_prompt_scaling_protocol() -> str:
    return f"""# Dense Prompt Scaling Protocol (Phase X, Section 8)

Date: {_today()}

## Positive selection strategies (section 8.1)

{", ".join(POSITIVE_SELECTORS)}

Definitions:

- `P_TOPK`: take top-K candidates by contrast score.
- `P_LOCAL_PEAKS`: take local maxima of the contrast score map.
- `P_SPATIAL_NMS`: spatial non-maximum suppression on top candidates.
- `P_SPREAD`: maximize spatial dispersion among high-score candidates.
- `P_COMPONENT_BALANCED`: balance selection across candidate spatial
  connected components.
- `P_DIVERSITY_GREEDY`: greedily pick the highest-scoring candidate least
  redundant with already-selected points.

Query GT must *not* be used for positive selection.

## Negative selection strategies (section 8.2)

{", ".join(NEGATIVE_SELECTORS)}

## Positive count grid (section 8.3)

{", ".join(str(k) for k in POSITIVE_K_GRID)}

If the candidate pool has fewer than the requested K, use the actual maximum,
record the shortfall, never duplicate points, never insert fake points.

## Negative count grid (section 8.4)

{", ".join(str(k) for k in NEGATIVE_K_GRID)}

## Grouping families (section 8.5)

{", ".join(GROUPINGS)}

Definitions follow section 8.5: `G_ONE_GROUP` puts all points in one SAM call;
`G_ONE_POINT_PER_GROUP` makes one SAM call per point; `G_FIXED_CHUNK_n` splits
into chunks of n; `G_KMEANS_k` clusters spatially into k groups;
`G_SPATIAL_GRID` groups by a fixed grid; `G_GRAPH` groups by graph relation;
`G_CANDIDATE_COMPONENT` groups by candidate-pool spatial connected components.

## SAM-call budgets (section 8.6)

{", ".join(str(b) for b in CALL_BUDGETS)}

## Total point budgets (section 8.7)

{", ".join(str(b) for b in POINT_BUDGETS)}

## Staged search (section 9)

Stage S1: single-variable K+ scaling
- selection = `P_TOPK`, negative = `N_NONE`, grouping = `G_ONE_GROUP`, calls = 1
- vary K+ only.

Stage S2: grouping scaling
- pick K+ in {{10, 30, 50, 100}} from S1; sweep grouping family.

Stage S3: negative scaling
- fix S2's best positive/grouping; sweep K- and negative strategy.

Stage S4: joint coarse grid
- 3 positive selectors x 3 K+ x 3 negative settings x 4 groupings x 3 call
  budgets.

Stage S5: successive halving
- all actions run on a small train-only audit subset; eliminate actions that
  are clearly dominated on Dice, component recall, merge rate, SAM calls, and
  runtime; survivors enter full validation. Formal test must not influence
  elimination.

## Fairness rules

Different methods must be compared under matched call budget, matched point
budget, or on an explicit Pareto curve. Per section 16, every action must
report requested vs actual valid positive/negative counts, group count, SAM
calls, wall-clock runtime, VRAM, failed calls, and fusion cost.
"""


def _build_search_oracle_protocol() -> str:
    return f"""# GT-Guided Search Oracle Protocol (Phase X, Section 13)

Date: {_today()}

## Purpose

The historical finite-bank oracle cannot establish action-space adequacy. A
GT-guided search oracle is therefore built on the legal train/validation audit
subset. It is **diagnostic only**: it must not be deployed, must not become an
automatic action, and must not influence the formal test.

## Search variants

### 13.1 Greedy Add

1. Start from a safe anchor (e.g. the strongest fixed historical action).
2. For each remaining positive candidate, evaluate the Dice of adding it.
3. Add the candidate with the largest Dice gain.
4. Repeat until no improvement or the point/call budget is exhausted.
5. Then search negatives in the same greedy manner.
6. Then search grouping changes (merge/split/reassign).

### 13.2 Greedy Add/Remove/Swap

Operations allowed on the current state:
- add positive, remove positive, swap positive
- add negative, remove negative, swap negative
- merge groups, split groups, change group assignment

### 13.3 Beam Search

- beam_width in {{{", ".join(str(b) for b in SEARCH_BEAM_WIDTHS)}}}
- max depth in {{{", ".join(str(d) for d in SEARCH_DEPTHS)}}}
- retain multiple high-Dice candidate states within the budget.

## Search budgets (section 13.4)

- B_call in {{{", ".join(str(b) for b in SEARCH_CALL_BUDGETS)}}}
- B_point in {{{", ".join(str(b) for b in SEARCH_POINT_BUDGETS)}}}

## Comparisons (section 13.5)

Report:

- `D_search_oracle(B)` - mean Dice of the search oracle at budget B.
- `D_search_oracle(B) - D_bank_oracle(B)` - hidden capacity beyond finite bank.
- `D_search_oracle(B) - D_fixed(B)` - adaptive value over strongest fixed.

## Interpretation rules (section 13)

- If `D_search_oracle - D_bank_oracle >= 0.01`:
  `OLD_ACTION_BANK_INADEQUATE`.
- If `D_search_oracle - D_fixed >= 0.01` and best configs differ across
  queries: `POLICY_SPACE_POTENTIALLY_MEANINGFUL`.
- If `D_search_oracle` itself is still low in absolute terms:
  `PROMPT_INTERFACE_OR_CANDIDATE_LIMITATION`.

## Three Oracles (section 14)

Reports must distinguish:

1. **Finite-Bank Oracle** - per-query best within the frozen action bank.
2. **Search Oracle** - GT-guided greedy/beam search over the frozen candidate
   pool and budget.
3. **Privileged Coordinate Oracle** - upper bound using GT
   component/centroid/box prompts.

The unqualified word "Oracle" is prohibited.
"""


def _build_action_space_saturation_protocol() -> str:
    return f"""# Action-Space Saturation Protocol (Phase X, Sections 11-12)

Date: {_today()}

## Nested action banks (section 10)

A_small subset A_medium subset A_large subset A_search_augmented.

| Bank | max K+ | max K- | max calls | Notes |
|---|---|---|---|---|
| A_small | {A_SMALL_LIMITS["max_positive"]} | {A_SMALL_LIMITS["max_negative"]} | {A_SMALL_LIMITS["max_calls"]} | matches historical bank scale |
| A_medium | {A_MEDIUM_LIMITS["max_positive"]} | {A_MEDIUM_LIMITS["max_negative"]} | {A_MEDIUM_LIMITS["max_calls"]} | medium dense prompts |
| A_large | {A_LARGE_LIMITS["max_positive"]} | {A_LARGE_LIMITS["max_negative"]} | {A_LARGE_LIMITS["max_calls"]} | large dense prompts + dense grouping |
| A_search_augmented | >= A_large | >= A_large | >= A_large | + greedy/beam-discovered actions |

A_search_augmented entries must be discovered on the train-only audit subset,
abstracted into a query-GT-free deterministic rule, frozen before validation,
and registered as a single allowed action-bank redesign. GT-guided search
results cannot be silently promoted.

Each action carries a unique `action_id`, `action_hash`, `selection_rule`,
`K+`, `K-`, `grouping`, `cluster_count`, `call_budget`, `point_budget`, and
`fusion`.

## Fixed baseline saturation (section 11)

For each nested bank A_j, compute

```text
D_fixed(A_j) = max over a in A_j of  (1/N) sum_q D(q, a)
```

Report per bank:

- strongest fixed action and exact config
- mean / median Dice
- IoU, AJI
- component recall
- calls, runtime
- win/tie/loss vs strongest fixed of A_small

Plots:

- best fixed Dice vs action-bank size
- best fixed Dice vs max positive K
- best fixed Dice vs point budget
- best fixed Dice vs call budget
- best fixed component recall vs max positive K

Saturation question: does best fixed Dice keep rising from K+ = 5 to K+ = 100?
If the argmax shifts from low K to high K, the historical baseline was
inadequate.

## Finite-bank oracle saturation (section 12)

For each nested bank A_j, compute

```text
D_bank_oracle(A_j) = (1/N) sum_q max over a in A_j of D(q, a)
Delta_bank(A_j) = D_bank_oracle(A_j) - D_fixed(A_j)
```

Plots:

- bank oracle Dice vs action-bank size
- fixed Dice vs action-bank size
- oracle gap vs action-bank size
- oracle gap vs point budget
- oracle gap vs call budget
- oracle action distribution vs target count

## Interpretation (section 12)

| Pattern | Code |
|---|---|
| Fixed and Oracle both keep rising | `ACTION_SPACE_NOT_SATURATED` |
| Oracle rises, Fixed saturates | `ADAPTIVE_HETEROGENEITY_EMERGES` |
| Fixed and Oracle rise together, gap small | `STRONG_FIXED_DENSE_PROMPT_SUFFICIENT` |
| Both saturated, gap small | `ACTION_SPACE_SATURATED_ORACLE_GAP_SMALL` |

Only the last state justifies rejecting adaptive value under the current
budget.

## Absolute performance gate (section 15)

Reports must always include absolute Dice, not just gaps. Joint judgement uses
Dice + component recall + catastrophic failure + merge rate, relative to the
strongest simple baseline.
"""


def _build_formal_test_lock_notice(art: HistoricalArtifacts) -> str:
    return f"""# Phase X Formal Test Lock Notice

Date: {_today()}

## Inheritance

Phase X inherits the formal-test locks recorded in
`docs/formal_test_lock_registry.md`:

| Formal test | Phase X status | Required before access |
|---|---|---|
| GlaS testA | `LOCKED_UNUSED` | `docs/test_freeze_glas.md`, frozen final config, hashes, code commit, metrics, qualitative sampling, cost/statistics plan |
| GlaS testB | `LOCKED_UNUSED` | testA report complete with no method changes; same frozen method |
| DDR test | `LOCKED_UNUSED_CANDIDATE_FRESH_DOMAIN` | DDR four-intent protocol freeze, train/valid-only selection, frozen final method, hashes |
| MoNuSeg formal use | `NOT_FORMAL_BY_DEFAULT` | re-audit official split/patient identity and define role |
| IDRID | `READ_ONLY_NON_TUNABLE` | proof of unobserved subset if any formal claim is desired |

## Phase X prohibitions

Per section 6 of the Phase X directive, the following are prohibited this round:

- Using formal test to select K, cluster count, or budget.
- Using GlaS testA/testB.
- Using IDRID for tuning.
- Repackaging observed test as validation.
- Adding Top-K candidates based on formal-test results.
- Changing cluster count or budget based on formal-test results.

## Scope

Phase X is a **development-stage action-space audit** on MoNuSAC
train/validation only. It is not a formal test paper result. All queries,
references, and splits must use the frozen MoNuSAC manifests.

## Block condition

If the historical manifest is incomplete or unreproducible:

```text
PHASEX_BLOCKED_MANIFEST_INCONSISTENCY
```

and the round stops.

## Current manifest anchor

- MoNuSAC validation episodes: `{art.episodes_val_path}`
- Historical v2 oracle pilot episodes: {art.oracle_v2_summary["episode_count"]}
- Reference exposure audit historical status: `PASS` (per
  `reports/reference_exposure_audit.json`).
"""


# ---------------------------------------------------------------------------
# Reports scaffolding
# ---------------------------------------------------------------------------


def _build_gate_status(art: HistoricalArtifacts) -> str:
    return f"""# Phase X Gate Status

Date: {_today()}

Phase: preregistration (no new SAM inference yet).

## Gate X1 — Historical Baseline Audit

Status: `PENDING_PREREGISTRATION_COMMIT`

Through conditions:

- Old 12-action exact definitions recoverable. -> DONE (see
  `reports/phaseX_historical_bank_audit/action_bank_exact_definitions.csv`).
- Strongest fixed exact config recoverable. -> DONE
  (`{art.fixed_oracle_summary["best_fixed_action_id"]}`).
- All absolute Dice recoverable. -> DONE.
- Cost metadata recoverable. -> DONE.

Failure code if any artifact is missing: `BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE`.

## Gate X2 — Candidate Scaling

Status: `NOT_RUN_YET`.

Will report one of:

- `CANDIDATE_SCALING_SUPPORTS_LARGE_K`
- `CANDIDATE_SCALING_SATURATES_SMALL_K`
- `CANDIDATE_UNIVERSE_INADEQUATE`

## Gate X3 — Dense Prompt Viability

Status: `NOT_RUN_YET`.

Will report one of:

- `DENSE_PROMPT_VIABILITY_ESTABLISHED`
- `DENSE_PROMPT_NO_GAIN`
- `DENSE_PROMPT_COST_ONLY_GAIN`

## Gate X4 — Action-Space Saturation

Status: `NOT_RUN_YET`.

Will report one of:

- `ACTION_SPACE_NOT_SATURATED`
- `ACTION_SPACE_SATURATED`
- `OLD_ACTION_BANK_INADEQUATE`

## Gate X5 — Search-Oracle Adequacy

Status: `NOT_RUN_YET`.

Will report one of:

- `SEARCH_ORACLE_EXCEEDS_BANK_ORACLE`
- `SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE`
- `SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW`

## Gate X6 — Adaptive Value

Status: `NOT_RUN_YET`.

Reopening policy requires *all* of:

1. `D_search_oracle - D_fixed >= {ADAPTIVE_VALUE_DICE_THRESHOLD}` Dice
2. At least {int(ADAPTIVE_VALUE_MIN_QUERY_FRACTION * 100)}% of queries have a
   best structure different from the strongest fixed.
3. The gain is not solely from more calls.
4. Absolute performance reaches a usable level.
5. Action-space saturation is essentially established.
6. Query-specific best config has a predictable relation to morphology /
   candidate statistics.

Pass: `REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE`.

Fail (one of): `PIVOT_STRONG_FIXED_DENSE_PROMPT`,
`PIVOT_PROMPT_INTERFACE_LIMITATION`, `PIVOT_CANDIDATE_LIMITATION`,
`PIVOT_ORACLE_ANALYSIS_EXPANDED_SPACE`.
"""


def _build_compute_ledger() -> str:
    rows = [
        {"phase": "phaseX-preregistration", "stage": "historical_audit", "gpu_hours": "0.0", "command": "scripts/build_phasex_preregistration.py", "notes": "CPU-only read of historical artifacts; no new SAM inference."},
    ]
    fieldnames = ["phase", "stage", "gpu_hours", "command", "notes"]
    output = ",".join(fieldnames) + "\n"
    for row in rows:
        output += ",".join(row[f] for f in fieldnames) + "\n"
    return output


def _build_protocol_snapshot(art: HistoricalArtifacts) -> dict[str, Any]:
    return {
        "phase": "phaseX",
        "name": PHASEX_NAME,
        "branch": PHASEX_BRANCH,
        "preregistration_date": _today(),
        "directive": "D:\\SAM\\adapromptset\\AdaPromptSet_ActionSpace_Oracle_Retry.md",
        "scope": "MoNuSAC train/validation only; no formal test access",
        "frozen_grids": {
            "positive_k": list(POSITIVE_K_GRID),
            "negative_k": list(NEGATIVE_K_GRID),
            "positive_selectors": list(POSITIVE_SELECTORS),
            "negative_selectors": list(NEGATIVE_SELECTORS),
            "groupings": list(GROUPINGS),
            "call_budgets": list(CALL_BUDGETS),
            "point_budgets": list(POINT_BUDGETS),
        },
        "nested_banks": {
            "A_small": A_SMALL_LIMITS,
            "A_medium": A_MEDIUM_LIMITS,
            "A_large": A_LARGE_LIMITS,
        },
        "search_oracle": {
            "beam_widths": list(SEARCH_BEAM_WIDTHS),
            "depths": list(SEARCH_DEPTHS),
            "call_budgets": list(SEARCH_CALL_BUDGETS),
            "point_budgets": list(SEARCH_POINT_BUDGETS),
        },
        "adaptive_value_thresholds": {
            "min_dice_gap": ADAPTIVE_VALUE_DICE_THRESHOLD,
            "min_query_fraction": ADAPTIVE_VALUE_MIN_QUERY_FRACTION,
        },
        "compute_caps_gpu_hours": COMPUTE_CAPS,
        "historical_anchors": {
            "v2_bank_hash": art.bank_v2["bank_hash"],
            "v2_valid_action_count": art.bank_v2["valid_action_count"],
            "strongest_fixed_action_id": art.fixed_oracle_summary["best_fixed_action_id"],
            "finite_bank_oracle_mean_dice": art.fixed_oracle_summary["mean_oracle_dice"],
            "strongest_fixed_mean_dice": art.fixed_oracle_summary["mean_best_fixed_dice"],
            "finite_bank_oracle_gap": art.fixed_oracle_summary["mean_regret_to_oracle"],
            "historical_phase5_status": art.phase5_decision["phase5_status"],
            "historical_episode_count": art.oracle_v2_summary["episode_count"],
        },
    }


def _build_input_hashes(art: HistoricalArtifacts) -> dict[str, Any]:
    paths = {
        "directive": Path("D:\\SAM\\adapromptset\\AdaPromptSet_ActionSpace_Oracle_Retry.md"),
        "bank_v1": art.bank_v1_path,
        "bank_v2": art.bank_v2_path,
        "oracle_v2_summary": art.oracle_v2_summary_path,
        "fixed_oracle_summary": art.fixed_oracle_summary_path,
        "phase5_decision": art.phase5_decision_path,
        "gate_f_report": art.gate_f_report_path,
        "action_utility": art.action_utility_path,
        "fixed_action_summary": art.fixed_action_summary_path,
        "budget_pareto": art.budget_pareto_path,
        "oracle_heatmap": art.oracle_heatmap_path,
        "episodes_val": art.episodes_val_path,
        "candidate_pilot": art.candidate_pilot_path,
    }
    hashes: dict[str, Any] = {}
    for key, path in paths.items():
        if path.is_file():
            hashes[key] = {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        else:
            hashes[key] = {"path": str(path), "sha256": None, "missing": True}
    return {
        "phase": "phaseX",
        "preregistration_date": _today(),
        "input_hashes": hashes,
    }


# ---------------------------------------------------------------------------
# Historical bank audit outputs
# ---------------------------------------------------------------------------


def _build_action_bank_exact_definitions_md(rows: list[dict[str, Any]], bank: dict[str, Any]) -> str:
    lines = [
        "# Phase X Historical Action Bank Exact Definitions",
        "",
        f"Source: `configs/actions/prompt_set_bank_v2.json`",
        f"Bank hash: `{bank['bank_hash']}`",
        f"Action count: {bank['action_count']}",
        f"Valid action count: {bank['valid_action_count']}",
        "",
        "| action_id | family | valid | K+ | K- | groups | calls | points | grouping | cluster_count | positive_selection | negative_selection | fusion | content_hash |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['action_id']} | {r['family']} | {r['valid']} | {r['num_positive']} | {r['num_negative']} | {r['num_groups']} | {r['sam_calls']} | {r['prompt_points']} | {r['grouping']} | {r['cluster_count']} | {r['positive_selection_rule']} | {r['negative_selection_rule']} | {r['fusion']} | `{r['content_hash']}` |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- All historical actions pull positives and negatives from a single 5-positive candidate pilot pool (`reports/candidate_pilot/candidate_pilot_candidates.csv`).",
        "- `A_top10_one_group` is invalid because the pilot only provides 5 positives.",
        "- Cluster-based groupings (kmeans, spatial grid, graph, candidate-component) are *not* represented in the historical bank.",
        "- Call budgets above 5 are *not* represented.",
        "- Negative counts above 3 are *not* represented.",
    ]
    return "\n".join(lines) + "\n"


def _build_fixed_action_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx + 1,
            "action_id": r["action_id"],
            "action_family": r["action_family"],
            "mean_dice": r["mean_dice"],
            "mean_iou": r["mean_iou"],
            "mean_aji": r["mean_aji"],
            "mean_sam_calls": r["mean_sam_calls"],
            "mean_runtime_ms": r["mean_runtime_ms"],
            "oracle_count": r["oracle_count"],
        }
        for idx, r in enumerate(rows)
    ]


def _build_old_bank_oracle_summary_md(
    art: HistoricalArtifacts,
    absolute_block: dict[str, Any],
) -> str:
    return f"""# Phase X Old Bank Oracle Summary

Date: {_today()}

## Historical finite-bank oracle (PIVOT_ORACLE_ANALYSIS anchor)

- Bank: `configs/actions/prompt_set_bank_v2.json`
- Bank hash: `{art.bank_v2['bank_hash']}`
- Valid actions: {art.bank_v2['valid_action_count']}
- Episodes: {art.fixed_oracle_summary['episode_count']}
- Strongest fixed action: `{art.fixed_oracle_summary['best_fixed_action_id']}`
- Strongest fixed mean Dice: {art.fixed_oracle_summary['mean_best_fixed_dice']:.6f}
- Strongest fixed mean IoU: {absolute_block['strongest_fixed_mean_iou']:.6f}
- Strongest fixed mean AJI: {art.fixed_oracle_summary['mean_best_fixed_aji']:.6f}
- Finite-bank oracle mean Dice: {art.fixed_oracle_summary['mean_oracle_dice']:.6f}
- Finite-bank oracle mean AJI: {art.fixed_oracle_summary['mean_oracle_aji']:.6f}
- Finite-bank Dice gap: {art.fixed_oracle_summary['mean_regret_to_oracle']:.6f}
- Finite-bank AJI gap: {art.fixed_oracle_summary['mean_aji_regret_to_oracle']:.6f}
- Median per-episode regret: {art.fixed_oracle_summary['median_regret_to_oracle']:.6f}
- Max per-episode regret: {art.fixed_oracle_summary['max_regret_to_oracle']:.6f}
- Near-oracle episode fraction (tol=0.005): {art.fixed_oracle_summary['near_oracle_episode_fraction']}
- Top oracle action fraction: {art.fixed_oracle_summary['top_oracle_action_fraction']}
- Oracle action distribution: {json.dumps(art.fixed_oracle_summary['oracle_action_counts'], sort_keys=True)}
- Historical Phase 5 status: `{art.phase5_decision['phase5_status']}`

## Top-K absolute Dice

- Top-1 mean Dice: {absolute_block['top1_mean_dice']}
- Top-3 mean Dice: {absolute_block['top3_mean_dice']}
- Top-5 mean Dice: {absolute_block['top5_mean_dice']}
- Top-10 status: `{absolute_block['top10_status']}`

## Bank coverage limits

- Max K+ tested: 5
- Max K- tested: 3
- Max SAM calls tested: 5
- K+ in {{10, 20, 30, 50, 75, 100}}: never tested
- K- in {{10, 20, 30, 50}}: never tested
- Groupings tested: G_ONE_GROUP, G_ONE_POINT_PER_GROUP, G_FIXED_CHUNK_3_2
- Groupings *not* tested: G_KMEANS_*, G_SPATIAL_GRID, G_GRAPH, G_CANDIDATE_COMPONENT
- Call budgets *not* tested: 10, 20

## Conclusion

The historical `0.005` gap is **not** a saturation proof. It is the gap between
two low-absolute-performance configurations (mean Dice ~0.37) inside a narrow
12-action bank. It cannot rule out adaptive value under an expanded action
space.

Renamed historical claim: `FINITE_BANK_ORACLE_GAP_SMALL`.
"""


def _build_historical_claim_correction_md(art: HistoricalArtifacts) -> str:
    return f"""# Phase X Historical Claim Correction

Date: {_today()}

## Old wording (prohibited from now on)

> Oracle gap = 0.005 proves adaptive prompting is ineffective on MoNuSAC.

## New wording (required)

> Within the historical 12-action bank on MoNuSAC validation, the per-query
> best action beats the strongest fixed action by {art.fixed_oracle_summary['mean_regret_to_oracle']:.6f} mean Dice. Both
> configurations have low absolute performance (mean Dice ~{art.fixed_oracle_summary['mean_best_fixed_dice']:.4f}). This
> `FINITE_BANK_ORACLE_GAP_SMALL` result does not establish action-space
> saturation: the bank never tested K+ >= 10, K- >= 10, dense kmeans/grid/
> graph/component groupings, or call budgets above 5.

## Anchor numbers

- mean best fixed Dice: {art.fixed_oracle_summary['mean_best_fixed_dice']:.6f}
- mean finite-bank oracle Dice: {art.fixed_oracle_summary['mean_oracle_dice']:.6f}
- gap: {art.fixed_oracle_summary['mean_regret_to_oracle']:.6f}
- AJI gap: {art.fixed_oracle_summary['mean_aji_regret_to_oracle']:.6f}
- best fixed action: `{art.fixed_oracle_summary['best_fixed_action_id']}`
- v2 bank hash: `{art.bank_v2['bank_hash']}`
- historical Phase 5 status: `{art.phase5_decision['phase5_status']}`

## What is *not* allowed

- Abbreviating the gap as "Oracle gap" without the `FINITE_BANK` qualifier.
- Using the small gap to claim adaptive prompting has no value.
- Using the small gap to justify skipping the action-space saturation audit.
- Lowering the 0.01 adaptive-value threshold after seeing results.

## Cross-references

- `docs/historical_oracle_correction.md`
- `docs/action_space_problem_statement.md`
- `reports/phaseX_historical_bank_audit/old_bank_oracle_summary.md`
"""


def _build_audit_summary_json(
    art: HistoricalArtifacts,
    absolute_block: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "phaseX",
        "stage": "historical_bank_audit",
        "preregistration_date": _today(),
        "historical_bank": {
            "name": "prompt_set_bank_v2",
            "bank_hash": art.bank_v2["bank_hash"],
            "action_count": art.bank_v2["action_count"],
            "valid_action_count": art.bank_v2["valid_action_count"],
            "max_positive_k_tested": 5,
            "max_negative_k_tested": 3,
            "max_sam_calls_tested": 5,
            "untested_positive_k": [10, 20, 30, 50, 75, 100],
            "untested_negative_k": [10, 20, 30, 50],
            "untested_call_budgets": [10, 20],
            "untested_groupings": ["G_KMEANS_3", "G_KMEANS_5", "G_KMEANS_10", "G_KMEANS_20", "G_SPATIAL_GRID", "G_GRAPH", "G_CANDIDATE_COMPONENT"],
        },
        "absolute_performance": absolute_block,
        "finite_bank_oracle": {
            "mean_dice": art.fixed_oracle_summary["mean_oracle_dice"],
            "mean_aji": art.fixed_oracle_summary["mean_oracle_aji"],
            "mean_best_fixed_dice": art.fixed_oracle_summary["mean_best_fixed_dice"],
            "mean_best_fixed_aji": art.fixed_oracle_summary["mean_best_fixed_aji"],
            "dice_gap": art.fixed_oracle_summary["mean_regret_to_oracle"],
            "aji_gap": art.fixed_oracle_summary["mean_aji_regret_to_oracle"],
            "median_regret": art.fixed_oracle_summary["median_regret_to_oracle"],
            "max_regret": art.fixed_oracle_summary["max_regret_to_oracle"],
            "near_oracle_episode_fraction": art.fixed_oracle_summary["near_oracle_episode_fraction"],
            "top_oracle_action_fraction": art.fixed_oracle_summary["top_oracle_action_fraction"],
            "oracle_action_counts": art.fixed_oracle_summary["oracle_action_counts"],
            "oracle_family_counts": art.fixed_oracle_summary["oracle_family_counts"],
        },
        "renamed_claim": "FINITE_BANK_ORACLE_GAP_SMALL",
        "historical_phase5_status": art.phase5_decision["phase5_status"],
    }


# ---------------------------------------------------------------------------
# Live doc appendices
# ---------------------------------------------------------------------------


def _protocol_history_appendix(art: HistoricalArtifacts) -> str:
    return f"""## {_today()} Phase X Action-Space Adequacy / Search-Oracle Audit Preregistration

- Branch `{PHASEX_BRANCH}` created for Phase X.
- Phase X is a development-stage action-space audit on MoNuSAC train/validation only. It does not reopen Phase 4+ Route B work and does not access formal test.
- Historical Oracle claim renamed to `FINITE_BANK_ORACLE_GAP_SMALL`. Historical anchors: v2 bank hash `{art.bank_v2['bank_hash']}`, best fixed `{art.fixed_oracle_summary['best_fixed_action_id']}`, mean best fixed Dice {art.fixed_oracle_summary['mean_best_fixed_dice']:.6f}, mean finite-bank oracle Dice {art.fixed_oracle_summary['mean_oracle_dice']:.6f}, gap {art.fixed_oracle_summary['mean_regret_to_oracle']:.6f}; both ends sit in a low-absolute-performance regime (~0.37 Dice), so the small gap cannot establish action-space saturation.
- Phase X preregistration docs created: `docs/phaseX_goal_read_receipt.md`, `docs/action_space_problem_statement.md`, `docs/historical_oracle_correction.md`, `docs/dense_prompt_scaling_protocol.md`, `docs/search_oracle_protocol.md`, `docs/action_space_saturation_protocol.md`, `docs/formal_test_lock_notice.md`.
- Phase X reports scaffolding created: `reports/phaseX_gate_status.md`, `reports/phaseX_compute_ledger.csv`, `reports/phaseX_protocol_snapshot.json`, `reports/phaseX_input_hashes.json`.
- Historical bank audit created under `reports/phaseX_historical_bank_audit/`: `action_bank_exact_definitions.{{csv,md}}`, `fixed_action_absolute_metrics.csv`, `fixed_action_rankings.csv`, `old_bank_oracle_query_metrics.csv`, `old_bank_oracle_summary.md`, `cost_distribution.csv`, `historical_claim_correction.md`, `audit_summary.json`.
- Frozen grids: K+ in {{{", ".join(str(k) for k in POSITIVE_K_GRID)}}}; K- in {{{", ".join(str(k) for k in NEGATIVE_K_GRID)}}}; groupings = {{{", ".join(GROUPINGS)}}}; call budgets = {{{", ".join(str(b) for b in CALL_BUDGETS)}}}; point budgets = {{{", ".join(str(b) for b in POINT_BUDGETS)}}}; nested banks A_small/A_medium/A_large with limits ({A_SMALL_LIMITS}, {A_MEDIUM_LIMITS}, {A_LARGE_LIMITS}); search beam_widths = {{{", ".join(str(b) for b in SEARCH_BEAM_WIDTHS)}}}, depths = {{{", ".join(str(d) for d in SEARCH_DEPTHS)}}}, call budgets = {{{", ".join(str(b) for b in SEARCH_CALL_BUDGETS)}}}, point budgets = {{{", ".join(str(b) for b in SEARCH_POINT_BUDGETS)}}}; adaptive-value Dice threshold = {ADAPTIVE_VALUE_DICE_THRESHOLD}; min query fraction = {ADAPTIVE_VALUE_MIN_QUERY_FRACTION}; GPU-hour cap = {COMPUTE_CAPS['total_initial_cap']}.
- No new SAM/matcher inference has been run. Preregistration commit must precede candidate coverage scaling, dense prompt scaling, search oracle, and any policy reopening decision.
"""


def _decision_log_appendix(art: HistoricalArtifacts) -> str:
    return f"""| {_today()} | Rename historical MoNuSAC oracle gap claim to `FINITE_BANK_ORACLE_GAP_SMALL` and open Phase X action-space audit | The historical 12-action bank (max K+=5, K-=3, calls=5) never tested K+>=10, K->=10, dense groupings, or calls>5; both fixed and oracle mean Dice sit near 0.37, so a 0.005 gap cannot establish action-space saturation | preregistered |
| {_today()} | Freeze Phase X grids (K+, K-, groupings, calls, points, search budgets, nested banks, adaptive-value threshold) before any new inference | Section 9 forbids full Cartesian search; grids must be frozen at preregistration to prevent post-hoc expansion | locked |
| {_today()} | Keep formal test (MoNuSAC test, GlaS testA/testB, IDRID, DDR test) locked for Phase X | Phase X is a development-stage audit only; section 6 prohibits test-driven K/cluster/budget selection | locked |
| {_today()} | Prohibit GT-guided search outputs from becoming automatic actions without a separate train-only abstraction step | Section 10 requires any search-discovered action be abstracted to a query-GT-free deterministic rule, frozen before validation, and registered as a single allowed action-bank redesign | locked |
| {_today()} | Prohibit PPO/DQN even if policy reopens | Section 18 limits policy to hierarchical classifiers/regressors/routers; PPO/DQN remain out of scope | locked |"""


def _data_exposure_appendix() -> str:
    return f"""| MoNuSAC train/validation | Phase X action-space audit | `manifests/episodes/monusac/episodes_train.csv`; `manifests/episodes/monusac/episodes_val.csv` | tuning_allowed=True within Phase X protocol only | Used for candidate coverage scaling, dense prompt scaling, search oracle, and saturation audit. MoNuSAC test remains locked. |
| Phase X preregistration artifacts | CPU-only audit + docs | `docs/phaseX_*.md`; `reports/phaseX_*`; `reports/phaseX_historical_bank_audit/*` | tuning_allowed=False | Read-only summaries of historical artifacts; no new model inference; no formal-test access. |"""


def _checklist_appendix() -> str:
    return f"""## Phase X — Action-Space Adequacy / Search-Oracle Audit (2026-07-20)

- [x] Read all historical action-bank and oracle artifacts before preregistration.
  - {_today()}: Read `configs/actions/prompt_set_bank_v1.json`, `configs/actions/prompt_set_bank_v2.json`, `reports/oracle_gap_pilot_v2/*`, `reports/fixed_oracle_analysis/*`, `reports/decision_checkpoint/*`, `reports/gates/gate_f_oracle_gap.*`, `reports/candidate_pilot/candidate_pilot_candidates.csv`.
- [x] Create historical Oracle correction doc.
  - {_today()}: `docs/historical_oracle_correction.md` and `reports/phaseX_historical_bank_audit/historical_claim_correction.md` rename the old claim to `FINITE_BANK_ORACLE_GAP_SMALL`.
- [x] Export old 12-action exact definitions.
  - {_today()}: `reports/phaseX_historical_bank_audit/action_bank_exact_definitions.{{csv,md}}`.
- [x] Export each historical action's absolute Dice / IoU / AJI / cost.
  - {_today()}: `reports/phaseX_historical_bank_audit/fixed_action_absolute_metrics.csv` and `fixed_action_rankings.csv`.
- [x] Pinpoint historical strongest fixed exact config.
  - {_today()}: `B_top5_pos3neg` (5 positives + 3 negatives in one group, 1 SAM call, MVP fusion). Mean Dice = 0.371238.
- [x] Freeze candidate scaling / dense prompt scaling / search oracle / saturation protocols.
  - {_today()}: `docs/dense_prompt_scaling_protocol.md`, `docs/search_oracle_protocol.md`, `docs/action_space_saturation_protocol.md`, `docs/formal_test_lock_notice.md`, `docs/action_space_problem_statement.md`, `docs/phaseX_goal_read_receipt.md`.
- [x] Freeze K / negative / grouping / call / point / search grids.
  - {_today()}: Recorded in `reports/phaseX_protocol_snapshot.json` and `reports/phaseX_input_hashes.json`.
- [x] Generate config/input hashes.
  - {_today()}: `reports/phaseX_input_hashes.json` records SHA256 of directive, v1/v2 banks, oracle summaries, decision checkpoint, gate F report, action utility CSV, fixed action summary CSV, budget Pareto CSV, oracle heatmap CSV, MoNuSAC validation episodes CSV, candidate pilot CSV.
- [ ] Run focused tests on the preregistration builder.
- [ ] Create preregistration git commit `{PHASEX_COMMIT_MESSAGE}` before any new SAM inference.
- [ ] Run candidate coverage scaling (Stage S1 / Gate X2).
- [ ] Run single-variable K+ scaling.
- [ ] Run grouping scaling.
- [ ] Run negative scaling.
- [ ] Run coarse joint grid + successive halving.
- [ ] Build nested action banks A_small ⊂ A_medium ⊂ A_large ⊂ A_search_augmented.
- [ ] Compute fixed baseline saturation.
- [ ] Compute finite-bank oracle saturation.
- [ ] Run GT-guided greedy/beam search oracle.
- [ ] Compare bank oracle vs search oracle under matched budgets.
- [ ] Generate Phase X tables, figures, and `reports/phaseX_final_report.{{md,json}}`.
- [ ] Output the unique final Phase X state (one of the section 24 conclusions).

Current Phase X status: `PREREGISTRATION_PENDING_COMMIT`.
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_phasex_preregistration(project_root: Path) -> dict[str, Any]:
    art = load_historical_artifacts(project_root)

    docs_dir = project_root / "docs"
    reports_dir = project_root / "reports"
    audit_dir = reports_dir / "phaseX_historical_bank_audit"

    # --- Preregistration docs ---
    _write_text(docs_dir / "phaseX_goal_read_receipt.md", _build_goal_read_receipt(art))
    _write_text(docs_dir / "action_space_problem_statement.md", _build_action_space_problem_statement(art))
    _write_text(docs_dir / "historical_oracle_correction.md", _build_historical_oracle_correction(art))
    _write_text(docs_dir / "dense_prompt_scaling_protocol.md", _build_dense_prompt_scaling_protocol())
    _write_text(docs_dir / "search_oracle_protocol.md", _build_search_oracle_protocol())
    _write_text(docs_dir / "action_space_saturation_protocol.md", _build_action_space_saturation_protocol())
    _write_text(docs_dir / "formal_test_lock_notice.md", _build_formal_test_lock_notice(art))

    # --- Reports scaffolding ---
    _write_text(reports_dir / "phaseX_gate_status.md", _build_gate_status(art))
    _write_text(reports_dir / "phaseX_compute_ledger.csv", _build_compute_ledger())
    _write_json(reports_dir / "phaseX_protocol_snapshot.json", _build_protocol_snapshot(art))
    _write_json(reports_dir / "phaseX_input_hashes.json", _build_input_hashes(art))

    # --- Historical bank audit ---
    definition_rows = _action_definition_rows(art.bank_v2)
    fixed_metrics = _fixed_action_metrics(art.fixed_action_summary_rows)
    oracle_query_metrics = _oracle_query_metrics(art.oracle_heatmap_rows)
    cost_distribution = _cost_distribution(art.action_utility_rows)
    absolute_block = _absolute_metrics_block(fixed_metrics)
    fixed_rankings = _build_fixed_action_rankings(fixed_metrics)

    definition_fieldnames = [
        "action_id", "family", "valid", "invalid_reason",
        "num_groups", "num_positive", "num_negative",
        "sam_calls", "prompt_points",
        "positive_selection_rule", "negative_selection_rule",
        "grouping", "cluster_count",
        "call_budget", "point_budget",
        "fusion", "content_hash",
    ]
    _write_csv(
        audit_dir / "action_bank_exact_definitions.csv",
        definition_rows,
        definition_fieldnames,
    )
    _write_text(
        audit_dir / "action_bank_exact_definitions.md",
        _build_action_bank_exact_definitions_md(definition_rows, art.bank_v2),
    )

    fixed_metrics_fieldnames = [
        "rank", "action_id", "action_family",
        "mean_dice", "mean_iou", "mean_aji",
        "mean_sam_calls", "mean_runtime_ms",
        "oracle_count", "episode_count",
    ]
    _write_csv(
        audit_dir / "fixed_action_absolute_metrics.csv",
        fixed_metrics,
        [
            "action_id", "action_family",
            "mean_dice", "mean_iou", "mean_aji",
            "mean_sam_calls", "mean_runtime_ms",
            "oracle_count", "episode_count",
        ],
    )
    _write_csv(
        audit_dir / "fixed_action_rankings.csv",
        fixed_rankings,
        fixed_metrics_fieldnames,
    )
    _write_csv(
        audit_dir / "old_bank_oracle_query_metrics.csv",
        oracle_query_metrics,
        ["episode_id", "action_id", "dice", "is_oracle"],
    )
    _write_text(
        audit_dir / "old_bank_oracle_summary.md",
        _build_old_bank_oracle_summary_md(art, absolute_block),
    )
    _write_csv(
        audit_dir / "cost_distribution.csv",
        cost_distribution,
        [
            "action_id", "action_family", "episode_count",
            "mean_sam_calls", "mean_decoder_calls", "mean_cache_hits",
            "mean_runtime_ms", "max_runtime_ms", "min_runtime_ms",
        ],
    )
    _write_text(
        audit_dir / "historical_claim_correction.md",
        _build_historical_claim_correction_md(art),
    )
    _write_json(
        audit_dir / "audit_summary.json",
        _build_audit_summary_json(art, absolute_block),
    )

    # --- Append to live docs (never overwrite) ---
    _append_md(docs_dir / "protocol_history.md", _protocol_history_appendix(art))
    _append_md(docs_dir / "decision_log.md", _decision_log_appendix(art))
    _append_md(docs_dir / "data_exposure_log.md", _data_exposure_appendix())
    _append_md(project_root / "checklist.md", _checklist_appendix())

    return {
        "phase": "phaseX",
        "stage": "preregistration",
        "status": "PREREGISTRATION_ARTIFACTS_READY",
        "preregistration_date": _today(),
        "branch": PHASEX_BRANCH,
        "commit_message": PHASEX_COMMIT_MESSAGE,
        "artifacts": {
            "docs": [
                str(docs_dir / "phaseX_goal_read_receipt.md"),
                str(docs_dir / "action_space_problem_statement.md"),
                str(docs_dir / "historical_oracle_correction.md"),
                str(docs_dir / "dense_prompt_scaling_protocol.md"),
                str(docs_dir / "search_oracle_protocol.md"),
                str(docs_dir / "action_space_saturation_protocol.md"),
                str(docs_dir / "formal_test_lock_notice.md"),
            ],
            "reports": [
                str(reports_dir / "phaseX_gate_status.md"),
                str(reports_dir / "phaseX_compute_ledger.csv"),
                str(reports_dir / "phaseX_protocol_snapshot.json"),
                str(reports_dir / "phaseX_input_hashes.json"),
            ],
            "audit": [
                str(audit_dir / "action_bank_exact_definitions.csv"),
                str(audit_dir / "action_bank_exact_definitions.md"),
                str(audit_dir / "fixed_action_absolute_metrics.csv"),
                str(audit_dir / "fixed_action_rankings.csv"),
                str(audit_dir / "old_bank_oracle_query_metrics.csv"),
                str(audit_dir / "old_bank_oracle_summary.md"),
                str(audit_dir / "cost_distribution.csv"),
                str(audit_dir / "historical_claim_correction.md"),
                str(audit_dir / "audit_summary.json"),
            ],
            "appended": [
                str(docs_dir / "protocol_history.md"),
                str(docs_dir / "decision_log.md"),
                str(docs_dir / "data_exposure_log.md"),
                str(project_root / "checklist.md"),
            ],
        },
        "historical_anchors": _build_protocol_snapshot(art)["historical_anchors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase X preregistration artifacts (CPU-only).")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    summary = build_phasex_preregistration(args.project_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
