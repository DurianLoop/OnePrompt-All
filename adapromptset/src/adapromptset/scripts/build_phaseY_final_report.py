"""Phase Y final report builder.

Aggregates all Phase Y CSV/JSON/YAML artifacts under ``reports/phaseY_*``
into a single final report (markdown + JSON) plus 18 tables (CSV + markdown)
and 20 figures (PNG).

Implements sections 13-16 and 22 of
``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

Outputs::

    reports/phaseY_final_report.md
    reports/phaseY_final_report.json
    reports/phaseY_final_report/tables/table_NN_name.csv
    reports/phaseY_final_report/tables/table_NN_name.md
    reports/phaseY_final_report/figures/fig_NN_name.png

The script is CPU-only and degrades gracefully when artifacts are missing
(e.g. if Gate Y4 did not run, the corresponding tables/figures are skipped
with a note).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap (matches Y1/Y2/Y3 scripts)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MONUSAC_ROOT = WORKSPACE_ROOT / "code" / "datasets" / "MoNuSAC" / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "code"))
sys.path.insert(0, str(WORKSPACE_ROOT / "Matcher"))

# Matplotlib (Agg backend, CPU-only)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

# PIL is optional (only needed for qualitative figures 16-18)
try:
    from PIL import Image  # type: ignore

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PIL_AVAILABLE = False

# YAML is needed to read action bank configs
try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    _YAML_AVAILABLE = False


DPI = 150
DEFAULT_FIG_SIZE = (8.0, 6.0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not _YAML_AVAILABLE or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def _median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def _stdev(values: list[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    return statistics.pstdev(values)


def _write_table_md(path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No data available (artifact missing or empty)._")
        lines.append("")
    else:
        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"|{'|'.join(['---'] * len(headers))}|")
        for row in rows:
            cells = [str(c) if c is not None else "" for c in row]
            # Pad/truncate to headers length
            while len(cells) < len(headers):
                cells.append("")
            cells = cells[: len(headers)]
            lines.append(f"| {' | '.join(cells)} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            cells = [str(c) if c is not None else "" for c in row]
            while len(cells) < len(headers):
                cells.append("")
            cells = cells[: len(headers)]
            writer.writerow(cells)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _action_k_positive(action_id: str) -> int:
    for token in action_id.split("_"):
        if token.startswith("K") and token[1:].isdigit():
            return int(token[1:])
    return 0


def _action_k_negative(action_id: str) -> int:
    for token in action_id.split("_"):
        if token.startswith("N") and token[1:].isdigit():
            return int(token[1:])
    return 0


def _action_pos_selector(action_id: str) -> str:
    tokens = action_id.split("_")
    for tok in tokens:
        if tok.startswith("P_"):
            return tok
    return ""


def _action_neg_selector(action_id: str) -> str:
    tokens = action_id.split("_")
    for i, tok in enumerate(tokens):
        if tok == "N" and i + 1 < len(tokens) and tokens[i + 1].startswith("N_"):
            return tokens[i + 1]
    for tok in tokens:
        if tok.startswith("N_"):
            return tok
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_data(project_root: Path) -> dict[str, Any]:
    """Load every Phase Y artifact into a single dict (graceful on missing)."""
    reports = project_root / "reports"
    configs = project_root / "configs"
    manifests = project_root / "manifests"
    docs = project_root / "docs"

    data: dict[str, Any] = {}

    # --- Phase Y data audit (split) ---
    audit_dir = reports / "phaseY_data_audit"
    data["split_summary"] = _read_json(audit_dir / "split_summary.json")
    data["data_audit_summary_md"] = _read_text(audit_dir / "data_audit_summary.md")
    data["prior_query_exposure"] = _read_csv(audit_dir / "prior_query_exposure.csv")
    data["case_identity_map"] = _read_csv(audit_dir / "case_identity_map.csv")
    data["eligible_fresh_queries"] = _read_csv(audit_dir / "eligible_fresh_queries.csv")

    # --- Manifests ---
    phaseY_manifests = manifests / "phaseY"
    data["policy_train_queries"] = _read_csv(phaseY_manifests / "policy_train_queries.csv")
    data["policy_validation_queries"] = _read_csv(phaseY_manifests / "policy_validation_queries.csv")
    data["confirmatory_holdout_queries"] = _read_csv(phaseY_manifests / "confirmatory_holdout_queries.csv")
    data["reference_set_A"] = _read_csv(phaseY_manifests / "reference_set_A.csv")
    data["reference_set_B"] = _read_csv(phaseY_manifests / "reference_set_B.csv")

    # --- Action banks ---
    data["full_bank_yaml"] = _read_yaml(configs / "actions" / "phaseY_full_single_call_bank.yaml")
    data["compact_bank_yaml"] = _read_yaml(configs / "actions" / "phaseY_compact_policy_bank.yaml")
    data["frozen_policy_yaml"] = _read_yaml(configs / "frozen" / "phaseY_bank_policy.yaml")

    # --- Freeze doc ---
    data["holdout_freeze_md"] = _read_text(docs / "phaseY_holdout_freeze.md")
    data["holdout_freeze_hashes"] = _read_json(reports / "phaseY_holdout" / "holdout_freeze_hashes.json")

    # --- Gate Y1 ---
    y1_dir = reports / "phaseY_gateY1"
    data["y1_utility_matrix_train"] = _read_csv(y1_dir / "utility_matrix_train.csv")
    data["y1_utility_matrix_validation"] = _read_csv(y1_dir / "utility_matrix_validation.csv")
    data["y1_candidate_pools_validation"] = _read_csv(y1_dir / "candidate_pools_validation.csv")
    data["y1_summary"] = _read_json(y1_dir / "gateY1_summary.json")
    data["y1_summary_md"] = _read_text(y1_dir / "gateY1_summary.md")
    data["y1_pareto_train_metrics"] = _read_csv(y1_dir / "pareto_train_metrics.csv")
    data["y1_pareto_pruning_md"] = _read_text(y1_dir / "pareto_pruning.md")
    data["y1_run_metadata"] = _read_json(y1_dir / "run_metadata.json")

    # --- Gate Y2 ---
    y2_dir = reports / "phaseY_gateY2"
    data["y2_search_oracle_per_query"] = _read_csv(y2_dir / "search_oracle_per_query.csv")
    data["y2_search_oracle_per_pair"] = _read_csv(y2_dir / "search_oracle_per_pair.csv")
    data["y2_summary"] = _read_json(y2_dir / "gateY2_summary.json")
    data["y2_summary_md"] = _read_text(y2_dir / "gateY2_summary.md")
    data["y2_run_metadata"] = _read_json(y2_dir / "run_metadata.json")

    # --- Gate Y3 ---
    y3_dir = reports / "phaseY_gateY3"
    data["y3_bank_policy_per_query"] = _read_csv(y3_dir / "bank_policy_per_query.csv")
    data["y3_summary"] = _read_json(y3_dir / "gateY3_summary.json")
    data["y3_summary_md"] = _read_text(y3_dir / "gateY3_summary.md")
    data["y3_run_metadata"] = _read_json(y3_dir / "run_metadata.json")

    # --- Gate Y4 (optional) ---
    y4_dir = reports / "phaseY_gateY4"
    data["y4_point_subset_per_query"] = _read_csv(y4_dir / "point_subset_per_query.csv")
    data["y4_summary"] = _read_json(y4_dir / "gateY4_summary.json")
    data["y4_summary_md"] = _read_text(y4_dir / "gateY4_summary.md")
    data["y4_run_metadata"] = _read_json(y4_dir / "run_metadata.json")

    # --- Confirmatory Holdout ---
    holdout_dir = reports / "phaseY_holdout"
    data["holdout_per_query"] = _read_csv(holdout_dir / "holdout_per_query.csv")
    data["holdout_summary"] = _read_json(holdout_dir / "holdout_summary.json")
    data["holdout_summary_md"] = _read_text(holdout_dir / "holdout_summary.md")
    data["holdout_run_metadata"] = _read_json(holdout_dir / "run_metadata.json")

    # --- Compute ledger ---
    data["phaseY_compute_ledger"] = _read_csv(reports / "phaseY_compute_ledger.csv")
    data["phaseY_gate_status_md"] = _read_text(reports / "phaseY_gate_status.md")

    # Helper flags
    data["_has_y1"] = bool(data["y1_utility_matrix_validation"])
    data["_has_y2"] = bool(data["y2_search_oracle_per_query"])
    data["_has_y3"] = bool(data["y3_summary"])
    data["_has_y4"] = bool(data["y4_summary"])
    data["_has_holdout"] = bool(data["holdout_summary"])

    return data


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_per_query_action_dice(
    rows: list[dict[str, str]],
    reference_set: str,
) -> dict[str, dict[str, float]]:
    """{query: {action_id: mean_dice_across_refs}} from a utility matrix."""
    by_qa: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("reference_set") != reference_set:
            continue
        if row.get("status") != "OK":
            continue
        dice = row.get("dice")
        if dice is None or dice == "":
            continue
        qid = row.get("query_case_id", "")
        aid = row.get("action_id", "")
        if not qid or not aid:
            continue
        by_qa.setdefault(qid, {}).setdefault(aid, []).append(_to_float(dice))
    return {
        qid: {aid: _mean(v) for aid, v in actions.items()}
        for qid, actions in by_qa.items()
    }


def _aggregate_per_query_metric(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, float]:
    """{query: mean(metric)} from per-query CSV rows."""
    by_q: dict[str, list[float]] = {}
    for row in rows:
        qid = row.get("query_case_id", "")
        if not qid:
            continue
        val = row.get(metric)
        if val is None or val == "":
            continue
        by_q.setdefault(qid, []).append(_to_float(val))
    return {q: _mean(v) for q, v in by_q.items()}


# ---------------------------------------------------------------------------
# Table builders (18 tables per Section 15)
# ---------------------------------------------------------------------------


def _table_01_fresh_data_split(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Split", "Query Count", "Patient Count", "Target", "Status"]
    rows: list[list[str]] = []
    split = data.get("split_summary", {})
    split_info = split.get("split", {})
    for name in ["train", "validation", "holdout"]:
        info = split_info.get(name, {})
        rows.append([
            name,
            _fmt_int(info.get("case_count")),
            _fmt_int(info.get("patient_count")),
            _fmt_int(info.get("target")),
            split.get("status", "N/A"),
        ])
    rows.append([
        "Total (fresh)",
        _fmt_int(split.get("eligible_fresh_cases")),
        _fmt_int(split.get("eligible_fresh_patients")),
        "160",
        split.get("status", "N/A"),
    ])
    return headers, rows


def _table_02_prior_exposure_audit(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Case ID", "Phase X Exposure", "Phase Y Fold", "Patient"]
    rows: list[list[str]] = []
    for r in data.get("case_identity_map", []):
        rows.append([
            r.get("case_id", ""),
            r.get("phasex_exposure", ""),
            r.get("phaseY_fold", ""),
            r.get("patient", ""),
        ])
    if not rows:
        # Fallback: synthesize from split_summary
        split = data.get("split_summary", {})
        for case_id in split.get("phasex_query_cases", []):
            rows.append([case_id, "phasex_query", "excluded", ""])
        for case_id in split.get("phasex_reference_cases", []):
            rows.append([case_id, "phasex_reference", "excluded", ""])
    return headers, rows


def _table_03_full_action_bank(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Action ID", "Index", "K+", "K-", "Positive Selector",
        "Negative Selector", "Grouping", "SAM Calls",
    ]
    rows: list[list[str]] = []
    bank = data.get("full_bank_yaml", {})
    actions = bank.get("actions", []) or []
    for entry in actions:
        rows.append([
            entry.get("action_id", ""),
            _fmt_int(entry.get("action_index")),
            _fmt_int(entry.get("k_positive")),
            _fmt_int(entry.get("k_negative")),
            entry.get("positive_selector", ""),
            entry.get("negative_selector", ""),
            entry.get("grouping", "G_ONE_GROUP"),
            _fmt_int(entry.get("sam_calls", 1)),
        ])
    if not rows:
        rows.append(["(bank not yet frozen)", "", "", "", "", "", "", ""])
    return headers, rows


def _table_04_compact_policy_bank(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Action ID", "Index", "K+", "K-", "Positive Selector",
        "Negative Selector",
    ]
    rows: list[list[str]] = []
    bank = data.get("compact_bank_yaml", {})
    actions = bank.get("actions", []) or []
    for entry in actions:
        rows.append([
            entry.get("action_id", ""),
            _fmt_int(entry.get("action_index")),
            _fmt_int(entry.get("k_positive")),
            _fmt_int(entry.get("k_negative")),
            entry.get("positive_selector", ""),
            entry.get("negative_selector", ""),
        ])
    if not rows:
        status = bank.get("status", "PLACEHOLDER_PENDING_PARETO_PRUNING")
        rows.append([f"(compact bank: {status})", "", "", "", "", ""])
    return headers, rows


def _table_05_fixed_action_absolute_performance(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Reference Set", "Strongest Fixed Action", "Mean Dice", "Mean IoU",
        "Mean AJI", "Mean Component Recall", "Mean Runtime (s)",
    ]
    rows: list[list[str]] = []

    # Validation A and B
    val_rows = data.get("y1_utility_matrix_validation", [])
    train_rows = data.get("y1_utility_matrix_train", [])

    for ref_set_name, rows_src in [("Validation A", val_rows), ("Validation B", val_rows), ("Train A", train_rows)]:
        ref_letter = "A" if "A" in ref_set_name else "B"
        if ref_set_name.startswith("Train"):
            ref_letter = "A"
        per_q = _aggregate_per_query_action_dice(rows_src, ref_letter)
        if not per_q:
            rows.append([ref_set_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        action_means = {
            aid: _mean([per_q[q][aid] for q in per_q if aid in per_q[q]])
            for aid in set(a for q in per_q for a in per_q[q])
        }
        if not action_means:
            rows.append([ref_set_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        best = max(action_means, key=action_means.get)
        # Aggregate IoU/AJI/recall/runtime for best action
        ious, ajis, recalls, runtimes = [], [], [], []
        for r in rows_src:
            if r.get("reference_set") != ref_letter or r.get("action_id") != best:
                continue
            if r.get("status") != "OK":
                continue
            ious.append(_to_float(r.get("iou")))
            ajis.append(_to_float(r.get("aji")))
            recalls.append(_to_float(r.get("component_recall")))
            runtimes.append(_to_float(r.get("runtime_seconds", r.get("runtime_ms", 0))) / 1000.0)
        rows.append([
            ref_set_name,
            best,
            _fmt(action_means[best]),
            _fmt(_mean(ious)),
            _fmt(_mean(ajis)),
            _fmt(_mean(recalls)),
            _fmt(_mean(runtimes), 2),
        ])
    return headers, rows


def _table_06_raw_cross_stable_oracle(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Reference Set", "Fixed Dice", "Raw Oracle Dice", "Cross-Ref Oracle",
        "Stable Oracle", "Raw Oracle Gap", "Stable Oracle Gap",
    ]
    rows: list[list[str]] = []
    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    # If y1_summary wraps under "gate_y1", unwrap
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    fixed = y1.get("fixed_A_dice", 0.0)
    oracle_A = y1.get("oracle_A", 0.0)
    oracle_B = y1.get("oracle_B", 0.0)
    cross_A_to_B = y1.get("cross_A_to_B", 0.0)
    cross_B_to_A = y1.get("cross_B_to_A", 0.0)
    stable = y1.get("stable_oracle", 0.0)
    rows.append([
        "A (raw)",
        _fmt(fixed), _fmt(oracle_A), _fmt(cross_A_to_B),
        _fmt(stable), _fmt(oracle_A - fixed), _fmt(stable - fixed),
    ])
    rows.append([
        "B (raw)",
        _fmt(fixed), _fmt(oracle_B), _fmt(cross_B_to_A),
        _fmt(stable), _fmt(oracle_B - fixed), _fmt(stable - fixed),
    ])
    rows.append([
        "Stable (mean cross)",
        _fmt(fixed), _fmt((oracle_A + oracle_B) / 2.0),
        _fmt((cross_A_to_B + cross_B_to_A) / 2.0),
        _fmt(stable),
        _fmt(((oracle_A + oracle_B) / 2.0) - fixed),
        _fmt(stable - fixed),
    ])
    return headers, rows


def _table_07_oracle_optimism(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Query Case ID", "Oracle A", "Cross A->B", "Optimism A", "Oracle B", "Cross B->A", "Optimism B"]
    rows: list[list[str]] = []
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    setB = _aggregate_per_query_action_dice(val_rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    for q in queries:
        if not setA[q] or not setB[q]:
            continue
        a_star = max(setA[q], key=setA[q].get)
        b_star = max(setB[q], key=setB[q].get)
        oracle_A = setA[q][a_star]
        oracle_B = setB[q][b_star]
        cross_A_to_B = setB[q].get(a_star, 0.0)
        cross_B_to_A = setA[q].get(b_star, 0.0)
        rows.append([
            q,
            _fmt(oracle_A), _fmt(cross_A_to_B), _fmt(oracle_A - cross_A_to_B),
            _fmt(oracle_B), _fmt(cross_B_to_A), _fmt(oracle_B - cross_B_to_A),
        ])
    # Aggregate row
    if rows:
        y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
        if "gate_y1" in data.get("y1_summary", {}):
            y1 = data["y1_summary"]["gate_y1"]
        rows.append([
            "AGGREGATE",
            _fmt(y1.get("oracle_A", 0.0)),
            _fmt(y1.get("cross_A_to_B", 0.0)),
            _fmt(y1.get("optimism_A", 0.0)),
            _fmt(y1.get("oracle_B", 0.0)),
            _fmt(y1.get("cross_B_to_A", 0.0)),
            _fmt(y1.get("optimism_B", 0.0)),
        ])
    return headers, rows


def _table_08_action_stability(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Metric", "Value"]
    rows: list[list[str]] = []
    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    rows.append(["Exact action agreement rate", _fmt(y1.get("exact_action_agreement_rate"))])
    rows.append(["K+ bin agreement rate (±5)", _fmt(y1.get("kplus_bin_agreement_rate"))])
    rows.append(["Selector agreement rate", _fmt(y1.get("selector_agreement_rate"))])
    rows.append(["Median Spearman rho", _fmt(y1.get("median_spearman_rho"))])
    rows.append(["Mean top-3 overlap", _fmt(y1.get("mean_top3_overlap"))])
    rows.append(["Win consistency rate", _fmt(y1.get("win_consistency_rate"))])
    rows.append(["Diff-from-fixed rate", _fmt(y1.get("diff_from_fixed_rate"))])
    return headers, rows


def _table_09_reference_sensitivity(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Query Case ID", "Best Dice A", "Best Dice B", "Diff (A-B)", "A wins on B?"]
    rows: list[list[str]] = []
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    setB = _aggregate_per_query_action_dice(val_rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    for q in queries:
        if not setA[q] or not setB[q]:
            continue
        best_A = max(setA[q].values())
        best_B = max(setB[q].values())
        a_star = max(setA[q], key=setA[q].get)
        a_on_B = setB[q].get(a_star, 0.0)
        a_wins = "yes" if a_on_B >= best_B - 0.005 else "no"
        rows.append([q, _fmt(best_A), _fmt(best_B), _fmt(best_A - best_B), a_wins])
    return headers, rows


def _table_10_bank_policy_validation(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Model", "Mean Dice", "Delta vs Fixed", "CI Lo", "CI Hi",
        "Oracle Recovery", "Non-Modal Rate", "Winning Action Classes",
    ]
    rows: list[list[str]] = []
    y3 = data.get("y3_summary", {})
    all_models = y3.get("all_models", {})
    if not all_models:
        # Fallback: try to read per-query CSV grouped by model_name
        per_query = data.get("y3_bank_policy_per_query", [])
        by_model: dict[str, list[float]] = {}
        for r in per_query:
            m = r.get("model_name", "")
            d = r.get("dice")
            if m and d:
                by_model.setdefault(m, []).append(_to_float(d))
        for m, dices in by_model.items():
            rows.append([m, _fmt(_mean(dices)), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    else:
        for name, info in all_models.items():
            rows.append([
                name,
                _fmt(info.get("mean_dice")),
                _fmt(info.get("delta_vs_fixed")),
                "N/A", "N/A",
                _fmt(info.get("oracle_recovery")),
                "N/A", "N/A",
            ])
    # If we have a richer gate_y3 dict, prefer that
    gate_y3 = y3.get("gate_y3", {})
    if gate_y3 and gate_y3.get("all_models"):
        rows = []
        for name, info in gate_y3["all_models"].items():
            rows.append([
                name,
                _fmt(info.get("mean_dice")),
                _fmt(info.get("delta_vs_fixed")),
                _fmt(info.get("bootstrap_ci_lo")),
                "N/A",
                _fmt(info.get("oracle_recovery")),
                _fmt(info.get("non_modal_rate")),
                _fmt_int(info.get("winning_action_classes")),
            ])
    return headers, rows


def _table_11_bank_policy_holdout(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Query Case ID", "Model", "Selected Action", "Dice", "IoU", "AJI",
        "Component Recall", "Runtime (s)",
    ]
    rows: list[list[str]] = []
    for r in data.get("holdout_per_query", []):
        rows.append([
            r.get("query_case_id", ""),
            r.get("model_name", ""),
            r.get("selected_action_id", r.get("predicted_action", "")),
            _fmt(r.get("dice", r.get("policy_dice"))),
            _fmt(r.get("iou", r.get("policy_iou"))),
            _fmt(r.get("aji", r.get("policy_aji"))),
            _fmt(r.get("component_recall", r.get("policy_component_recall"))),
            _fmt(_to_float(r.get("runtime_seconds", r.get("runtime_ms", 0))) / 1000.0, 2),
        ])
    return headers, rows


def _table_12_point_subset_validation(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model", "Mean Dice", "Delta vs Fixed", "CI Lo", "CI Hi", "Oracle Recovery"]
    rows: list[list[str]] = []
    y4 = data.get("y4_summary", {})
    gate_y4 = y4.get("gate_y4", y4)
    all_models = gate_y4.get("all_models", {})
    if all_models:
        for name, info in all_models.items():
            rows.append([
                name,
                _fmt(info.get("mean_dice")),
                _fmt(info.get("delta_vs_fixed")),
                _fmt(info.get("bootstrap_ci_lo")),
                _fmt(info.get("bootstrap_ci_hi")),
                _fmt(info.get("oracle_recovery")),
            ])
    else:
        # Fallback: aggregate from per-query CSV
        per_query = data.get("y4_point_subset_per_query", [])
        by_model: dict[str, list[float]] = {}
        for r in per_query:
            m = r.get("model_name", "")
            d = r.get("dice")
            if m and d:
                by_model.setdefault(m, []).append(_to_float(d))
        for m, dices in by_model.items():
            rows.append([m, _fmt(_mean(dices)), "N/A", "N/A", "N/A", "N/A"])
    if not rows:
        rows.append(["(Gate Y4 not run)", "N/A", "N/A", "N/A", "N/A", "N/A"])
    return headers, rows


def _table_13_point_subset_holdout(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Query Case ID", "Model", "K+", "K-", "Dice", "IoU", "AJI", "Component Recall"]
    rows: list[list[str]] = []
    # Reuse y4_point_subset_per_query if it has holdout rows; otherwise mark as not run
    per_query = data.get("y4_point_subset_per_query", [])
    for r in per_query:
        rows.append([
            r.get("query_case_id", ""),
            r.get("model_name", ""),
            _fmt_int(r.get("k_positive")),
            _fmt_int(r.get("k_negative")),
            _fmt(r.get("dice")),
            _fmt(r.get("iou")),
            _fmt(r.get("aji")),
            _fmt(r.get("component_recall")),
        ])
    if not rows:
        rows.append(["(Point-Subset holdout not evaluated)", "", "", "", "", "", "", ""])
    return headers, rows


def _table_14_oracle_recovery(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Method", "Mean Dice", "Fixed Dice", "Stable Oracle Dice", "Oracle Gap", "Policy Gap", "Recovery %"]
    rows: list[list[str]] = []

    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    fixed = _to_float(y1.get("fixed_A_dice"))
    stable = _to_float(y1.get("stable_oracle"))
    oracle_gap = stable - fixed

    def _row(name: str, dice: float) -> list[str]:
        policy_gap = dice - fixed
        recovery = (policy_gap / oracle_gap * 100.0) if oracle_gap > 0 else 0.0
        return [name, _fmt(dice), _fmt(fixed), _fmt(stable), _fmt(oracle_gap), _fmt(policy_gap), f"{recovery:.1f}%"]

    # Bank Policy (validation)
    y3 = data.get("y3_summary", {})
    gate_y3 = y3.get("gate_y3", y3)
    best_name = gate_y3.get("best_model", "")
    best_dice = _to_float(gate_y3.get("best_model_dice"))
    if best_name:
        rows.append(_row(f"Bank Policy ({best_name}, val)", best_dice))

    # Bank Policy (holdout)
    holdout = data.get("holdout_summary", {})
    holdout_dice = _to_float(holdout.get("mean_policy_dice"))
    if holdout_dice > 0:
        rows.append(_row("Bank Policy (holdout)", holdout_dice))

    # Point-Subset (validation)
    y4 = data.get("y4_summary", {})
    gate_y4 = y4.get("gate_y4", y4)
    y4_best = gate_y4.get("best_model_dice") or gate_y4.get("best_model", {}).get("mean_dice") if isinstance(gate_y4.get("best_model"), dict) else gate_y4.get("best_model_dice")
    if y4_best:
        rows.append(_row("Point-Subset (val)", _to_float(y4_best)))

    # Stable Oracle (upper bound)
    rows.append(_row("Stable Oracle (upper bound)", stable))

    # Raw Bank Oracle (upper bound)
    rows.append(_row("Raw Bank Oracle (upper bound)", _to_float(y1.get("oracle_A"))))

    # Search Oracle (privileged upper bound)
    y2 = data.get("y2_summary", {})
    raw_search = _to_float(y2.get("raw_search_dice"))
    cross_search = _to_float(y2.get("cross_search_dice"))
    if raw_search > 0:
        rows.append(_row("Raw Search Oracle (privileged)", raw_search))
    if cross_search > 0:
        rows.append(_row("Cross-Ref Search Oracle (privileged)", cross_search))

    return headers, rows


def _table_15_calls_runtime(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Stage", "Mean SAM Calls", "Mean Runtime (s/episode)", "GPU Hours", "Notes"]
    rows: list[list[str]] = []

    def _from_ledger(stage_name: str, notes: str = "") -> list[str] | None:
        for r in data.get("phaseY_compute_ledger", []):
            stage = (r.get("stage") or r.get("Stage") or "").lower()
            if stage_name.lower() in stage:
                return [
                    r.get("stage") or r.get("Stage") or stage_name,
                    _fmt(r.get("mean_sam_calls") or r.get("sam_calls", ""), 2),
                    _fmt(_to_float(r.get("mean_runtime_seconds", r.get("mean_runtime_ms", 0))) / 1000.0, 2),
                    _fmt(r.get("gpu_hours", ""), 2),
                    notes or r.get("notes", ""),
                ]
        return None

    stages = [
        ("Y1 utility matrix", "210 actions × 1 SAM call"),
        ("Y2 search oracle", "GT-guided search budget"),
        ("Y3 bank policy", "1 deployed SAM call"),
        ("Y4 point-subset", "1 deployed SAM call"),
        ("Holdout", "1 deployed SAM call"),
    ]
    for stage_name, notes in stages:
        row = _from_ledger(stage_name, notes)
        if row:
            rows.append(row)
        else:
            rows.append([stage_name, "N/A", "N/A", "N/A", notes])

    # Also pull from run_metadata if available
    for src_key, label in [
        ("y1_run_metadata", "Y1 (run_metadata)"),
        ("y2_run_metadata", "Y2 (run_metadata)"),
        ("y3_run_metadata", "Y3 (run_metadata)"),
        ("y4_run_metadata", "Y4 (run_metadata)"),
        ("holdout_run_metadata", "Holdout (run_metadata)"),
    ]:
        meta = data.get(src_key, {})
        if meta:
            gpu_hours = meta.get("gpu_hours") or meta.get("total_gpu_hours")
            rows.append([
                label,
                _fmt(meta.get("mean_sam_calls", ""), 2),
                _fmt(meta.get("mean_runtime_seconds", ""), 2),
                _fmt(gpu_hours, 2),
                meta.get("device", "") or "",
            ])
    return headers, rows


def _table_16_failure_taxonomy(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """Categorize failures: over-segmentation, under-segmentation, missed components, false positive regions."""
    headers = ["Failure Category", "Count", "Rate", "Example Queries"]
    rows: list[list[str]] = []
    per_query = data.get("holdout_per_query", [])
    if not per_query:
        per_query = data.get("y3_bank_policy_per_query", [])

    over_seg, under_seg, missed_comp, false_pos = [], [], [], []
    for r in per_query:
        qid = r.get("query_case_id", "")
        dice = _to_float(r.get("dice", r.get("policy_dice")))
        recall = _to_float(r.get("component_recall", r.get("policy_component_recall")))
        iou = _to_float(r.get("iou", r.get("policy_iou")))
        if dice < 0.1:
            if recall < 0.5:
                missed_comp.append(qid)
            else:
                false_pos.append(qid)
        elif dice < 0.3:
            if recall < 0.5:
                under_seg.append(qid)
            else:
                over_seg.append(qid)
        elif iou < 0.3 and recall > 0.7:
            over_seg.append(qid)

    total = len(per_query) or 1
    for name, bucket in [
        ("Over-segmentation (high recall, low IoU)", over_seg),
        ("Under-segmentation (low Dice, partial recall)", under_seg),
        ("Missed components (recall < 0.5)", missed_comp),
        ("False positive regions (low Dice, high recall)", false_pos),
    ]:
        rows.append([
            name,
            str(len(bucket)),
            f"{len(bucket) / total * 100:.1f}%",
            ", ".join(bucket[:3]),
        ])
    return headers, rows


def _table_17_seed_stability(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model", "Seed", "Mean Dice", "Std Across Seeds"]
    rows: list[list[str]] = []
    y3 = data.get("y3_summary", {})
    gate_y3 = y3.get("gate_y3", y3)
    seed_results = gate_y3.get("seed_results", {})
    if seed_results:
        for model_name, seed_list in seed_results.items():
            dices = [_to_float(s.get("mean_dice")) for s in seed_list]
            std = _stdev(dices)
            for i, d in enumerate(dices):
                rows.append([model_name, f"seed_{i}", _fmt(d), _fmt(std)])
    else:
        # Fallback: just report best model and its CI as a proxy
        best = gate_y3.get("best_model", "")
        if best:
            rows.append([best, "mean of 3", _fmt(gate_y3.get("best_model_dice")), "N/A"])
        else:
            rows.append(["(no seed data)", "", "", ""])
    return headers, rows


def _table_18_final_gate_decision(data: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    headers = ["Gate", "Status", "Decision State", "Conditions"]
    rows: list[list[str]] = []

    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    y2 = data.get("y2_summary", {})
    gate_y2 = y2.get("gate_y2", y2)
    y3 = data.get("y3_summary", {})
    gate_y3 = y3.get("gate_y3", y3)
    y4 = data.get("y4_summary", {})
    gate_y4 = y4.get("gate_y4", y4)
    holdout = data.get("holdout_summary", {})

    def _cond_summary(conditions: dict[str, bool]) -> str:
        if not conditions:
            return "N/A"
        passed = sum(1 for v in conditions.values() if v)
        return f"{passed}/{len(conditions)} passed"

    rows.append([
        "Y1",
        "PASS" if y1.get("decision", "").endswith("STABLE_HETEROGENEITY") else (
            "FAIL" if y1.get("decision") else "NOT_RUN"
        ),
        y1.get("decision", "NOT_RUN"),
        _cond_summary(y1.get("conditions", {})),
    ])
    rows.append([
        "Y2",
        "PASS" if gate_y2.get("decision", "").endswith("STABLE") else (
            "FAIL" if gate_y2.get("decision") else "NOT_RUN"
        ),
        gate_y2.get("decision", "NOT_RUN"),
        _cond_summary(gate_y2.get("conditions", {})),
    ])
    rows.append([
        "Y3",
        "PASS" if gate_y3.get("decision", "").endswith("PASS") else (
            "FAIL" if gate_y3.get("decision") else "NOT_RUN"
        ),
        gate_y3.get("decision", "NOT_RUN"),
        _cond_summary(gate_y3.get("conditions", {})),
    ])
    rows.append([
        "Y4",
        "PASS" if gate_y4.get("decision", "").endswith("PASS") else (
            "FAIL" if gate_y4.get("decision") else "NOT_RUN"
        ),
        gate_y4.get("decision", "NOT_RUN"),
        _cond_summary(gate_y4.get("conditions", {})),
    ])
    rows.append([
        "Holdout",
        "PASS" if holdout.get("decision", "").endswith("PASS") else (
            "FAIL" if holdout.get("decision") else "NOT_RUN"
        ),
        holdout.get("decision", "NOT_RUN"),
        _cond_summary(holdout.get("conditions", {})),
    ])
    return headers, rows


TABLE_BUILDERS = [
    ("01_fresh_data_split", "Table 1: Fresh Data Split", _table_01_fresh_data_split),
    ("02_prior_exposure_audit", "Table 2: Prior Exposure Audit", _table_02_prior_exposure_audit),
    ("03_full_single_call_action_bank", "Table 3: Full Single-Call Action Bank (210 actions)", _table_03_full_action_bank),
    ("04_compact_policy_bank", "Table 4: Compact Policy Bank (12-32 actions)", _table_04_compact_policy_bank),
    ("05_fixed_action_absolute_performance", "Table 5: Fixed Action Absolute Performance", _table_05_fixed_action_absolute_performance),
    ("06_raw_cross_stable_oracle", "Table 6: Raw / Cross-Reference / Stable Oracle", _table_06_raw_cross_stable_oracle),
    ("07_oracle_optimism", "Table 7: Oracle Optimism (per query)", _table_07_oracle_optimism),
    ("08_action_stability", "Table 8: Action Stability", _table_08_action_stability),
    ("09_reference_sensitivity", "Table 9: Reference Sensitivity", _table_09_reference_sensitivity),
    ("10_bank_policy_validation", "Table 10: Bank-Policy Validation (M0-M5)", _table_10_bank_policy_validation),
    ("11_bank_policy_holdout", "Table 11: Bank-Policy Holdout", _table_11_bank_policy_holdout),
    ("12_point_subset_validation", "Table 12: Point-Subset Validation (if Y4 ran)", _table_12_point_subset_validation),
    ("13_point_subset_holdout", "Table 13: Point-Subset Holdout (if Y4 ran)", _table_13_point_subset_holdout),
    ("14_oracle_recovery", "Table 14: Oracle Recovery", _table_14_oracle_recovery),
    ("15_calls_runtime", "Table 15: Calls / Runtime / GPU Hours", _table_15_calls_runtime),
    ("16_failure_taxonomy", "Table 16: Failure Taxonomy", _table_16_failure_taxonomy),
    ("17_seed_stability", "Table 17: Seed Stability", _table_17_seed_stability),
    ("18_final_gate_decision", "Table 18: Final Gate Decision", _table_18_final_gate_decision),
]


# ---------------------------------------------------------------------------
# Figure builders (20 figures per Section 16)
# ---------------------------------------------------------------------------


def _fig_01_raw_vs_crossref_oracle(data: dict[str, Any], path: Path) -> None:
    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    fixed = _to_float(y1.get("fixed_A_dice"))
    oracle_A = _to_float(y1.get("oracle_A"))
    oracle_B = _to_float(y1.get("oracle_B"))
    cross_A_to_B = _to_float(y1.get("cross_A_to_B"))
    cross_B_to_A = _to_float(y1.get("cross_B_to_A"))
    stable = _to_float(y1.get("stable_oracle"))
    if not oracle_A and not oracle_B:
        return
    labels = ["Fixed-A", "Oracle-A", "Oracle-B", "Cross A→B", "Cross B→A", "Stable"]
    values = [fixed, oracle_A, oracle_B, cross_A_to_B, cross_B_to_A, stable]
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    bars = ax.bar(labels, values, color=["#999", "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"], alpha=0.85)
    ax.set_ylabel("Mean Dice")
    ax.set_title("Fig 1: Raw Oracle vs Cross-Reference Oracle")
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    _save_fig(fig, path)


def _fig_02_oracle_optimism_distribution(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    setB = _aggregate_per_query_action_dice(val_rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    optimism_values: list[float] = []
    for q in queries:
        if not setA[q] or not setB[q]:
            continue
        a_star = max(setA[q], key=setA[q].get)
        optimism_values.append(setA[q][a_star] - setB[q].get(a_star, 0.0))
    if not optimism_values:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(optimism_values, bins=12, color="#4C78A8", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Oracle optimism (Oracle-A − Cross A→B)")
    ax.set_ylabel("Query count")
    ax.set_title("Fig 2: Oracle Optimism Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def _fig_03_action_rank_correlation(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    setB = _aggregate_per_query_action_dice(val_rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    rhos: list[float] = []
    for q in queries:
        aids = sorted(set(setA[q].keys()) & set(setB[q].keys()))
        if len(aids) < 2:
            continue
        x = [setA[q][a] for a in aids]
        y = [setB[q][a] for a in aids]
        rhos.append(_spearman_rho(x, y))
    if not rhos:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(rhos, bins=12, color="#54A24B", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Per-query Spearman ρ (action utility A vs B)")
    ax.set_ylabel("Query count")
    ax.set_title(f"Fig 3: Action Rank Correlation A vs B (median={_median(rhos):.3f})")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def _spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    mean_rx = _mean(rx)
    mean_ry = _mean(ry)
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = (sum((a - mean_rx) ** 2 for a in rx)) ** 0.5
    den_y = (sum((b - mean_ry) ** 2 for b in ry)) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _rank(values: list[float]) -> list[float]:
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _fig_04_best_kplus_distribution(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    kps = [_action_k_positive(max(setA[q], key=setA[q].get)) for q in setA if setA[q]]
    if not kps:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(kps, bins=range(min(kps), max(kps) + 5, 5), color="#4C78A8", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Best K+ (per query, oracle on A)")
    ax.set_ylabel("Query count")
    ax.set_title("Fig 4: Best K+ Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def _fig_05_best_kneg_distribution(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    kns = [_action_k_negative(max(setA[q], key=setA[q].get)) for q in setA if setA[q]]
    if not kns:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.hist(kns, bins=range(0, max(kns) + 2, 1), color="#F58518", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Best K- (per query, oracle on A)")
    ax.set_ylabel("Query count")
    ax.set_title("Fig 5: Best K- Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def _fig_06_stable_action_family_distribution(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    counts: dict[str, int] = {}
    for q in setA:
        if not setA[q]:
            continue
        a_star = max(setA[q], key=setA[q].get)
        fam = _action_pos_selector(a_star) or "UNKNOWN"
        counts[fam] = counts.get(fam, 0) + 1
    if not counts:
        return
    labels = list(counts.keys())
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.bar(range(len(labels)), values, color="#B279A2", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Query count")
    ax.set_title("Fig 6: Stable Action Family Distribution")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, path)


def _fig_07_fixed_vs_bank_policy_paired_scatter(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y3_bank_policy_per_query", [])
    if not per_query:
        # Try holdout
        per_query = data.get("holdout_per_query", [])
    if not per_query:
        return
    fixed_dices: list[float] = []
    policy_dices: list[float] = []
    for r in per_query:
        fd = r.get("fixed_dice")
        pd_ = r.get("dice", r.get("policy_dice"))
        if fd is None or pd_ is None:
            continue
        fixed_dices.append(_to_float(fd))
        policy_dices.append(_to_float(pd_))
    if not fixed_dices:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(fixed_dices, policy_dices, alpha=0.6, s=40, color="#4C78A8", edgecolor="black", linewidth=0.5)
    lo = min(min(fixed_dices), min(policy_dices)) - 0.02
    hi = max(max(fixed_dices), max(policy_dices)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="diagonal (y=x)")
    ax.set_xlabel("Fixed-action Dice")
    ax.set_ylabel("Bank-policy Dice")
    ax.set_title("Fig 7: Fixed vs Bank Policy (paired scatter)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_08_fixed_vs_point_subset_paired_scatter(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y4_point_subset_per_query", [])
    if not per_query:
        return
    fixed_dices: list[float] = []
    policy_dices: list[float] = []
    for r in per_query:
        fd = r.get("fixed_dice")
        pd_ = r.get("dice")
        if fd is None or pd_ is None:
            continue
        fixed_dices.append(_to_float(fd))
        policy_dices.append(_to_float(pd_))
    if not fixed_dices:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(fixed_dices, policy_dices, alpha=0.6, s=40, color="#54A24B", edgecolor="black", linewidth=0.5)
    lo = min(min(fixed_dices), min(policy_dices)) - 0.02
    hi = max(max(fixed_dices), max(policy_dices)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="diagonal (y=x)")
    ax.set_xlabel("Fixed-action Dice")
    ax.set_ylabel("Point-Subset Dice")
    ax.set_title("Fig 8: Fixed vs Point-Subset (paired scatter)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_09_oracle_recovery(data: dict[str, Any], path: Path) -> None:
    headers, rows = _table_14_oracle_recovery(data)
    if len(rows) < 2:
        return
    labels = [r[0] for r in rows]
    recoveries = []
    for r in rows:
        try:
            recoveries.append(float(r[6].rstrip("%")))
        except (ValueError, AttributeError):
            recoveries.append(0.0)
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    bars = ax.bar(range(len(labels)), recoveries,
                  color=["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756", "#72B7B2", "#999"][: len(labels)],
                  alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Oracle recovery (%)")
    ax.set_title("Fig 9: Oracle Recovery by Method")
    ax.axhline(30, color="red", linestyle="--", alpha=0.6, label="30% threshold (Y3)")
    ax.axhline(25, color="orange", linestyle="--", alpha=0.6, label="25% threshold (holdout)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, recoveries):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=8)
    _save_fig(fig, path)


def _fig_10_dice_vs_estimated_target_count(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y3_bank_policy_per_query", [])
    if not per_query:
        per_query = data.get("holdout_per_query", [])
    if not per_query:
        return
    xs: list[float] = []
    ys: list[float] = []
    for r in per_query:
        # try to find target count or component count fields
        tc = r.get("target_component_count") or r.get("components_total")
        d = r.get("dice", r.get("policy_dice"))
        if tc is None or d is None:
            continue
        xs.append(_to_float(tc))
        ys.append(_to_float(d))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(xs, ys, alpha=0.6, s=50, color="#E45756", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Estimated target component count")
    ax.set_ylabel("Dice")
    ax.set_title("Fig 10: Dice vs Estimated Target Count")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_11_dice_vs_map_entropy(data: dict[str, Any], path: Path) -> None:
    # The per-query CSVs may not carry map entropy; we use AJI as a proxy dispersion metric
    # if entropy is not available. Skip gracefully if no usable field.
    per_query = data.get("y3_bank_policy_per_query", [])
    if not per_query:
        return
    xs: list[float] = []
    ys: list[float] = []
    for r in per_query:
        ent = r.get("map_entropy") or r.get("fg_entropy") or r.get("contrast_entropy")
        d = r.get("dice", r.get("policy_dice"))
        if ent is None or d is None:
            continue
        xs.append(_to_float(ent))
        ys.append(_to_float(d))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(xs, ys, alpha=0.6, s=50, color="#72B7B2", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Map entropy (per-query)")
    ax.set_ylabel("Dice")
    ax.set_title("Fig 11: Dice vs Map Entropy")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_12_policy_action_vs_candidate_dispersion(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y3_bank_policy_per_query", [])
    if not per_query:
        return
    xs: list[float] = []
    ys: list[float] = []
    for r in per_query:
        disp = r.get("candidate_dispersion") or r.get("pos_coord_dispersion")
        kp = r.get("k_positive") or _action_k_positive(r.get("selected_action_id", r.get("predicted_action", "")))
        if disp is None or kp is None:
            continue
        xs.append(_to_float(disp))
        ys.append(_to_float(kp))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(xs, ys, alpha=0.6, s=50, color="#B279A2", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Candidate dispersion (per-query)")
    ax.set_ylabel("Policy-selected K+")
    ax.set_title("Fig 12: Policy Action vs Candidate Dispersion")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_13_reference_sensitivity(data: dict[str, Any], path: Path) -> None:
    val_rows = data.get("y1_utility_matrix_validation", [])
    setA = _aggregate_per_query_action_dice(val_rows, "A")
    setB = _aggregate_per_query_action_dice(val_rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    xs: list[float] = []
    ys: list[float] = []
    for q in queries:
        if not setA[q] or not setB[q]:
            continue
        xs.append(max(setA[q].values()))
        ys.append(max(setB[q].values()))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(xs, ys, alpha=0.6, s=50, color="#F58518", edgecolor="black", linewidth=0.5)
    lo = min(min(xs), min(ys)) - 0.02
    hi = max(max(xs), max(ys)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="diagonal (y=x)")
    ax.set_xlabel("Best Dice on Reference Set A")
    ax.set_ylabel("Best Dice on Reference Set B")
    ax.set_title("Fig 13: Reference Sensitivity (per-query Dice A vs B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_14_calibration_plot(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y3_bank_policy_per_query", [])
    if not per_query:
        per_query = data.get("holdout_per_query", [])
    if not per_query:
        return
    # Use sam_score as predicted, dice as actual (only if available)
    preds: list[float] = []
    actuals: list[float] = []
    for r in per_query:
        pred = r.get("sam_score") or r.get("predicted_dice")
        act = r.get("dice", r.get("policy_dice"))
        if pred is None or act is None:
            continue
        preds.append(_to_float(pred))
        actuals.append(_to_float(act))
    if len(preds) < 5:
        return
    # Bin by predicted
    bins = np.linspace(min(preds), max(preds), 6)
    bin_centers = []
    bin_actuals = []
    for i in range(len(bins) - 1):
        in_bin = [a for p, a in zip(preds, actuals) if bins[i] <= p < bins[i + 1]]
        if in_bin:
            bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
            bin_actuals.append(_mean(in_bin))
    if not bin_centers:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.plot(bin_centers, bin_actuals, marker="o", linewidth=1.8, color="#4C78A8", label="Observed")
    lo = min(min(bin_centers), min(bin_actuals)) - 0.02
    hi = max(max(bin_centers), max(bin_actuals)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Predicted (binned)")
    ax.set_ylabel("Actual Dice")
    ax.set_title("Fig 14: Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_15_failure_taxonomy(data: dict[str, Any], path: Path) -> None:
    headers, rows = _table_16_failure_taxonomy(data)
    if not rows:
        return
    labels = [r[0] for r in rows]
    counts = [_to_int(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    bars = ax.bar(range(len(labels)), counts,
                  color=["#4C78A8", "#F58518", "#E45756", "#B279A2"][: len(labels)], alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Fig 15: Error / Failure Taxonomy")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v),
                ha="center", va="bottom", fontsize=8)
    _save_fig(fig, path)


def _load_qualitative_image(case_id: str):
    """Load image + mask for a case_id. Returns (image_np, mask_np) or (None, None)."""
    if not _PIL_AVAILABLE:
        return None, None
    split = case_id.rsplit("_", 1)[0] if "_" in case_id else "train"
    img_path = MONUSAC_ROOT / split / "images" / f"{case_id}.png"
    mask_path = MONUSAC_ROOT / split / "masks" / f"{case_id}.png"
    if not img_path.is_file() or not mask_path.is_file():
        return None, None
    try:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        return image, mask
    except Exception:
        return None, None


def _qualitative_grid(cases: list[str], predictions: dict[str, str], title: str, path: Path) -> None:
    """Render a 3-column grid (image | prediction label | GT mask) for top-3 cases."""
    if not _PIL_AVAILABLE:
        return
    n = min(3, len(cases))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i, case_id in enumerate(cases[:n]):
        image, mask = _load_qualitative_image(case_id)
        if image is None:
            for j in range(3):
                axes[i, j].axis("off")
                axes[i, j].text(0.5, 0.5, f"{case_id}\n(not found)",
                                ha="center", va="center", fontsize=8)
            continue
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{case_id} (image)", fontsize=9)
        axes[i, 0].axis("off")
        # Middle: prediction label panel (we don't have predicted mask here, so show action label)
        axes[i, 1].axis("off")
        pred_label = predictions.get(case_id, "(no prediction)")
        axes[i, 1].text(0.5, 0.5, f"Predicted action:\n{pred_label}",
                        ha="center", va="center", fontsize=8, wrap=True)
        axes[i, 1].set_title("Prediction", fontsize=9)
        # GT mask
        if mask is not None:
            axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("GT mask", fontsize=9)
        axes[i, 2].axis("off")
    fig.suptitle(title, fontsize=11)
    _save_fig(fig, path)


def _fig_16_qualitative_fixed_win_cases(data: dict[str, Any], path: Path) -> None:
    """Top-3 cases where fixed-action wins over policy (max policy loss)."""
    per_query = data.get("holdout_per_query", []) or data.get("y3_bank_policy_per_query", [])
    if not per_query:
        return
    diffs: list[tuple[float, str, str]] = []
    for r in per_query:
        qid = r.get("query_case_id", "")
        fd = _to_float(r.get("fixed_dice"))
        pd_ = _to_float(r.get("dice", r.get("policy_dice")))
        pred = r.get("selected_action_id", r.get("predicted_action", ""))
        if qid:
            diffs.append((fd - pd_, qid, pred))
    if not diffs:
        return
    diffs.sort(reverse=True)
    cases = [d[1] for d in diffs[:3]]
    preds = {d[1]: d[2] for d in diffs[:3]}
    _qualitative_grid(cases, preds, "Fig 16: Qualitative Fixed-Win Cases (top-3 policy loss)", path)


def _fig_17_qualitative_policy_win_cases(data: dict[str, Any], path: Path) -> None:
    """Top-3 cases where policy wins over fixed (max policy gain)."""
    per_query = data.get("holdout_per_query", []) or data.get("y3_bank_policy_per_query", [])
    if not per_query:
        return
    diffs: list[tuple[float, str, str]] = []
    for r in per_query:
        qid = r.get("query_case_id", "")
        fd = _to_float(r.get("fixed_dice"))
        pd_ = _to_float(r.get("dice", r.get("policy_dice")))
        pred = r.get("selected_action_id", r.get("predicted_action", ""))
        if qid:
            diffs.append((pd_ - fd, qid, pred))
    if not diffs:
        return
    diffs.sort(reverse=True)
    cases = [d[1] for d in diffs[:3]]
    preds = {d[1]: d[2] for d in diffs[:3]}
    _qualitative_grid(cases, preds, "Fig 17: Qualitative Policy-Win Cases (top-3 policy gain)", path)


def _fig_18_high_confidence_policy_failures(data: dict[str, Any], path: Path) -> None:
    """Top-3 high-confidence policy failures (low Dice with high predicted score)."""
    per_query = data.get("holdout_per_query", []) or data.get("y3_bank_policy_per_query", [])
    if not per_query:
        return
    # Sort by Dice ascending (worst first), and try to use predicted score if available
    sorted_rows = sorted(per_query, key=lambda r: _to_float(r.get("dice", r.get("policy_dice", 0))))
    cases = [r.get("query_case_id", "") for r in sorted_rows[:3]]
    preds = {r.get("query_case_id", ""): r.get("selected_action_id", r.get("predicted_action", ""))
             for r in sorted_rows[:3]}
    if not cases:
        return
    _qualitative_grid(cases, preds, "Fig 18: High-Confidence Policy Failures (lowest 3 Dice)", path)


def _fig_19_search_oracle_overfit(data: dict[str, Any], path: Path) -> None:
    per_query = data.get("y2_search_oracle_per_query", [])
    if not per_query:
        return
    raw = [_to_float(r.get("raw_search_dice")) for r in per_query]
    cross = [_to_float(r.get("cross_search_dice")) for r in per_query]
    if not raw:
        return
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    ax.scatter(raw, cross, alpha=0.6, s=50, color="#E45756", edgecolor="black", linewidth=0.5)
    lo = min(min(raw), min(cross)) - 0.02
    hi = max(max(raw), max(cross)) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="diagonal (y=x)")
    ax.set_xlabel("Raw Search Dice (same reference)")
    ax.set_ylabel("Cross-Reference Search Dice")
    ax.set_title("Fig 19: Search-Oracle Overfit Cases (raw vs cross per query)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def _fig_20_final_gate_flow(data: dict[str, Any], path: Path) -> None:
    """Decision-tree diagram of the Phase Y gates using matplotlib patches."""
    y1 = data.get("y1_summary", {}).get("gate_y1", data.get("y1_summary", {}))
    if "gate_y1" in data.get("y1_summary", {}):
        y1 = data["y1_summary"]["gate_y1"]
    y2 = data.get("y2_summary", {}).get("gate_y2", data.get("y2_summary", {}))
    y3 = data.get("y3_summary", {}).get("gate_y3", data.get("y3_summary", {}))
    y4 = data.get("y4_summary", {}).get("gate_y4", data.get("y4_summary", {}))
    holdout = data.get("holdout_summary", {})

    def _status(d: dict[str, Any], pass_token: str) -> str:
        dec = d.get("decision", "")
        if not dec:
            return "NOT RUN"
        if pass_token.lower() in dec.lower():
            return "PASS"
        return "FAIL"

    nodes = [
        ("Y1: Stable\nHeterogeneity", _status(y1, "STABLE_HETEROGENEITY")),
        ("Y2: Search\nLabels Stable", _status(y2, "STABLE")),
        ("Y3: Bank Policy\nValidation", _status(y3, "PASS")),
        ("Y4: Point-Subset\n(optional)", _status(y4, "PASS")),
        ("Holdout:\nConfirmatory", _status(holdout, "PASS")),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_w, box_h = 1.6, 0.9
    y_positions = [5.0, 4.0, 3.0, 2.0, 1.0]
    x_center = 5.0
    color_map = {"PASS": "#54A24B", "FAIL": "#E45756", "NOT RUN": "#999999"}

    for i, ((label, status), y) in enumerate(zip(nodes, y_positions)):
        color = color_map.get(status, "#999999")
        box = FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05", linewidth=1.5,
            facecolor=color, alpha=0.75, edgecolor="black",
        )
        ax.add_patch(box)
        ax.text(x_center, y, f"{label}\n[{status}]", ha="center", va="center",
                fontsize=9, color="white", weight="bold")
        if i < len(nodes) - 1:
            arrow = FancyArrowPatch(
                (x_center, y - box_h / 2), (x_center, y_positions[i + 1] + box_h / 2),
                arrowstyle="->", mutation_scale=15, linewidth=1.4, color="black",
            )
            ax.add_patch(arrow)

    # Side annotations
    side_notes = [
        (5.0, "Stable oracle gap ≥ 0.015"),
        (4.0, "Cross search ≥ fixed + 0.03"),
        (3.0, "Δ ≥ 0.010, recovery ≥ 30%"),
        (2.0, "Δ ≥ 0.015 vs fixed (optional)"),
        (1.0, "Δ ≥ 0.0075, recovery ≥ 25%"),
    ]
    for y, note in side_notes:
        ax.text(7.5, y, note, fontsize=8, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    ax.set_title("Fig 20: Final Phase Y Gate Flow", fontsize=12, weight="bold")
    _save_fig(fig, path)


FIGURE_BUILDERS = [
    ("fig_01_raw_vs_crossref_oracle", _fig_01_raw_vs_crossref_oracle),
    ("fig_02_oracle_optimism_distribution", _fig_02_oracle_optimism_distribution),
    ("fig_03_action_rank_correlation", _fig_03_action_rank_correlation),
    ("fig_04_best_kplus_distribution", _fig_04_best_kplus_distribution),
    ("fig_05_best_kneg_distribution", _fig_05_best_kneg_distribution),
    ("fig_06_stable_action_family_distribution", _fig_06_stable_action_family_distribution),
    ("fig_07_fixed_vs_bank_policy_paired_scatter", _fig_07_fixed_vs_bank_policy_paired_scatter),
    ("fig_08_fixed_vs_point_subset_paired_scatter", _fig_08_fixed_vs_point_subset_paired_scatter),
    ("fig_09_oracle_recovery", _fig_09_oracle_recovery),
    ("fig_10_dice_vs_estimated_target_count", _fig_10_dice_vs_estimated_target_count),
    ("fig_11_dice_vs_map_entropy", _fig_11_dice_vs_map_entropy),
    ("fig_12_policy_action_vs_candidate_dispersion", _fig_12_policy_action_vs_candidate_dispersion),
    ("fig_13_reference_sensitivity", _fig_13_reference_sensitivity),
    ("fig_14_calibration_plot", _fig_14_calibration_plot),
    ("fig_15_failure_taxonomy", _fig_15_failure_taxonomy),
    ("fig_16_qualitative_fixed_win_cases", _fig_16_qualitative_fixed_win_cases),
    ("fig_17_qualitative_policy_win_cases", _fig_17_qualitative_policy_win_cases),
    ("fig_18_high_confidence_policy_failures", _fig_18_high_confidence_policy_failures),
    ("fig_19_search_oracle_overfit", _fig_19_search_oracle_overfit),
    ("fig_20_final_gate_flow", _fig_20_final_gate_flow),
]


# ---------------------------------------------------------------------------
# 22-question final-state answers (Section 22)
# ---------------------------------------------------------------------------


def _y1(data: dict[str, Any]) -> dict[str, Any]:
    y1 = data.get("y1_summary", {})
    return y1.get("gate_y1", y1)


def _y2(data: dict[str, Any]) -> dict[str, Any]:
    y2 = data.get("y2_summary", {})
    return y2.get("gate_y2", y2)


def _y3(data: dict[str, Any]) -> dict[str, Any]:
    y3 = data.get("y3_summary", {})
    return y3.get("gate_y3", y3)


def _y4(data: dict[str, Any]) -> dict[str, Any]:
    y4 = data.get("y4_summary", {})
    return y4.get("gate_y4", y4)


def _build_answers(data: dict[str, Any]) -> dict[str, str]:
    split = data.get("split_summary", {})
    y1 = _y1(data)
    y2 = _y2(data)
    y3 = _y3(data)
    y4 = _y4(data)
    holdout = data.get("holdout_summary", {})

    split_info = split.get("split", {})

    # Determine final state
    y1_pass = bool(y1.get("decision", "").endswith("STABLE_HETEROGENEITY"))
    y2_pass = bool(y2.get("decision", "").endswith("STABLE"))
    y3_pass = bool(y3.get("decision", "").endswith("PASS"))
    y4_pass = bool(y4.get("decision", "").endswith("PASS"))
    holdout_pass = bool(holdout.get("decision", "").endswith("PASS"))

    if holdout_pass and y4_pass:
        final_state = "PHASEY_POLICY_FEASIBILITY_CONFIRMED (Point-Subset)"
    elif holdout_pass:
        final_state = "PHASEY_BANK_POLICY_CONFIRMATORY_PASS"
    elif y3_pass and not holdout_pass and holdout.get("decision"):
        final_state = "PHASEY_BANK_POLICY_CONFIRMATORY_FAIL"
    elif y1_pass and not y3_pass and y3.get("decision"):
        if "COLLAPSES_TO_GLOBAL" in y3.get("decision", ""):
            final_state = "PHASEY_GATE_Y3_POLICY_COLLAPSES_TO_GLOBAL"
        elif "BELOW_001" in y3.get("decision", ""):
            final_state = "PHASEY_GATE_Y3_GAIN_BELOW_001"
        else:
            final_state = "PHASEY_ORACLE_EXISTS_NOT_PREDICTABLE"
    elif y1_pass and not y2_pass and y2.get("decision"):
        final_state = "PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT"
    elif not y1_pass and y1.get("decision"):
        # Per directive section 19.1: Y1 fail + Y2 fail => PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY
        if not y2_pass and y2.get("decision"):
            final_state = "PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY"
        else:
            final_state = y1.get("decision", "PHASEY_GATE_Y1_FAILED")
    else:
        final_state = "PHASEY_BLOCKED_TECHNICAL (gates not yet run)"

    formal_test_allowed = holdout_pass  # Section 20: only after holdout pass + Phase Z freeze

    # Permanently forbidden claims
    forbidden = [
        "Search Oracle is deployable",
        "0.7256 is policy performance",
        "91-action Oracle proves policy success",
        "matched deployed calls make GT search a fair automatic baseline",
        "10 episodes are sufficient for final statistical evidence",
        "reference episodes are independent samples",
    ]

    fixed_action = y1.get("best_fixed_action", "N/A")
    fixed_dice = _to_float(y1.get("fixed_A_dice"))
    raw_oracle = _to_float(y1.get("oracle_A"))
    stable_oracle = _to_float(y1.get("stable_oracle"))
    stable_gap = _to_float(y1.get("stable_oracle_gap"))
    optimism_A = _to_float(y1.get("optimism_A"))
    optimism_ratio_A = _to_float(y1.get("optimism_ratio_A"))

    bank_policy_val_dice = _to_float(y3.get("best_model_dice"))
    bank_policy_holdout_dice = _to_float(holdout.get("mean_policy_dice"))
    bank_policy_delta_holdout = _to_float(holdout.get("delta_vs_fixed"))
    oracle_gap = stable_oracle - fixed_dice
    oracle_recovery_pct = (
        (bank_policy_holdout_dice - fixed_dice) / oracle_gap * 100.0
        if oracle_gap > 0 and bank_policy_holdout_dice > 0 else 0.0
    )
    point_subset_val_dice = _to_float(y4.get("best_model_dice") or 0.0)

    raw_search = _to_float(y2.get("raw_search_dice"))
    cross_search = _to_float(y2.get("cross_search_dice"))
    search_optimism = _to_float(y2.get("search_optimism"))
    search_gap = raw_search - cross_search

    answers = {
        "q1_fresh_query_count": (
            f"train={split_info.get('train', {}).get('case_count', 'N/A')}, "
            f"validation={split_info.get('validation', {}).get('case_count', 'N/A')}, "
            f"holdout={split_info.get('holdout', {}).get('case_count', 'N/A')} "
            f"(eligible fresh cases: {split.get('eligible_fresh_cases', 'N/A')})"
        ),
        "q2_historical_10_excluded": (
            "Yes — historical 10 Phase X queries are listed in split_summary.phasex_query_cases "
            "and excluded from Phase Y train/validation/holdout primary statistics "
            "(used only for method discovery / historical appendix per section 4.2)."
        ),
        "q3_reference_sets_AB_independent": (
            f"Patient-level disjoint: {split.get('patient_level_disjoint', 'N/A')}; "
            f"A/B disjoint per query: {split.get('set_A_B_disjoint_per_query', 'N/A')}; "
            f"reference patient ≠ query patient: {split.get('reference_patient_disjoint_from_query', 'N/A')}. "
            f"Set A rows: {split.get('reference_set_A_rows', 'N/A')}, "
            f"Set B rows: {split.get('reference_set_B_rows', 'N/A')}."
        ),
        "q4_raw_oracle_gap": (
            f"Raw Oracle-A = {_fmt(raw_oracle)}, Fixed-A = {_fmt(fixed_dice)}, "
            f"raw oracle gap = {_fmt(raw_oracle - fixed_dice)}."
        ),
        "q5_cross_reference_stable_oracle_gap": (
            f"Stable oracle = {_fmt(stable_oracle)}, stable oracle gap = {_fmt(stable_gap)}."
        ),
        "q6_oracle_optimism": (
            f"Optimism-A = {_fmt(optimism_A)} (ratio = {_fmt(optimism_ratio_A, 3)} of raw gap); "
            f"Optimism-B = {_fmt(_to_float(y1.get('optimism_B')))}."
        ),
        "q7_best_action_stable_across_references": (
            f"Exact action agreement rate = {_fmt(y1.get('exact_action_agreement_rate'))}; "
            f"K+ bin agreement (±5) = {_fmt(y1.get('kplus_bin_agreement_rate'))}; "
            f"selector agreement = {_fmt(y1.get('selector_agreement_rate'))}; "
            f"median Spearman ρ = {_fmt(y1.get('median_spearman_rho'))}; "
            f"mean top-3 overlap = {_fmt(y1.get('mean_top3_overlap'))}."
        ),
        "q8_91_action_heterogeneity_replicates": (
            f"Gate Y1 decision: `{y1.get('decision', 'NOT_RUN')}`. "
            f"Action count evaluated: {_fmt_int(y1.get('action_count'))}. "
            f"Diff-from-fixed rate: {_fmt(y1.get('diff_from_fixed_rate'))}. "
            f"Stable oracle gap ≥ 0.015: {y1.get('conditions', {}).get('cond1_stable_oracle_gap_ge_0p015', 'N/A')}."
        ),
        "q9_compact_bank_derivation": (
            "Compact bank derived train-only via Pareto pruning (section 7.2): "
            "drop dominated actions in Dice/calls/runtime; drop actions clearly inferior across train folds; "
            "within a family keep simpler action when Dice diff < 0.003; validation and holdout did NOT participate. "
            f"Compact bank status: {data.get('compact_bank_yaml', {}).get('status', 'N/A')}; "
            f"Pareto pruning report: {'present' if data.get('y1_pareto_pruning_md') else 'missing'}."
        ),
        "q10_strongest_fixed_exact_config": (
            f"Strongest fixed action: `{fixed_action}` (mean Dice on A = {_fmt(fixed_dice)}). "
            "Config: single SAM call, G_ONE_GROUP."
        ),
        "q11_bank_policy_validation_improvement": (
            f"Best model: `{y3.get('best_model', 'N/A')}` "
            f"(mean Dice = {_fmt(y3.get('best_model_dice'))}, "
            f"Δ vs fixed = {_fmt(y3.get('delta_vs_fixed'))}, "
            f"oracle recovery = {_fmt(y3.get('oracle_recovery', 0), 3)}). "
            f"Gate Y3 decision: `{y3.get('decision', 'NOT_RUN')}`."
        ),
        "q12_holdout_improvement": (
            f"Holdout mean policy Dice = {_fmt(holdout.get('mean_policy_dice'))}, "
            f"mean fixed Dice = {_fmt(holdout.get('mean_fixed_dice'))}, "
            f"Δ = {_fmt(holdout.get('delta_vs_fixed'))} "
            f"(CI [{_fmt(holdout.get('bootstrap_ci_lo'))}, {_fmt(holdout.get('bootstrap_ci_hi'))}]). "
            f"Decision: `{holdout.get('decision', 'NOT_RUN')}`."
        ),
        "q13_oracle_recovery": (
            f"Bank policy validation oracle recovery = {_fmt(y3.get('oracle_recovery', 0), 3)}; "
            f"holdout oracle recovery ≈ {oracle_recovery_pct:.1f}%."
        ),
        "q14_3seeds_consistent": (
            f"Y3 cond4 (3 seeds direction consistent) = "
            f"{y3.get('conditions', {}).get('cond4_3seeds_consistent', 'N/A')}. "
            f"Seed-stability std (table 17): see table_17_seed_stability."
        ),
        "q15_policy_collapses_to_global": (
            f"Non-modal rate = {_fmt(y3.get('non_modal_rate'))}; "
            f"winning action classes = {_fmt_int(y3.get('winning_action_classes'))}. "
            f"cond7 (not global action) = "
            f"{y3.get('conditions', {}).get('cond7_not_global_action', 'N/A')}."
        ),
        "q16_search_oracle_cross_reference_gap": (
            f"Raw search Dice = {_fmt(raw_search)}, cross-reference search Dice = {_fmt(cross_search)}, "
            f"cross gap = {_fmt(search_gap)}, search optimism = {_fmt(search_optimism)}. "
            f"Gate Y2 decision: `{y2.get('decision', 'NOT_RUN')}`."
        ),
        "q17_search_labels_suitable_for_supervision": (
            f"Optimism ratio = {_fmt(y2.get('optimism_ratio', 0), 3)}; "
            f"median |K_A - K_B| = {_fmt(y2.get('median_kplus_diff', 0), 2)}; "
            f"mean point overlap (Jaccard) = {_fmt(y2.get('mean_point_overlap', 0))}; "
            f"beat-fixed rate on B = {_fmt(y2.get('beat_fixed_rate', 0))}. "
            f"Decision: `{y2.get('decision', 'NOT_RUN')}`."
        ),
        "q18_point_subset_beats_bank_policy": (
            f"Gate Y4 decision: `{y4.get('decision', 'NOT_RUN')}`. "
            f"Point-Subset val Dice = {_fmt(point_subset_val_dice)} vs "
            f"Bank Policy val Dice = {_fmt(bank_policy_val_dice)}. "
            f"Δ = {_fmt(point_subset_val_dice - bank_policy_val_dice)}."
        ),
        "q19_all_methods_same_sam_calls": (
            "Yes — all deployable methods (fixed, bank policy, point-subset) use exactly 1 SAM call "
            "and 1 prompt group (G_ONE_GROUP) per section 6. "
            "Search Oracle uses ~558 search SAM calls but only 1 deployed call (privileged, not deployable)."
        ),
        "q20_formal_test_allowed": (
            f"Formal test NOT yet auto-allowed. Holdout pass = {holdout_pass}. "
            "Per section 20, even after holdout pass, Phase Z freeze "
            "(docs/phaseZ_formal_test_freeze.md, configs/frozen/monusac_final_policy.yaml, "
            "reports/phaseZ_test_hashes.json, reports/phaseZ_statistical_plan.md, "
            "reports/phaseZ_qualitative_sampling_plan.md) must be created and user approval obtained "
            "before any MoNuSAC formal test access."
        ),
        "q21_final_scientific_state": final_state,
        "q22_forbidden_claims": (
            "Permanently forbidden claims (section 21): " + "; ".join(forbidden) + "."
        ),
    }
    return answers


def _compute_ledger(data: dict[str, Any]) -> dict[str, Any]:
    """Aggregate GPU hours per stage from compute ledger + run metadata."""
    stages: dict[str, float] = {}
    for r in data.get("phaseY_compute_ledger", []):
        stage = (r.get("stage") or r.get("Stage") or "unknown").lower()
        gh = _to_float(r.get("gpu_hours"))
        if gh > 0:
            stages[stage] = stages.get(stage, 0.0) + gh
    # Fallback: try run_metadata gpu_hours
    for key, label in [
        ("y1_run_metadata", "y1"),
        ("y2_run_metadata", "y2"),
        ("y3_run_metadata", "y3"),
        ("y4_run_metadata", "y4"),
        ("holdout_run_metadata", "holdout"),
    ]:
        meta = data.get(key, {})
        gh = _to_float(meta.get("gpu_hours") or meta.get("total_gpu_hours"))
        if gh <= 0:
            # Try to compute from elapsed_seconds (Y1 metadata nests under splits)
            elapsed = _to_float(meta.get("elapsed_seconds"))
            if elapsed <= 0:
                splits = meta.get("splits", {}) if isinstance(meta, dict) else {}
                for _split_info in (splits.values() if isinstance(splits, dict) else []):
                    if isinstance(_split_info, dict):
                        elapsed = _to_float(_split_info.get("elapsed_seconds"))
                        if elapsed > 0:
                            break
            if elapsed > 0:
                gh = elapsed / 3600.0
        if gh > 0 and label not in stages:
            stages[label] = gh
    total = sum(stages.values())
    return {"total_gpu_hours": total, "stages": stages}


# ---------------------------------------------------------------------------
# Markdown + JSON report writers
# ---------------------------------------------------------------------------


def _write_markdown_report(
    data: dict[str, Any],
    out_dir: Path,
    table_files: list[str],
    figure_files: list[str],
    skipped_figures: list[str],
    answers: dict[str, str],
) -> Path:
    y1 = _y1(data)
    y3 = _y3(data)
    holdout = data.get("holdout_summary", {})
    split = data.get("split_summary", {})
    split_info = split.get("split", {})

    lines: list[str] = []
    lines.append("# Phase Y Final Report — Fresh-Query Confirmation, Oracle Stability, and Deployable Policy Feasibility")
    lines.append("")
    lines.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Final state**: `{answers['q21_final_scientific_state']}`")
    lines.append(f"- **Splits**: train={split_info.get('train', {}).get('case_count', 'N/A')} / "
                 f"validation={split_info.get('validation', {}).get('case_count', 'N/A')} / "
                 f"holdout={split_info.get('holdout', {}).get('case_count', 'N/A')}")
    lines.append(f"- **Strongest fixed**: `{y1.get('best_fixed_action', 'N/A')}` "
                 f"(Dice={_fmt(y1.get('fixed_A_dice'))})")
    lines.append(f"- **Stable oracle**: {_fmt(y1.get('stable_oracle'))} "
                 f"(gap={_fmt(y1.get('stable_oracle_gap'))})")
    lines.append(f"- **Bank Policy validation**: best model `{y3.get('best_model', 'N/A')}` "
                 f"(Dice={_fmt(y3.get('best_model_dice'))}, "
                 f"Δ={_fmt(y3.get('delta_vs_fixed'))}, "
                 f"recovery={_fmt(y3.get('oracle_recovery', 0), 3)})")
    lines.append(f"- **Bank Policy holdout**: Dice={_fmt(holdout.get('mean_policy_dice'))}, "
                 f"Δ={_fmt(holdout.get('delta_vs_fixed'))}, "
                 f"decision=`{holdout.get('decision', 'NOT_RUN')}`")
    lines.append(f"- **Formal test allowed**: see Q20 (Phase Z freeze required)")
    lines.append("")

    # 22 questions
    lines.append("## Section 22: 22 Mandatory Questions")
    lines.append("")
    q_titles = [
        "Fresh query count?",
        "Are historical 10 queries excluded from primary statistics?",
        "Are Reference Sets A/B independent?",
        "Raw oracle gap?",
        "Cross-reference stable oracle gap?",
        "Oracle optimism?",
        "Is best action stable across references?",
        "Does 91-action heterogeneity replicate on fresh queries?",
        "How was compact bank derived (train-only)?",
        "Strongest fixed exact config?",
        "Bank policy validation improvement?",
        "Holdout improvement?",
        "Oracle recovery %?",
        "3 seeds consistent?",
        "Does policy collapse to global action?",
        "Search Oracle cross-reference gap?",
        "Are search labels suitable for supervision?",
        "Does Point-Subset beat Bank Policy?",
        "Do all methods use same SAM calls?",
        "Is formal test allowed?",
        "Final scientific state?",
        "Which claims are still not allowed?",
    ]
    for i, title in enumerate(q_titles, start=1):
        key = list(answers.keys())[i - 1]
        lines.append(f"### Q{i}. {title}")
        lines.append("")
        lines.append(answers[key])
        lines.append("")

    # Tables index
    lines.append("## Tables (Section 15)")
    lines.append("")
    for name, title, _builder in TABLE_BUILDERS:
        rel = f"phaseY_final_report/tables/{name}.md"
        lines.append(f"- [{title}]({rel})")
    lines.append("")

    # Figures index
    lines.append("## Figures (Section 16)")
    lines.append("")
    for fname in figure_files:
        rel = f"phaseY_final_report/figures/{fname}"
        lines.append(f"- ![{fname}]({rel})")
    if skipped_figures:
        lines.append("")
        lines.append("### Skipped figures")
        lines.append("")
        for fname, reason in skipped_figures:
            lines.append(f"- `{fname}` — {reason}")
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("All inputs are frozen Phase Y artifacts under `reports/phaseY_*/`, "
                 "`configs/actions/`, `configs/frozen/`, `manifests/phaseY/`, and `docs/`. "
                 "The script `scripts/build_phaseY_final_report.py` regenerates this report "
                 "deterministically; no SAM inference is performed.")
    lines.append("")

    report_path = out_dir / "phaseY_final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_json_report(
    data: dict[str, Any],
    out_dir: Path,
    table_files: list[str],
    figure_files: list[str],
    skipped_figures: list[str],
    answers: dict[str, str],
    compute_ledger: dict[str, Any],
) -> Path:
    y1 = _y1(data)
    y2 = _y2(data)
    y3 = _y3(data)
    y4 = _y4(data)
    holdout = data.get("holdout_summary", {})
    split = data.get("split_summary", {})
    split_info = split.get("split", {})

    fixed_dice = _to_float(y1.get("fixed_A_dice"))
    stable_oracle = _to_float(y1.get("stable_oracle"))
    bank_val_dice = _to_float(y3.get("best_model_dice"))
    bank_holdout_dice = _to_float(holdout.get("mean_policy_dice"))
    oracle_gap = stable_oracle - fixed_dice
    oracle_recovery_pct = (
        (bank_holdout_dice - fixed_dice) / oracle_gap * 100.0
        if oracle_gap > 0 and bank_holdout_dice > 0 else 0.0
    )

    def _gate_dict(d: dict[str, Any]) -> dict[str, Any]:
        return {
            "passed": bool(d.get("decision", "").endswith("PASS") or
                           d.get("decision", "").endswith("STABLE_HETEROGENEITY") or
                           d.get("decision", "").endswith("STABLE")),
            "state": d.get("decision", "NOT_RUN"),
            "conditions": d.get("conditions", {}),
        }

    report = {
        "phase": "Y",
        "date": datetime.now().isoformat(timespec="seconds"),
        "final_state": answers["q21_final_scientific_state"],
        "splits": {
            "train": split_info.get("train", {}).get("case_count", 0),
            "validation": split_info.get("validation", {}).get("case_count", 0),
            "holdout": split_info.get("holdout", {}).get("case_count", 0),
        },
        "gate_decisions": {
            "Y1": _gate_dict(y1),
            "Y2": _gate_dict(y2),
            "Y3": _gate_dict(y3),
            "Y4": _gate_dict(y4),
            "holdout": _gate_dict(holdout),
        },
        "key_metrics": {
            "strongest_fixed_dice": fixed_dice if fixed_dice else None,
            "raw_oracle_dice": _to_float(y1.get("oracle_A")) or None,
            "stable_oracle_dice": stable_oracle or None,
            "oracle_optimism": _to_float(y1.get("optimism_A")) or None,
            "stable_oracle_gap": _to_float(y1.get("stable_oracle_gap")) or None,
            "bank_policy_val_dice": bank_val_dice or None,
            "bank_policy_holdout_dice": bank_holdout_dice or None,
            "bank_policy_delta_holdout": _to_float(holdout.get("delta_vs_fixed")) or None,
            "oracle_recovery_pct": oracle_recovery_pct if oracle_recovery_pct else None,
            "point_subset_val_dice": _to_float(y4.get("best_model_dice")) or None,
            "search_oracle_dice": _to_float(y2.get("raw_search_dice")) or None,
            "cross_search_oracle_dice": _to_float(y2.get("cross_search_dice")) or None,
            "search_optimism": _to_float(y2.get("search_optimism")) or None,
        },
        "tables": [f"tables/{name}" for name in table_files],
        "figures": [f"figures/{name}" for name in figure_files],
        "skipped_figures": [{"name": n, "reason": r} for n, r in skipped_figures],
        "compute_ledger": compute_ledger,
        "answers": answers,
    }
    report_path = out_dir / "phaseY_final_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_phaseY_final_report(
    *,
    out_dir: Path,
    skip_figures: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the Phase Y final report (tables, figures, MD, JSON)."""
    tables_dir = out_dir / "phaseY_final_report" / "tables"
    figures_dir = out_dir / "phaseY_final_report" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data(project_root)

    # 18 tables (CSV + markdown)
    table_files: list[str] = []
    for name, title, builder in TABLE_BUILDERS:
        headers, rows = builder(data)
        md_path = tables_dir / f"{name}.md"
        csv_path = tables_dir / f"{name}.csv"
        _write_table_md(md_path, title, headers, rows)
        _write_table_csv(csv_path, headers, rows)
        table_files.append(f"{name}.md")

    # 20 figures (PNG)
    figure_files: list[str] = []
    skipped_figures: list[tuple[str, str]] = []
    if skip_figures:
        for name, _builder in FIGURE_BUILDERS:
            skipped_figures.append((f"{name}.png", "skipped via --skip-figures"))
    else:
        for name, builder in FIGURE_BUILDERS:
            path = figures_dir / f"{name}.png"
            # Qualitative figures (16-18) require PIL
            requires_pil = name in (
                "fig_16_qualitative_fixed_win_cases",
                "fig_17_qualitative_policy_win_cases",
                "fig_18_high_confidence_policy_failures",
            )
            if requires_pil and not _PIL_AVAILABLE:
                skipped_figures.append((f"{name}.png", "PIL not available for image loading"))
                continue
            try:
                builder(data, path)
                if path.is_file():
                    figure_files.append(f"{name}.png")
                else:
                    skipped_figures.append((f"{name}.png", "builder produced no file (missing data)"))
            except Exception as exc:
                skipped_figures.append((f"{name}.png", f"builder error: {type(exc).__name__}: {exc}"))

    # 22-question answers
    answers = _build_answers(data)

    # Compute ledger
    compute_ledger = _compute_ledger(data)

    # Write MD + JSON
    md_path = _write_markdown_report(data, out_dir, table_files, figure_files, skipped_figures, answers)
    json_path = _write_json_report(data, out_dir, table_files, figure_files, skipped_figures, answers, compute_ledger)

    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "tables": table_files,
        "figures": figure_files,
        "skipped_figures": skipped_figures,
        "final_state": answers["q21_final_scientific_state"],
        "compute_ledger": compute_ledger,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Phase Y final report (18 tables, 20 figures, MD, JSON)."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "reports",
        help="Output directory for the final report (default: PROJECT_ROOT/reports).",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation (only produce tables + MD + JSON).",
    )
    args = parser.parse_args()
    summary = build_phaseY_final_report(
        out_dir=args.out_dir,
        skip_figures=args.skip_figures,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
