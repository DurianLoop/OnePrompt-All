from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def format_decimal(value: Any, *, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _display_label(value: Any) -> str:
    text = str(value).replace("_", " ").lower()
    return text[:1].upper() + text[1:]


def _oracle_gap(summary: dict[str, Any]) -> Any:
    return summary.get("mean_regret_to_oracle", summary.get("dice_oracle_gap"))


def build_pivot_rows(
    monusac_summary: dict[str, Any],
    glas_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset": "MoNuSAC",
            "decision": "PIVOT_ORACLE_ANALYSIS",
            "episodes": monusac_summary.get("episode_count"),
            "actions": monusac_summary.get("action_count"),
            "best_fixed_action": monusac_summary.get("best_fixed_action_id"),
            "best_fixed_dice": monusac_summary.get("mean_best_fixed_dice"),
            "oracle_dice": monusac_summary.get("mean_oracle_dice"),
            "oracle_gap": _oracle_gap(monusac_summary),
            "aji_oracle_gap": monusac_summary.get("mean_aji_regret_to_oracle", monusac_summary.get("aji_oracle_gap")),
            "top_oracle_action_fraction": monusac_summary.get("top_oracle_action_fraction"),
            "allowed_claim": monusac_summary.get("recommended_claim", "oracle_analysis_only"),
            "policy_training": "prohibited",
        }
    ]
    if glas_summary is not None:
        rows.append(
            {
                "dataset": "GlaS",
                "decision": glas_summary.get("gate_f_status", glas_summary.get("status", "UNKNOWN")),
                "episodes": glas_summary.get("episode_count"),
                "actions": glas_summary.get("action_count", "N/A"),
                "best_fixed_action": glas_summary.get("best_fixed_action_id"),
                "best_fixed_dice": glas_summary.get("mean_best_fixed_dice"),
                "oracle_dice": glas_summary.get("mean_oracle_dice"),
                "oracle_gap": _oracle_gap(glas_summary),
                "aji_oracle_gap": glas_summary.get("mean_aji_regret_to_oracle", glas_summary.get("aji_oracle_gap")),
                "top_oracle_action_fraction": glas_summary.get("top_oracle_action_fraction"),
                "allowed_claim": "extension_smoke_only",
                "policy_training": "not_started",
            }
        )
    return rows


def _write_macros(
    path: Path,
    rows: list[dict[str, Any]],
    representation_summary: dict[str, Any] | None = None,
) -> None:
    monusac = rows[0]
    lines = [
        "% Auto-generated from real oracle-analysis artifacts.",
        r"\newcommand{\OracleAnalysisDecision}{PIVOT\_ORACLE\_ANALYSIS}",
        rf"\newcommand{{\OracleAnalysisBestFixed}}{{{_latex_escape(monusac['best_fixed_action'])}}}",
        rf"\newcommand{{\OracleAnalysisBestFixedDice}}{{{format_decimal(monusac['best_fixed_dice'], digits=3)}}}",
        rf"\newcommand{{\OracleAnalysisOracleDice}}{{{format_decimal(monusac['oracle_dice'], digits=3)}}}",
        rf"\newcommand{{\OracleAnalysisGap}}{{{format_decimal(monusac['oracle_gap'], digits=3)}}}",
        rf"\newcommand{{\OracleAnalysisAJIGap}}{{{format_decimal(monusac['aji_oracle_gap'], digits=3)}}}",
        rf"\newcommand{{\OracleAnalysisTopOracleFraction}}{{{format_decimal(monusac['top_oracle_action_fraction'], digits=3)}}}",
        r"\newcommand{\OracleAnalysisClaim}{oracle\_analysis\_only}",
        r"\newcommand{\OracleAnalysisPolicyTraining}{prohibited}",
    ]
    if representation_summary is not None:
        lines.extend(
            [
                rf"\newcommand{{\RepresentationAnalysisStatus}}{{{_latex_escape(representation_summary['status'])}}}",
                rf"\newcommand{{\RepresentationAnalysisClaim}}{{{_latex_escape(representation_summary['representation_claim'])}}}",
                rf"\newcommand{{\RepresentationAnalysisTopTwoMargin}}{{{format_decimal(representation_summary['mean_top2_dice_margin'], digits=3)}}}",
                rf"\newcommand{{\RepresentationAnalysisNearTieFraction}}{{{format_decimal(representation_summary['near_tie_episode_fraction'], digits=3)}}}",
                rf"\newcommand{{\RepresentationAnalysisPointPrecision}}{{{format_decimal(representation_summary['mean_candidate_point_precision'], digits=3)}}}",
                rf"\newcommand{{\RepresentationAnalysisComponentCoverage}}{{{format_decimal(representation_summary['mean_candidate_component_coverage'], digits=3)}}}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pivot_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated from real oracle-analysis artifacts.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Gate-F pivot status from real pilot artifacts. "
            r"Oracle columns are analysis upper bounds, not deployable method outputs; "
            r"no privileged-input row is included.}"
        ),
        r"\label{tab:oracle-pivot}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Dataset & Decision & Episodes & Actions & Fixed Dice & Oracle Dice & Claim \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["dataset"]),
                    _latex_escape(_display_label(row["decision"])),
                    str(row["episodes"]),
                    str(row["actions"]),
                    format_decimal(row["best_fixed_dice"], digits=3),
                    format_decimal(row["oracle_dice"], digits=3),
                    _latex_escape(_display_label(row["allowed_claim"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_action_table(path: Path, action_rows: list[dict[str, str]]) -> None:
    lines = [
        "% Auto-generated from reports/fixed_oracle_analysis/action_summary.csv.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{MoNuSAC action-oracle summary after the Gate-F pivot. "
            r"Oracle wins are post-hoc analysis labels, not deployable method selections; "
            r"no privileged-input row is included.}"
        ),
        r"\label{tab:action-summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Action & Dice & Calls & Oracle wins \\",
        r"\midrule",
    ]
    for row in action_rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["action_id"]),
                    format_decimal(row["mean_dice"], digits=3),
                    format_decimal(row["mean_sam_calls"], digits=2),
                    str(int(float(row["oracle_count"]))),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_representation_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "% Auto-generated from reports/representation_analysis/representation_summary.json.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Representation diagnostics for the Gate-F pivot. "
            r"The row uses fixed artifact diagnostics only; no privileged-input row is included.}"
        ),
        r"\label{tab:representation-pivot}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Status & Top-2 margin & Near ties & Point precision & Component coverage & Precision-regret $r$ \\",
        r"\midrule",
        " & ".join(
            [
                _latex_escape(_display_label(summary["status"])),
                format_decimal(summary["mean_top2_dice_margin"], digits=3),
                format_decimal(summary["near_tie_episode_fraction"], digits=3),
                format_decimal(summary["mean_candidate_point_precision"], digits=3),
                format_decimal(summary["mean_candidate_component_coverage"], digits=3),
                format_decimal(summary["pearson_precision_vs_regret"], digits=3),
            ]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    outputs: dict[str, Path],
    representation_summary: dict[str, Any] | None = None,
) -> None:
    monusac = rows[0]
    lines = [
        "# Oracle Analysis Paper Assets",
        "",
        "Status: `oracle_analysis_only`",
        "",
        "These assets are generated from real Gate-F pivot artifacts. They do not contain simulated results and do not claim adaptive-policy success.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Decision | `{monusac['decision']}` |",
        f"| Best fixed action | `{monusac['best_fixed_action']}` |",
        f"| Mean best fixed Dice | {format_decimal(monusac['best_fixed_dice'], digits=6)} |",
        f"| Mean oracle Dice | {format_decimal(monusac['oracle_dice'], digits=6)} |",
        f"| Oracle gap | {format_decimal(monusac['oracle_gap'], digits=6)} |",
        f"| AJI oracle gap | {format_decimal(monusac['aji_oracle_gap'], digits=6)} |",
        f"| Allowed claim | `{monusac['allowed_claim']}` |",
        f"| Policy training | `{monusac['policy_training']}` |",
    ]
    if representation_summary is not None:
        lines.extend(
            [
                f"| Representation pivot | `{representation_summary['status']}` |",
                f"| Representation claim | `{representation_summary['representation_claim']}` |",
                f"| Mean top-2 Dice margin | {format_decimal(representation_summary['mean_top2_dice_margin'], digits=6)} |",
                f"| Near-tie episode fraction | {format_decimal(representation_summary['near_tie_episode_fraction'], digits=6)} |",
                f"| Candidate point precision | {format_decimal(representation_summary['mean_candidate_point_precision'], digits=6)} |",
                f"| Candidate component coverage | {format_decimal(representation_summary['mean_candidate_component_coverage'], digits=6)} |",
            ]
        )
        for action in representation_summary.get("prohibited_next_actions", []):
            lines.append(f"| Prohibited after representation pivot | `{action}` |")
    lines.extend(["", "Generated files:"])
    for label, output_path in outputs.items():
        lines.append(f"- `{label}`: `{output_path}`")
    lines.extend(
        [
            "",
            "Claim boundary:",
            "",
            "- MoNuSAC policy training remains prohibited after the failed Gate F.",
            "- Post-hoc MoNuSAC action-bank redesign remains prohibited.",
            "- GlaS evidence is a separate extension smoke unless a later predeclared protocol provides enough episodes.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_oracle_analysis_assets(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    monusac_summary = _read_json(project_root / "reports" / "fixed_oracle_analysis" / "fixed_oracle_summary.json")
    glas_path = project_root / "reports" / "oracle_gap_pilot_glas" / "oracle_gap_pilot_summary.json"
    glas_summary = _read_json(glas_path) if glas_path.exists() else None
    representation_path = project_root / "reports" / "representation_analysis" / "representation_summary.json"
    representation_summary = _read_json(representation_path) if representation_path.exists() else None
    action_rows = _read_csv(project_root / "reports" / "fixed_oracle_analysis" / "action_summary.csv")
    rows = build_pivot_rows(monusac_summary, glas_summary)
    outputs = {
        "macros": project_root / "paper" / "generated" / "oracle_analysis_macros.tex",
        "pivot_table": project_root / "paper" / "tables" / "table_oracle_pivot.tex",
        "action_table": project_root / "paper" / "tables" / "table_action_summary.tex",
        "report": project_root / "reports" / "oracle_analysis_paper_assets.md",
    }
    if representation_summary is not None:
        outputs["representation_table"] = project_root / "paper" / "tables" / "table_representation_pivot.tex"
    _write_macros(outputs["macros"], rows, representation_summary)
    _write_pivot_table(outputs["pivot_table"], rows)
    _write_action_table(outputs["action_table"], action_rows)
    if representation_summary is not None:
        _write_representation_table(outputs["representation_table"], representation_summary)
    _write_report(outputs["report"], rows, outputs, representation_summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export claim-safe oracle-analysis paper assets.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_oracle_analysis_assets(args.project_root)


if __name__ == "__main__":
    main()
