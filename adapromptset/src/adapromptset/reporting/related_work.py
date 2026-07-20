from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROWS = [
    {
        "label": "GF-SAM",
        "reference_guided": True,
        "negative_prompting": "background_reference_evidence",
        "learned_prompt_policy": "none",
        "grouping": "graph_rule",
        "budgeted_complete_action_selection": False,
        "adapromptset_boundary": "deterministic_graph_baseline",
    },
    {
        "label": "SynPo",
        "reference_guided": True,
        "negative_prompting": "explicit_rule_negatives",
        "learned_prompt_policy": "none",
        "grouping": "fixed_kmeans",
        "budgeted_complete_action_selection": False,
        "adapromptset_boundary": "fixed_negative_prompt_rule",
    },
    {
        "label": "AlignSAM",
        "reference_guided": False,
        "negative_prompting": "learned_label_module",
        "learned_prompt_policy": "sequential_rl",
        "grouping": "patch_center_actions",
        "budgeted_complete_action_selection": False,
        "adapromptset_boundary": "one_shot_offline_set_level",
    },
    {
        "label": "Point Prompt Defender",
        "reference_guided": False,
        "negative_prompting": "adversarial_prompt_optimization",
        "learned_prompt_policy": "adversarial_dqn",
        "grouping": "dual_space_graph_environment",
        "budgeted_complete_action_selection": False,
        "adapromptset_boundary": "not_reference_constructed_action_bank",
    },
]


def build_related_work_summary() -> dict[str, Any]:
    return {
        "schema_version": "adapromptset.related_work_boundaries.v1",
        "status": "PASS_RELATED_WORK_BOUNDARIES",
        "work_count": len(ROWS),
        "claims": {
            "first_to_learn_prompts": False,
            "all_prior_methods_fixed_topk_or_clustering": False,
            "background_reference_is_explicit_negative_prompting": False,
        },
        "rows": ROWS,
    }


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Related Work Boundaries",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Work | Reference-guided | Negative evidence | Learned prompt policy | Grouping | Budgeted complete action | AdaPromptSet boundary |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {label} | {reference_guided} | `{negative_prompting}` | `{learned_prompt_policy}` | `{grouping}` | {budgeted} | `{boundary}` |".format(
                label=row["label"],
                reference_guided=_format_bool(row["reference_guided"]),
                negative_prompting=row["negative_prompting"],
                learned_prompt_policy=row["learned_prompt_policy"],
                grouping=row["grouping"],
                budgeted=_format_bool(row["budgeted_complete_action_selection"]),
                boundary=row["adapromptset_boundary"],
            )
        )
    lines.extend(
        [
            "",
            "Claim guardrails:",
            "",
            "- Do not claim first-to-learn prompting.",
            "- Do not describe GF-SAM background-reference evidence as SynPo-style explicit negative point prompting.",
            "- Treat AlignSAM and Point Prompt Defender as learned/RL prompt-optimization boundaries rather than identical tasks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&")


DISPLAY = {
    "background_reference_evidence": "background ref.",
    "explicit_rule_negatives": "explicit neg.",
    "learned_label_module": "label module",
    "adversarial_prompt_optimization": "adv. prompt opt.",
    "sequential_rl": "seq. RL",
    "adversarial_dqn": "adv. DQN",
    "graph_rule": "graph rule",
    "fixed_kmeans": "fixed K-means",
    "patch_center_actions": "patch centers",
    "dual_space_graph_environment": "graph env.",
}


def _display(text: str) -> str:
    return DISPLAY.get(text, text)


def _write_tex(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Related-work boundaries used by the current oracle-analysis skeleton. The table is generated from the project boundary registry and does not assert an adaptive-policy result.}",
        r"\label{tab:related-work-boundaries}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Work & Ref. guided & Negative evidence & Learned policy & Grouping & Budgeted complete action \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["label"]),
                _format_bool(row["reference_guided"]),
                _tex_escape(_display(row["negative_prompting"])),
                _tex_escape(_display(row["learned_prompt_policy"])),
                _tex_escape(_display(row["grouping"])),
                _format_bool(row["budgeted_complete_action_selection"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{0.3em}",
            r"\footnotesize AdaPromptSet boundary: one-shot, offline, complete prompt-set selection under frozen matching and segmentation modules.",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_related_work_assets(project_root: Path) -> dict[str, Path]:
    summary = build_related_work_summary()
    report_dir = project_root / "reports" / "related_work_boundaries"
    table_path = project_root / "paper" / "tables" / "table_related_work.tex"
    outputs = {
        "json": report_dir / "related_work_boundaries.json",
        "markdown": report_dir / "related_work_boundaries.md",
        "table": table_path,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    _write_tex(outputs["table"], summary)
    return outputs
