from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _f(value: str | float | int | None, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def _tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _table(headers: list[str], rows: Iterable[list[object]], caption: str, label: str) -> str:
    spec = "l" * len(headers)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _plot_assets(project_root: Path, figures_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    r1_rows = _read_csv(project_root / "reports/phase3_glas/gate_r1_region_proposal/r1_family_summary.csv")
    q_rows = _read_csv(project_root / "reports/phase3_glas/gate_r1_region_proposal/r1_query_level_metrics.csv")
    grid_rows = _read_csv(project_root / "reports/phase3_glas/gate_r1_region_proposal/parameter_grid_results.csv")
    control_rows = _read_csv(project_root / "reports/phase3_interpretation_audit/phase2d_control_decomposition.csv")
    failure_rows = _read_csv(project_root / "reports/phase3_interpretation_audit/p3_failure_taxonomy.csv")

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    names = [r["family"].replace("R00_", "R00\n").replace("R03_", "R03\n").replace("R04_", "R04\n").replace("R05_", "R05\n") for r in r1_rows]
    metrics = [float(r["mean_metric"]) for r in r1_rows]
    controls = [float(r["mean_control"]) for r in r1_rows]
    wins = [float(r["win_rate"]) for r in r1_rows]
    x = list(range(len(names)))
    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], metrics, width=width, label="automatic metric", color="#3b6ea8")
    ax1.bar([i + width / 2 for i in x], controls, width=width, label="matched control", color="#b8b8b8")
    ax1.set_ylabel("Mean metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(x, wins, color="#b33f3f", marker="o", linewidth=2, label="win rate")
    ax2.axhline(0.70, color="#b33f3f", linestyle="--", linewidth=1)
    ax2.set_ylabel("Win rate")
    ax2.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = figures_dir / "r1_family_metrics.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    fig, ax = plt.subplots(figsize=(4.1, 3.3))
    for family, marker, color in [
        ("R00_direct_matching_mask", "o", "#3b6ea8"),
        ("R03_automatic_union_box", "s", "#6a9f58"),
        ("R04_automatic_largest_component_box", "^", "#a8753b"),
        ("R05_automatic_top3_component_boxes", "D", "#8e5ea2"),
    ]:
        sub = [r for r in q_rows if r["family"] == family]
        ax.scatter(
            [float(r["control_value"]) for r in sub],
            [float(r["metric_value"]) for r in sub],
            s=18,
            alpha=0.75,
            marker=marker,
            color=color,
            label=family.split("_")[0],
        )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Matched control")
    ax.set_ylabel("Automatic proposal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = figures_dir / "r1_automatic_vs_control_scatter.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    by_family_q: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in grid_rows:
        by_family_q[row["family"]][float(row["contrast_quantile"])].append(float(row["mean_metric"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for family, values_by_q in sorted(by_family_q.items()):
        qs = sorted(values_by_q)
        means = [sum(values_by_q[q]) / len(values_by_q[q]) for q in qs]
        ax.plot(qs, means, marker="o", linewidth=1.6, label=family.split("_")[0])
    ax.set_xlabel("Contrast quantile")
    ax.set_ylabel("Selection-pool mean metric")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = figures_dir / "r1_parameter_grid_curves.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    p3_p4 = [
        r
        for r in control_rows
        if r["prompt_family"] in {"P3_component_grouped_centroids", "P4_union_mask_bounding_box"}
    ]
    labels = [f"{r['segmenter']} {r['prompt_family'].split('_')[0]}" for r in p3_p4]
    margins = [float(r["mean_privileged_minus_c1"]) for r in p3_p4]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    colors = ["#b33f3f" if m < 0 else "#3b6ea8" for m in margins]
    ax.bar(labels, margins, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Margin vs C1 random")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    out = figures_dir / "phase2d_point_vs_region_upper_bound.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    counts = Counter(r["failure_type"] for r in failure_rows)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = list(counts)
    ax.bar(labels, [counts[k] for k in labels], color="#5d7f87")
    ax.set_ylabel("Query-segmenter rows")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    out = figures_dir / "p3_failure_taxonomy.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    states = [
        ("Interpretation", "complete"),
        ("R1", "failed"),
        ("R2", "blocked"),
        ("R3", "blocked"),
        ("Router", "prohibited"),
        ("Test", "locked"),
    ]
    for i, (name, state) in enumerate(states):
        color = "#6a9f58" if state == "complete" else "#b33f3f" if state == "failed" else "#b8b8b8"
        ax.add_patch(plt.Rectangle((i, 0), 0.85, 0.5, color=color, alpha=0.9))
        ax.text(i + 0.425, 0.25, f"{name}\n{state}", ha="center", va="center", fontsize=8, color="white" if state in {"complete", "failed"} else "black")
        if i < len(states) - 1:
            ax.annotate("", xy=(i + 0.98, 0.25), xytext=(i + 0.85, 0.25), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.set_xlim(-0.1, len(states))
    ax.set_ylim(-0.1, 0.65)
    ax.axis("off")
    fig.tight_layout()
    out = figures_dir / "phase3_gate_state_flow.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(str(out))

    return created


def export_assets(project_root: Path) -> dict:
    tables_dir = project_root / "paper_phase3/tables"
    figures_dir = project_root / "paper_phase3/figures"

    final_report = _read_json(project_root / "reports/phase3_final_report.json")
    r1_rows = _read_csv(project_root / "reports/phase3_glas/gate_r1_region_proposal/r1_family_summary.csv")
    control_rows = _read_csv(project_root / "reports/phase3_interpretation_audit/phase2d_control_decomposition.csv")
    selected_configs = _read_json(project_root / "reports/phase3_glas/gate_r1_region_proposal/selected_region_configs.json")

    p3_p4_rows = [
        row
        for row in control_rows
        if row["prompt_family"] in {"P3_component_grouped_centroids", "P4_union_mask_bounding_box"}
    ]
    _write(
        tables_dir / "table_phase2d_interpretation.tex",
        _table(
            ["Segmenter", "Prompt", "Dice", "C1", "C2", "Margin vs C1", "Margin vs C2", "Win vs C2"],
            [
                [
                    row["segmenter"],
                    row["prompt_family"].replace("_", " "),
                    _f(row["mean_privileged_dice"]),
                    _f(row["mean_c1_random_dice"]),
                    _f(row["mean_c2_alternative_reference_dice"]),
                    _f(row["mean_privileged_minus_c1"]),
                    _f(row["mean_privileged_minus_c2"]),
                    _f(row["win_rate_vs_c2"]),
                ]
                for row in p3_p4_rows
            ],
            "Phase 2D interpretation correction. P3 grouped sparse points fail, while the privileged P4 region box remains effective; C2 is an alternative-reference control.",
            "tab:phase2d_interpretation",
        ),
    )

    _write(
        tables_dir / "table_control_decomposition.tex",
        _table(
            ["Segmenter", "Prompt", "P - C1", "P - C2", "C1 - C2", "C2 label"],
            [
                [
                    row["segmenter"],
                    row["prompt_family"].split("_")[0],
                    _f(row["mean_privileged_minus_c1"]),
                    _f(row["mean_privileged_minus_c2"]),
                    _f(row["mean_c1_minus_c2"]),
                    row["c2_label"],
                ]
                for row in control_rows
            ],
            "C1/C2 decomposition over all Phase 2D prompt families. C2 is retained historically but interpreted as an alternative-reference control.",
            "tab:control_decomposition",
        ),
    )

    action_rows = [
        ["R00", "Direct matching mask", "mask", "0", "automatic", "raw proposal baseline"],
        ["R01", "Full-image box", "box", "1", "control", "trivial control"],
        ["R02", "Size-matched random box", "box", "1", "control", "matched random control"],
        ["R03", "Automatic union box", "box", "1", "automatic", "tight union of proposals"],
        ["R04", "Largest-component box", "box", "1", "automatic", "largest proposal component"],
        ["R05", "Top-3 component boxes", "box", "<=3", "automatic", "largest three components"],
        ["R06", "Hard mask prompt", "mask", "1", "automatic", "blocked by R1"],
        ["R07", "Soft-logit mask prompt", "mask", "1", "automatic", "blocked by R1"],
        ["R08", "Box + soft mask", "hybrid", "1", "automatic", "blocked by R1"],
        ["R09", "Top-point baseline", "point", "1", "automatic", "not grouped P3 reopening"],
        ["U01-U03", "GT region upper bounds", "privileged", "1-3", "upper bound", "not automatic"],
    ]
    _write(
        tables_dir / "table_region_action_definitions.tex",
        _table(
            ["ID", "Action", "Type", "Calls", "Role", "Phase 3 note"],
            action_rows,
            "Pre-registered Phase 3 region action bank. Privileged upper bounds are separated from automatic actions.",
            "tab:region_actions",
        ),
    )

    _write(
        tables_dir / "table_r1_region_proposal_quality.tex",
        _table(
            ["Family", "Route", "Metric", "Control", "Improvement", "Win", "Recall", "Empty", "Status"],
            [
                [
                    row["family"].replace("_", " "),
                    row["route"],
                    _f(row["mean_metric"]),
                    _f(row["mean_control"]),
                    _f(row["mean_improvement"]),
                    _f(row["win_rate"]),
                    _f(row["foreground_recall"]),
                    _f(row["empty_rate"]),
                    row["status"],
                ]
                for row in r1_rows
            ],
            "Gate R1 proposal quality on 30 strict-validation queries. No automatic family passed the pre-registered thresholds.",
            "tab:r1_quality",
        ),
    )

    _write(
        tables_dir / "table_selected_region_configs.tex",
        _table(
            ["Family", "Aggregation", "q", "min area", "closing", "top k", "positive limit"],
            [
                [
                    family.replace("_", " "),
                    cfg.get("aggregation", "mean"),
                    cfg["contrast_quantile"],
                    cfg["min_component_area_ratio"],
                    cfg["closing_radius"],
                    cfg["top_component_count"],
                    cfg["positive_limit"],
                ]
                for family, cfg in selected_configs.items()
            ],
            "Global region configurations selected before strict validation. Selection used only the 40/15 internal protocol.",
            "tab:selected_configs",
        ),
    )

    gate_states = final_report["gate_states"]
    _write(
        tables_dir / "table_downstream_gate_status.tex",
        _table(
            ["Stage", "State", "Reported value"],
            [
                ["R2 SAM region viability", gate_states["R2"], "not run"],
                ["SAM gain", gate_states["R2"], "N/A"],
                ["Region recovery", gate_states["R2"], "N/A"],
                ["R3 oracle gap", gate_states["R3"], "N/A"],
                ["Router", gate_states["router"], "not trained"],
                ["Formal test", gate_states["formal_test"], "testA/testB locked"],
            ],
            "Downstream Phase 3 quantities are explicitly blocked by R1 failure rather than silently omitted.",
            "tab:downstream_status",
        ),
    )

    _write(
        tables_dir / "table_compute_and_access.tex",
        _table(
            ["Item", "Value"],
            [
                ["Phase 3 GPU hours", _f(final_report["compute"]["phase3_gpu_hours"], 6)],
                ["Budget", _f(final_report["compute"]["phase3_budget_gpu_hours"], 0)],
                ["testA", final_report["formal_test"]["testA_status"]],
                ["testB", final_report["formal_test"]["testB_status"]],
                ["Preregistration commit", final_report["preregistration_commit"]],
                ["Final state", final_report["final_state"]],
            ],
            "Compute, access, and final-state ledger for Phase 3.",
            "tab:compute_access",
        ),
    )

    figure_paths = _plot_assets(project_root, figures_dir)
    phase2_figure_sources = {
        "phase2d_representative_failure_panels.png": project_root
        / "paper_phase2/figures/gate_d_confirmation/representative_failure_panels.png",
        "phase2d_dice_vs_component_count.png": project_root
        / "paper_phase2/figures/gate_d_confirmation/dice_vs_component_count.png",
    }
    for name, source in phase2_figure_sources.items():
        if source.exists():
            destination = figures_dir / name
            shutil.copyfile(source, destination)
            figure_paths.append(str(destination))
    summary = {
        "status": "PASS_PHASE3_PAPER_ASSETS",
        "tables": sorted(str(p.relative_to(project_root)) for p in tables_dir.glob("*.tex")),
        "figures": sorted(str(Path(p).relative_to(project_root)) for p in figure_paths),
        "claim_boundary": final_report["claim_boundary"]["allowed"],
    }
    _write(project_root / "reports/phase3_paper_assets.json", json.dumps(summary, indent=2) + "\n")
    _write(
        project_root / "reports/phase3_paper_assets.md",
        "# Phase 3 Paper Assets\n\n"
        f"Status: `{summary['status']}`\n\n"
        "Tables:\n"
        + "\n".join(f"- `{item}`" for item in summary["tables"])
        + "\n\nFigures:\n"
        + "\n".join(f"- `{item}`" for item in summary["figures"])
        + "\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    summary = export_assets(args.project_root.resolve())
    print(json.dumps(summary, indent=2))
    return 0
