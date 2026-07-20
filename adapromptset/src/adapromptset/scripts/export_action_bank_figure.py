from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _tex_escape(text: Any) -> str:
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _family_counts(bank: dict[str, Any]) -> dict[str, int]:
    valid_actions = [action for action in bank["actions"] if action.get("valid")]
    counts = Counter(action["family"] for action in valid_actions)
    return dict(sorted(counts.items()))


def _action_bank_tikz(bank: dict[str, Any]) -> str:
    family_counts = _family_counts(bank)
    bank_hash = str(bank["bank_hash"])[:12]
    action_count = int(bank["action_count"])
    valid_action_count = int(bank["valid_action_count"])
    invalid_action_count = int(bank.get("invalid_action_count", action_count - valid_action_count))
    source = bank.get("source", {})
    positive_limit = source.get("positive_limit", "unknown")
    implemented = ", ".join(f"{_tex_escape(name)}={count}" for name, count in family_counts.items())
    bank_name = _tex_escape(bank["name"])
    return rf"""% Auto-generated action-bank structure figure for the current Gate-F pivot state.
\begin{{figure*}}[t]
\centering
\begin{{tikzpicture}}[
  node distance=0.55cm and 0.65cm,
  block/.style={{draw, rounded corners=2pt, align=center, minimum height=0.72cm, font=\scriptsize}},
  source/.style={{block, fill=gray!8, draw=gray!65!black, minimum width=3.2cm}},
  family/.style={{block, fill=blue!7, draw=blue!55!black, minimum width=2.75cm}},
  rescue/.style={{block, fill=orange!10, draw=orange!70!black, minimum width=2.75cm}},
  deferred/.style={{block, fill=red!7, draw=red!60!black, minimum width=3.2cm}},
  metric/.style={{block, fill=green!7, draw=green!45!black, minimum width=2.4cm}},
  arrow/.style={{->, >=Stealth, thick}}
]
\node[source] (cand) {{Shared candidate pool\\$C^+$ top {positive_limit} and $C^-$}};
\node[family, below left=of cand, xshift=-2.1cm] (topk) {{Top-K one group\\1 SAM call\\points: 1--5}};
\node[family, below=of cand] (posneg) {{Positive + negative\\1 SAM call\\points: 4--8}};
\node[family, below right=of cand, xshift=2.1cm] (ind) {{Independent groups\\3 or 5 SAM calls\\one point per group}};
\node[rescue, below=of posneg, yshift=-0.1cm] (rescue) {{Rescue split/local negatives\\2 or 5 SAM calls\\group-specific negatives}};
\node[deferred, below left=of rescue, xshift=-1.6cm] (deferred) {{Deferred in current MVP subset\\KMeans, density, GF-SAM-style graph\\not claimed as implemented}};
\node[metric, below right=of rescue, xshift=1.6cm] (cost) {{Frozen accounting\\valid {valid_action_count}/{action_count}; invalid {invalid_action_count}\\bank hash {bank_hash}}};
\node[source, below=of rescue, yshift=-1.25cm, minimum width=7.1cm] (selector) {{Complete prompt-set proposals only\\policy path remains prohibited after Gate F; current use is oracle analysis}};

\draw[arrow] (cand) -- (topk);
\draw[arrow] (cand) -- (posneg);
\draw[arrow] (cand) -- (ind);
\draw[arrow] (topk) -- (selector);
\draw[arrow] (posneg) -- (rescue);
\draw[arrow] (ind) -- (rescue);
\draw[arrow] (rescue) -- (selector);
\draw[arrow] (deferred) -- (selector);
\draw[arrow] (cost) -- (selector);
\end{{tikzpicture}}
\caption{{Current AdaPromptSet action-bank structure. The figure is generated from \texttt{{{bank_name}}} and summarizes complete prompt-set proposals with frozen SAM-call and prompt-point accounting. Deferred families are shown as missing from the current MVP subset rather than as implemented methods. Implemented valid families: {implemented}.}}
\label{{fig:action-bank}}
\end{{figure*}}
"""


def _checks(text: str, bank: dict[str, Any]) -> dict[str, bool]:
    lowered = text.lower()
    family_counts = _family_counts(bank)
    return {
        "tikz_present": r"\begin{tikzpicture}" in text,
        "source_bank_present": _tex_escape(bank["name"]) in text,
        "topk_family_present": "top-k one group" in lowered,
        "positive_negative_family_present": "positive + negative" in lowered,
        "independent_groups_family_present": "independent groups" in lowered,
        "rescue_family_present": "rescue split/local negatives" in lowered,
        "deferred_families_disclosed": "deferred in current mvp subset" in lowered
        and "not claimed as implemented" in lowered,
        "sam_calls_and_points_present": "sam call" in lowered and "points" in lowered,
        "complete_prompt_set_boundary_present": "complete prompt-set proposals only" in lowered,
        "valid_counts_match_source": int(bank["valid_action_count"]) == sum(family_counts.values()),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Action Bank Figure",
        "",
        f"Status: `{summary['status']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        "",
        f"Source bank: `{summary['source_bank']['path']}`",
        f"Bank hash: `{summary['source_bank']['bank_hash']}`",
        f"Figure: `{summary['figure']['path']}`",
        f"Figure SHA256: `{summary['figure']['sha256']}`",
        "",
        "## Family Counts",
        "",
    ]
    for family, count in summary["family_counts"].items():
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Checks", ""])
    for name, passed in summary["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_action_bank_figure(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    bank_path = project_root / "configs" / "actions" / "prompt_set_bank_v2.json"
    bank = _read_json(bank_path)
    outputs = {
        "tex": project_root / "paper" / "figures" / "action_bank.tex",
        "json": project_root / "reports" / "action_bank_figure.json",
        "markdown": project_root / "reports" / "action_bank_figure.md",
    }
    outputs["tex"].parent.mkdir(parents=True, exist_ok=True)
    text = _action_bank_tikz(bank)
    outputs["tex"].write_text(text, encoding="utf-8")
    checks = _checks(text, bank)
    summary = {
        "schema_version": "adapromptset.action_bank_figure.v1",
        "status": "PASS_ACTION_BANK_FIGURE" if all(checks.values()) else "FAIL_ACTION_BANK_FIGURE",
        "claim_boundary": "oracle_analysis_only_current_mvp_subset",
        "source_bank": {
            "path": _relative(project_root, bank_path),
            "sha256": _sha256(bank_path),
            "name": bank["name"],
            "bank_hash": bank["bank_hash"],
            "action_count": bank["action_count"],
            "valid_action_count": bank["valid_action_count"],
            "invalid_action_count": bank.get("invalid_action_count", 0),
        },
        "family_counts": _family_counts(bank),
        "figure": {
            "path": _relative(project_root, outputs["tex"]),
            "bytes": outputs["tex"].stat().st_size,
            "sha256": _sha256(outputs["tex"]),
        },
        "checks": checks,
        "prohibited_claims": [
            "full_action_bank_complete",
            "adaptive_policy_success",
            "test_time_action_search",
        ],
        "deferred_families": [
            "kmeans_groups",
            "density_groups",
            "gf_sam_style_graph_groups",
        ],
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the claim-safe AdaPromptSet action-bank figure.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_action_bank_figure(args.project_root)


if __name__ == "__main__":
    main()
