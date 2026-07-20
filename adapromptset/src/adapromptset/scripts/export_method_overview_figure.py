from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _method_overview_tikz() -> str:
    return r"""% Auto-generated method overview for the Gate-F pivot state.
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
  node distance=0.75cm and 0.9cm,
  block/.style={draw, rounded corners=2pt, align=center, minimum width=2.6cm, minimum height=0.78cm, font=\small},
  frozen/.style={block, fill=blue!8, draw=blue!55!black},
  learned/.style={block, fill=green!8, draw=green!45!black},
  trainonly/.style={block, fill=orange!10, draw=orange!60!black},
  testtime/.style={block, fill=gray!8, draw=gray!65!black},
  stop/.style={block, fill=red!8, draw=red!60!black},
  arrow/.style={->, >=Stealth, thick}
]
\node[testtime] (ref) {Reference image\\and mask};
\node[testtime, right=of ref] (query) {Query image};
\node[frozen, below=of ref, xshift=1.75cm] (matcher) {Frozen matcher\\DINOv2-L};
\node[testtime, below=of matcher] (evidence) {Matching evidence\\candidates $C^+, C^-$};
\node[testtime, below left=of evidence, xshift=-0.35cm] (bank) {Predeclared\\prompt-set bank};
\node[trainonly, below right=of evidence, xshift=0.35cm] (labels) {Train/validation only\\utility labels};
\node[learned, below=of labels] (policy) {Learned utility predictor\\only if Gate F passes};
\node[stop, below=of bank] (pivot) {Policy is prohibited after Gate F\\current oracle-analysis pivot};
\node[testtime, below=of pivot, xshift=2.65cm] (select) {Test-time safe path\\fixed/oracle-analysis assets};
\node[frozen, below=of select] (segmenter) {Frozen segmenter\\SAM-B};
\node[testtime, below=of segmenter] (report) {Claim-safe reports\\tables and figures};

\draw[arrow] (ref) -- (matcher);
\draw[arrow] (query) -- (matcher);
\draw[arrow] (matcher) -- (evidence);
\draw[arrow] (evidence) -- (bank);
\draw[arrow] (evidence) -- (labels);
\draw[arrow] (labels) -- (policy);
\draw[arrow] (bank) -- (pivot);
\draw[arrow] (pivot) -- (select);
\draw[arrow] (select) -- (segmenter);
\draw[arrow] (segmenter) -- (report);
\draw[arrow, dashed] (policy) -- node[font=\scriptsize, align=center, right]{blocked in\\current evidence} (select);
\end{tikzpicture}
\caption{AdaPromptSet method overview under the current Gate-F pivot. Frozen modules and test-time-safe paths are separated from train/validation-only utility labels and the learned predictor path.}
\label{fig:method-overview}
\end{figure*}
"""


def _checks(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "frozen_matcher_present": "frozen matcher" in lowered and "dinov2-l" in lowered,
        "frozen_segmenter_present": "frozen segmenter" in lowered and "sam-b" in lowered,
        "train_only_utility_labels_present": "train/validation only" in lowered and "utility labels" in lowered,
        "test_time_safe_path_present": "test-time safe path" in lowered,
        "gate_f_policy_prohibition_present": "policy is prohibited after gate f" in lowered,
        "tikz_present": r"\begin{tikzpicture}" in text,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Method Overview Figure",
        "",
        f"Status: `{summary['status']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        "",
        f"Figure: `{summary['figure']['path']}`",
        f"SHA256: `{summary['figure']['sha256']}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_overview_figure(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    outputs = {
        "tex": project_root / "paper" / "figures" / "method_overview.tex",
        "json": project_root / "reports" / "method_overview_figure.json",
        "markdown": project_root / "reports" / "method_overview_figure.md",
    }
    outputs["tex"].parent.mkdir(parents=True, exist_ok=True)
    text = _method_overview_tikz()
    outputs["tex"].write_text(text, encoding="utf-8")
    checks = _checks(text)
    summary = {
        "schema_version": "adapromptset.method_overview_figure.v1",
        "status": "PASS_METHOD_OVERVIEW_FIGURE" if all(checks.values()) else "FAIL_METHOD_OVERVIEW_FIGURE",
        "claim_boundary": "oracle_analysis_only",
        "figure": {
            "path": _relative(project_root, outputs["tex"]),
            "bytes": outputs["tex"].stat().st_size,
            "sha256": _sha256(outputs["tex"]),
        },
        "checks": checks,
        "prohibited_claims": [
            "adaptive_policy_success",
            "end_to_end_differentiable",
            "test_time_action_search",
        ],
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the claim-safe AdaPromptSet method overview figure.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    write_method_overview_figure(args.project_root)


if __name__ == "__main__":
    main()
