from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def case_name_from_path(path: Path) -> str:
    rel = path.relative_to(WORKSPACE_ROOT)
    parts = list(rel.parts)
    useful = [part for part in parts if part not in {"runs", "research_direction", "feature_scores.json", "stage3_feature_matching", "stage3_feature"}]
    name = "_".join(useful[-3:])
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or "case"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an expanded Matcher learning casebook manifest from Stage3 feature outputs.")
    parser.add_argument("--search-root", default="runs/research_direction")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_learning_casebook/expanded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_root = resolve_path(args.search_root)
    out_dir = resolve_path(args.out_dir)
    paths = sorted(search_root.rglob("feature_scores.json")) if search_root.exists() else []
    cases = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        proposals = payload.get("metrics") or payload.get("proposals") or payload.get("feature_metrics") or []
        if not proposals:
            continue
        cases.append({
            "case_name": case_name_from_path(path),
            "stage3_feature_json": str(path.relative_to(WORKSPACE_ROOT)),
            "proposal_count": int(payload.get("proposal_count", len(proposals))),
            "retained_count": int(payload.get("retained_count", len(payload.get("retained_group_ids", [])))),
            "k_pred": int(payload.get("k_pred", payload.get("retained_count", 0) or 0)),
        })
    manifest = {
        "search_root": str(search_root.relative_to(WORKSPACE_ROOT)) if search_root.is_relative_to(WORKSPACE_ROOT) else str(search_root),
        "case_count": len(cases),
        "cases": cases,
        "label_mode": "pseudo_retained_vs_rejected unless labels_json is added later",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expanded_cases.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Expanded Matcher Casebook Manifest",
        "",
        f"- Search root: `{manifest['search_root']}`",
        f"- Case count: `{manifest['case_count']}`",
        "",
        "| Case | Proposals | Retained | K_pred |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(f"| {case['case_name']} | {case['proposal_count']} | {case['retained_count']} | {case['k_pred']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'expanded_cases.json'}")


if __name__ == "__main__":
    main()
