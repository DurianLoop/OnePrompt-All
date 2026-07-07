from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def prompt_type(prompt: str) -> str:
    lower = prompt.lower().strip()
    action_words = {"execute", "run", "jump", "walk", "hold", "eat", "play"}
    medical_words = {"lesion", "tumor", "polyp", "nucleus", "cup", "disc", "vessel", "cell"}
    if lower in action_words:
        return "action"
    if any(word in lower for word in medical_words):
        return "medical_term"
    if len(lower.split()) >= 2:
        return "attribute_or_phrase"
    return "simple_noun"


def normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        records = payload["results"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                row = {"case": key}
                row.update(value)
                records.append(row)
    else:
        records = []
    normalized: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        prompt = str(record.get("prompt") or record.get("text_prompt") or record.get("query") or record.get("case") or f"case_{idx}")
        instances = int(record.get("instances", record.get("instance_count", record.get("num_instances", 0))) or 0)
        score = record.get("score", record.get("best_score"))
        area = record.get("union_area", record.get("area_pct", record.get("mask_area_pct")))
        normalized.append({
            "case": str(record.get("case", f"case_{idx}")),
            "prompt": prompt,
            "instances": instances,
            "score": None if score is None else float(score),
            "area": None if area is None else float(area),
            "source": "teacher_minimal_validation",
        })
    return normalized


def fallback_decision(row: dict[str, Any]) -> tuple[bool, str]:
    instances = int(row["instances"])
    ptype = str(row["prompt_type"])
    score = row.get("score")
    area = row.get("area")
    reasons: list[str] = []
    if instances == 0:
        reasons.append("instances == 0")
    if ptype in {"action", "attribute_or_phrase", "medical_term"}:
        reasons.append(f"risky_prompt_type:{ptype}")
    if instances > 5:
        reasons.append("too_many_instances")
    if score is not None and float(score) < 0.5:
        reasons.append("low_score")
    if area is not None and (float(area) < 0.1 or float(area) > 75.0):
        reasons.append("abnormal_area")
    return bool(reasons), ";".join(reasons) if reasons else "text_prompt_ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SAM3 text failure taxonomy and fallback trigger table.")
    parser.add_argument("--results-json", default="runs/teacher_minimal_validation/results.json")
    parser.add_argument("--out-dir", default="runs/research_direction/p0_p3_closure/p1_text_taxonomy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_json)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        rows = normalize_records(payload)
    else:
        rows = []
    extra_prompts = ["white girl", "execute", "person", "elephant", "girl", "lesion", "polyp"]
    seen = {row["prompt"].lower() for row in rows}
    for prompt in extra_prompts:
        if prompt not in seen:
            rows.append({"case": f"manual_{prompt.replace(' ', '_')}", "prompt": prompt, "instances": 0, "score": None, "area": None, "source": "manual_prior"})
    for row in rows:
        row["prompt_type"] = prompt_type(row["prompt"])
        fallback_required, reason = fallback_decision(row)
        row["fallback_required"] = fallback_required
        row["fallback_reason"] = reason
    with (out_dir / "taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case", "prompt", "prompt_type", "instances", "score", "area", "fallback_required", "fallback_reason", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(results_path.relative_to(WORKSPACE_ROOT)) if results_path.exists() else str(results_path),
        "record_count": len(rows),
        "fallback_required_count": sum(1 for row in rows if row["fallback_required"]),
        "by_prompt_type": {},
        "rows": rows,
    }
    for row in rows:
        ptype = row["prompt_type"]
        summary["by_prompt_type"].setdefault(ptype, {"count": 0, "fallback_required": 0})
        summary["by_prompt_type"][ptype]["count"] += 1
        summary["by_prompt_type"][ptype]["fallback_required"] += int(bool(row["fallback_required"]))
    (out_dir / "taxonomy.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# SAM3 Text Failure Taxonomy Summary",
            "",
            f"- Records: `{summary['record_count']}`",
            f"- Fallback required: `{summary['fallback_required_count']}`",
            f"- By prompt type: `{summary['by_prompt_type']}`",
            "",
            "## Artifacts",
            "",
            "- `taxonomy.csv`",
            "- `taxonomy.json`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Records: {summary['record_count']}")
    print(f"Fallback required: {summary['fallback_required_count']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
