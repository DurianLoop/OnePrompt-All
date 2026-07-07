from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matcher_feedback_learning import (
    labeled_values,
    learn_scalar_threshold,
    score_stage3_proposal,
    select_threshold_by_budget,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def load_proposals(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("proposals", "feature_metrics", "metrics"):
        if isinstance(payload.get(key), list):
            return payload, payload[key]
    raise ValueError(f"No proposal list found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn lightweight Matcher thresholds from Stage 3 proposal evidence.")
    parser.add_argument("--stage3-feature-json", required=True)
    parser.add_argument("--labels-json", default="")
    parser.add_argument("--min-selected", type=int, default=1)
    parser.add_argument("--max-selected", type=int, default=6)
    parser.add_argument("--out-dir", default="runs/research_direction/feedback_learning_policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_json = resolve_path(args.stage3_feature_json)
    out_dir = resolve_path(args.out_dir)
    payload, proposals = load_proposals(feature_json)
    scored = []
    for proposal in proposals:
        row = dict(proposal)
        row["feedback_learning_score"] = score_stage3_proposal(row)
        scored.append(row)

    instance_values = [float(row.get("instance_similarity", 0.0)) for row in scored]
    learned_policy: dict[str, Any] | None = None
    if args.labels_json.strip():
        label_payload = json.loads(resolve_path(args.labels_json).read_text(encoding="utf-8"))
        values, labels, group_ids = labeled_values(scored, label_payload, "instance_similarity")
        score_values, score_labels, _ = labeled_values(scored, label_payload, "feedback_learning_score")
        if values and labels:
            learned_instance = learn_scalar_threshold(values, labels)
            learned_score = learn_scalar_threshold(score_values, score_labels)
            learned_policy = {
                "labels_json": str(resolve_path(args.labels_json).relative_to(WORKSPACE_ROOT)) if resolve_path(args.labels_json).is_relative_to(WORKSPACE_ROOT) else str(resolve_path(args.labels_json)),
                "labeled_group_ids": group_ids,
                "instance_similarity_policy": learned_instance.as_dict(),
                "feedback_score_policy": learned_score.as_dict(),
            }

    if learned_policy is not None:
        threshold = float(learned_policy["instance_similarity_policy"]["threshold"])
        threshold_source = "learned_from_labels"
    else:
        threshold = select_threshold_by_budget(
            instance_values,
            min_selected=args.min_selected,
            max_selected=args.max_selected,
            floor=0.0,
            ceiling=1.0,
        )
        threshold_source = "budget_adaptive"

    selected = [row for row in scored if float(row.get("instance_similarity", 0.0)) >= threshold]
    selected.sort(key=lambda row: (float(row.get("feedback_learning_score", 0.0)), float(row.get("instance_similarity", 0.0))), reverse=True)
    policy = {
        "stage3_feature_json": str(feature_json.relative_to(WORKSPACE_ROOT)) if feature_json.is_relative_to(WORKSPACE_ROOT) else str(feature_json),
        "proposal_count": len(scored),
        "threshold_source": threshold_source,
        "selected_group_ids": [int(row.get("group_id", -1)) for row in selected],
        "instance_similarity_threshold": round(float(threshold), 6),
        "instance_similarity_policy": learned_policy["instance_similarity_policy"] if learned_policy else None,
        "feedback_score_policy": learned_policy["feedback_score_policy"] if learned_policy else None,
        "scored_proposals": scored,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matcher_learning_policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Matcher Feedback Learning Policy",
        "",
        f"- Stage3 feature json: `{policy['stage3_feature_json']}`",
        f"- Proposal count: `{policy['proposal_count']}`",
        f"- Threshold source: `{policy['threshold_source']}`",
        f"- Instance similarity threshold: `{policy['instance_similarity_threshold']}`",
        f"- Selected group ids: `{policy['selected_group_ids']}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'matcher_learning_policy.json'}")


if __name__ == "__main__":
    main()
