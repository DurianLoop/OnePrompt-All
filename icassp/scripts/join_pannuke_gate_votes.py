"""Join five version-isolated gate decisions and report the PanNuke cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pannuke_zero_shot_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def interval(delta: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260809)
    means = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10_000)])
    return float(delta.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def win_tie_loss(candidate: np.ndarray, reference: np.ndarray) -> dict[str, int]:
    delta = candidate - reference
    return {"win": int((delta > 1e-12).sum()), "tie": int((np.abs(delta) <= 1e-12).sum()), "loss": int((delta < -1e-12).sum())}


def main() -> None:
    raw = read_csv(RUN / "per_episode.csv")
    if len(raw) != 12700:
        raise RuntimeError("Raw cohort incomplete")
    votes: dict[str, list[dict[str, str]]] = {}
    versions: dict[str, str] = {}
    for fold in range(5):
        rows = read_csv(RUN / "gate_votes" / f"fold_{fold}.csv")
        if len(rows) != len(raw):
            raise RuntimeError(f"fold {fold} vote file incomplete")
        for row in rows:
            if row["selected_action"] not in {"neg0", "neg5"}:
                raise RuntimeError("unsupported action in vote file")
            votes.setdefault(row["episode_id"], []).append(row)
            versions[str(fold)] = row["sklearn_version"]
    selected: list[dict[str, object]] = []
    for row in raw:
        ballot = votes.get(row["episode_id"], [])
        if len(ballot) != 5:
            raise RuntimeError(f"missing/duplicate ballot for {row['episode_id']}")
        n5_votes = sum(v["selected_action"] == "neg5" for v in ballot)
        action = "neg5" if n5_votes >= 3 else "neg0"
        selected.append({
            "episode_id": row["episode_id"], "query_case_id": row["query_case_id"], "support_case_id": row["support_case_id"],
            "n0_dice": float(row["n0_dice"]), "n5_dice": float(row["n5_dice"]),
            "selected_action": action, "n5_votes": n5_votes,
            "selected_dice": float(row[f"{action.replace('neg', 'n', 1)}_dice"]),
        })
    write_csv(RUN / "per_episode_majority_gate.csv", selected)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault(str(row["query_case_id"]), []).append(row)
    per_query = []
    for query, rows in sorted(grouped.items()):
        if len(rows) != 5:
            raise RuntimeError("unexpected support count")
        per_query.append({"query_case_id": query, **{key: float(np.mean([float(r[key]) for r in rows])) for key in ("n0_dice", "n5_dice", "selected_dice")}})
    write_csv(RUN / "per_query_majority_gate.csv", per_query)
    n0, n5, learned = (np.asarray([r[key] for r in per_query]) for key in ("n0_dice", "n5_dice", "selected_dice"))
    summary = {
        "status": "GATE_COMPLETE", "protocol": "icassp.pannuke_zero_shot_v1", "query_count": len(per_query), "supports_per_query": 5,
        "sklearn_versions_by_fold": versions,
        "means": {"n0": float(n0.mean()), "n5": float(n5.mean()), "five_fold_majority_gate": float(learned.mean())},
        "n5_vs_n0": dict(zip(("delta", "ci95_low", "ci95_high"), interval(n5-n0))),
        "gate_vs_n0": dict(zip(("delta", "ci95_low", "ci95_high"), interval(learned-n0))),
        "gate_vs_n5": dict(zip(("delta", "ci95_low", "ci95_high"), interval(learned-n5))),
        "win_tie_loss": {"n5_vs_n0": win_tie_loss(n5, n0), "gate_vs_n0": win_tie_loss(learned, n0), "gate_vs_n5": win_tie_loss(learned, n5)},
        "episode_n5_vote_histogram": {str(votes): int(sum(int(row["n5_votes"]) == votes for row in selected)) for votes in range(6)},
        "query_gt_policy": "raw masks opened after action predictions; gate consumes no Dice or ground truth",
    }
    (RUN / "summary_gate_complete.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
