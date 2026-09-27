"""Apply one frozen MoNuSeg gate to PanNuke raw N0/N5 episodes.

Run each fold only with the sklearn version recorded in its joblib artifact.
This script never uses the recorded Dice values as inputs to a decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RAW = ROOT / "runs" / "pannuke_zero_shot_v1" / "per_episode.csv"
POLICY_ROOT = WORKSPACE / "runs" / "research_direction" / "nested_cv5_negative_budget"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delta_features(raw: dict[str, str], anchor: str, alternative: str, post_features: list[str]) -> list[float]:
    anchor_key = anchor.replace("neg", "n", 1)
    alternative_key = alternative.replace("neg", "n", 1)
    values: list[float] = []
    for name in post_features:
        if name == "negative_count":
            value = float(alternative.removeprefix("neg")) - float(anchor.removeprefix("neg"))
        elif name == "sam_score":
            value = float(raw[f"{alternative_key}_sam_score"]) - float(raw[f"{anchor_key}_sam_score"])
        elif name == "pred_area_pct":
            value = float(raw[f"{alternative_key}_area_pct"]) - float(raw[f"{anchor_key}_area_pct"])
        else:
            # Matcher and positive-prompt settings are intentionally identical for N0 and N5.
            value = 0.0
        values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one version-compatible frozen gate to raw PanNuke episodes.")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "pannuke_zero_shot_v1" / "gate_votes"))
    args = parser.parse_args()

    raw_rows = read_csv(RAW)
    if len(raw_rows) != 2540 * 5:
        raise RuntimeError(f"Raw PanNuke cohort is incomplete: expected 12700 episodes, got {len(raw_rows)}")
    policy_dir = POLICY_ROOT / f"fold_{args.fold}" / "policy"
    config = json.loads((policy_dir / "model.json").read_text(encoding="utf-8"))
    anchor = str(config["anchor_selected_on_validation"])
    if anchor not in {"neg0", "neg5"} or int(config["candidate_budget"]) != 2:
        raise RuntimeError("This raw N0/N5 interface supports only a two-action N0/N5 gate.")
    alternative = "neg5" if anchor == "neg0" else "neg0"
    post_features = [str(name) for name in config["post_features"]]
    model = joblib.load(policy_dir / "negative_post_sam.joblib")
    expected_features = len(post_features)
    if int(getattr(model, "n_features_in_", expected_features)) != expected_features:
        raise RuntimeError("Frozen post-SAM model feature count disagrees with model.json")
    threshold = float(config["selection"]["gain_threshold"])

    result: list[dict[str, object]] = []
    for raw in raw_rows:
        gain = float(model.predict(np.asarray([delta_features(raw, anchor, alternative, post_features)], dtype=np.float64))[0])
        action = alternative if gain > threshold else anchor
        result.append({
            "episode_id": raw["episode_id"], "query_case_id": raw["query_case_id"], "support_case_id": raw["support_case_id"],
            "fold": args.fold, "sklearn_version": sklearn.__version__, "anchor_action": anchor,
            "alternative_action": alternative, "gain_threshold": threshold, "predicted_gain": gain,
            "selected_action": action,
        })
    out_dir = Path(args.out_dir)
    write_csv(out_dir / f"fold_{args.fold}.csv", result)
    metadata = {
        "fold": args.fold, "sklearn_version": sklearn.__version__, "policy_dir": str(policy_dir),
        "model_json_sha256": sha256(policy_dir / "model.json"),
        "negative_post_sam_joblib_sha256": sha256(policy_dir / "negative_post_sam.joblib"),
        "raw_episode_count": len(result), "decision_inputs": ["N0/N5 SAM score", "N0/N5 predicted area", "fixed action delta"],
        "forbidden_inputs": ["Dice", "ground-truth mask"],
    }
    (out_dir / f"fold_{args.fold}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
