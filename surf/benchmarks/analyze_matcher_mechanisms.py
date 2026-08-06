from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib_matcher_mechanisms"))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2

    if hasattr(cv2, "connectedComponents"):
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    else:
        count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32), int(count - 1)


def point_target_metrics(mask: np.ndarray, points: Iterable[tuple[float, float]]) -> dict[str, float | int]:
    coordinates = list(points)
    labels, component_count = connected_components(mask)
    hit_count = 0
    hit_components: set[int] = set()
    height, width = mask.shape
    for x, y in coordinates:
        px = min(width - 1, max(0, int(round(x))))
        py = min(height - 1, max(0, int(round(y))))
        if mask[py, px]:
            hit_count += 1
            label = int(labels[py, px])
            if label > 0:
                hit_components.add(label)
    count = len(coordinates)
    return {
        "point_count": count,
        "point_hit_count": hit_count,
        "point_precision": hit_count / count if count else 0.0,
        "hit_component_count": len(hit_components),
        "gt_component_count": component_count,
        "component_coverage": len(hit_components) / component_count if component_count else 1.0,
    }


def patch_target_labels(mask: np.ndarray, score_shape: tuple[int, int]) -> np.ndarray:
    score_h, score_w = score_shape
    y_edges = np.linspace(0, mask.shape[0], score_h + 1, dtype=int)
    x_edges = np.linspace(0, mask.shape[1], score_w + 1, dtype=int)
    labels = np.zeros((score_h, score_w), dtype=bool)
    for py in range(score_h):
        for px in range(score_w):
            labels[py, px] = bool(mask[y_edges[py] : y_edges[py + 1], x_edges[px] : x_edges[px + 1]].any())
    return labels


def average_tied_ranks(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def rank_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    x = average_tied_ranks(left)
    y = average_tied_ranks(right)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Rank correlation requires equally sized sequences with at least two values.")
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def normalized_entropy(score_map: np.ndarray) -> float:
    values = np.asarray(score_map, dtype=np.float64).reshape(-1)
    shifted = values - float(values.max())
    probabilities = np.exp(shifted)
    probabilities /= float(probabilities.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / math.log(len(values)) if len(values) > 1 else 0.0


def local_maxima_count(score_map: np.ndarray) -> int:
    padded = np.pad(score_map, 1, mode="edge")
    count = 0
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            if float(score_map[y, x]) >= float(padded[y : y + 3, x : x + 3].max()):
                count += 1
    return count


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_path(row: dict[str, str]) -> Path:
    points = resolve(row["points_path"])
    return points.parent.parent / "t1_matching" / points.name.replace("_points.csv", "_score.npy")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    return np.any(array > 0, axis=2) if array.ndim == 3 else array > 0


def load_points(path: Path) -> list[tuple[float, float]]:
    return [(float(row["x"]), float(row["y"])) for row in read_csv(path)]


def ranking_metrics(score_map: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    labels = patch_target_labels(mask, score_map.shape).reshape(-1).astype(np.uint8)
    scores = score_map.reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(labels, scores)), float(roc_auc_score(labels, scores))


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if len(array) else float("nan")


def build_mechanism_rows(
    candidates: list[dict[str, str]],
    query_masks: dict[str, Path],
) -> list[dict[str, Any]]:
    mask_cache: dict[str, np.ndarray] = {}
    point_cache: dict[Path, dict[str, float | int]] = {}
    score_cache: dict[tuple[str, Path], dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        query_id = candidate["query_case_id"]
        if query_id not in mask_cache:
            mask_cache[query_id] = load_mask(query_masks[query_id])
        mask = mask_cache[query_id]
        points_path = resolve(candidate["points_path"])
        if points_path not in point_cache:
            point_cache[points_path] = point_target_metrics(mask, load_points(points_path))
        candidate_score_path = score_path(candidate)
        score_key = (query_id, candidate_score_path)
        if score_key not in score_cache:
            score = np.load(candidate_score_path)
            ap, auc = ranking_metrics(score, mask)
            score_cache[score_key] = {
                "patch_ap": ap,
                "patch_auc": auc,
                "score_entropy": normalized_entropy(score),
                "local_maxima_count": local_maxima_count(score),
            }
        rows.append({
            "episode_id": candidate.get("episode_id") or candidate.get("case_name") or query_id,
            "query_case_id": query_id,
            "round_name": candidate["round_name"],
            "method": candidate["method"],
            "matching": candidate["matching"],
            "layout": candidate["layout"],
            "k": int(float(candidate["k"])),
            "lambda_bg": float(candidate["lambda_bg"]),
            "dice": float(candidate["dice"]),
            "iou": float(candidate["iou"]),
            "mask_fg_precision": float(candidate["fg_precision"]),
            "mask_fg_recall": float(candidate["fg_recall"]),
            "object_recall": float(candidate["object_recall"]),
            "pred_area_pct": float(candidate["pred_area_pct"]),
            **point_cache[points_path],
            **score_cache[score_key],
        })
    return rows


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        first = items[0]
        summary.append({
            "method": method,
            "matching": first["matching"],
            "layout": first["layout"],
            "k": first["k"],
            "lambda_bg": first["lambda_bg"],
            "episode_count": len(items),
            "mean_point_precision": mean_or_nan(float(row["point_precision"]) for row in items),
            "mean_component_coverage": mean_or_nan(float(row["component_coverage"]) for row in items),
            "mean_patch_ap": mean_or_nan(float(row["patch_ap"]) for row in items),
            "mean_patch_auc": mean_or_nan(float(row["patch_auc"]) for row in items),
            "mean_dice": mean_or_nan(float(row["dice"]) for row in items),
            "mean_object_recall": mean_or_nan(float(row["object_recall"]) for row in items),
            "mean_pred_area_pct": mean_or_nan(float(row["pred_area_pct"]) for row in items),
        })
    return summary


def paired_method_comparison(
    rows: list[dict[str, Any]],
    baseline_action: str,
    treatment_action: str,
    metrics: list[str],
    bootstrap_samples: int = 10000,
    seed: int = 20260715,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        if method in {baseline_action, treatment_action}:
            grouped[str(row["episode_id"])][method] = row
    paired = [items for items in grouped.values() if baseline_action in items and treatment_action in items]
    if not paired:
        raise ValueError(f"No paired episodes for {baseline_action} and {treatment_action}.")
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "baseline_action": baseline_action,
        "treatment_action": treatment_action,
        "episode_count": len(paired),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metrics": {},
    }
    for metric in metrics:
        baseline = np.asarray([float(items[baseline_action][metric]) for items in paired], dtype=np.float64)
        treatment = np.asarray([float(items[treatment_action][metric]) for items in paired], dtype=np.float64)
        delta = treatment - baseline
        indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
        bootstrap_means = delta[indices].mean(axis=1)
        result["metrics"][metric] = {
            "baseline_mean": float(baseline.mean()),
            "treatment_mean": float(treatment.mean()),
            "mean_delta": float(delta.mean()),
            "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
            "wins": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "losses": int(np.sum(delta < -1e-12)),
        }
    return result


def build_oracle_rows(rows: list[dict[str, Any]], fixed_action: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    oracle_rows: list[dict[str, Any]] = []
    for episode_id, items in sorted(grouped.items()):
        oracle = max(items, key=lambda row: (float(row["dice"]), str(row["method"])))
        fixed = next((row for row in items if row["method"] == fixed_action), None)
        if fixed is None:
            raise ValueError(f"Missing fixed action {fixed_action} for {episode_id}.")
        oracle_rows.append({
            "episode_id": episode_id,
            "query_case_id": oracle["query_case_id"],
            "oracle_action": oracle["method"],
            "oracle_k": oracle["k"],
            "oracle_lambda_bg": oracle["lambda_bg"],
            "oracle_dice": oracle["dice"],
            "fixed_dice": fixed["dice"],
            "oracle_gap": float(oracle["dice"]) - float(fixed["dice"]),
            "gt_component_count": oracle["gt_component_count"],
            "score_entropy": oracle["score_entropy"],
            "local_maxima_count": oracle["local_maxima_count"],
        })
    summary = {
        "episode_count": len(oracle_rows),
        "mean_fixed_dice": mean_or_nan(float(row["fixed_dice"]) for row in oracle_rows),
        "mean_oracle_dice": mean_or_nan(float(row["oracle_dice"]) for row in oracle_rows),
        "mean_oracle_gap": mean_or_nan(float(row["oracle_gap"]) for row in oracle_rows),
        "oracle_k_unique": sorted({int(row["oracle_k"]) for row in oracle_rows}),
        "oracle_lambda_unique": sorted({float(row["oracle_lambda_bg"]) for row in oracle_rows}),
        "spearman_oracle_k_vs_gt_components": rank_correlation(
            [float(row["oracle_k"]) for row in oracle_rows],
            [float(row["gt_component_count"]) for row in oracle_rows],
        ),
        "spearman_oracle_k_vs_score_entropy": rank_correlation(
            [float(row["oracle_k"]) for row in oracle_rows],
            [float(row["score_entropy"]) for row in oracle_rows],
        ),
        "spearman_oracle_k_vs_local_maxima": rank_correlation(
            [float(row["oracle_k"]) for row in oracle_rows],
            [float(row["local_maxima_count"]) for row in oracle_rows],
        ),
    }
    return oracle_rows, summary


def save_plots(method_summary: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topk = sorted(
        [row for row in method_summary if row["layout"] == "topk" and row["matching"] == "fg_minus_bg"],
        key=lambda row: (float(row["lambda_bg"]), int(row["k"])),
    )
    lambdas = sorted({float(row["lambda_bg"]) for row in topk})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for lam in lambdas:
        selected = [row for row in topk if float(row["lambda_bg"]) == lam]
        axes[0].plot([row["k"] for row in selected], [row["mean_point_precision"] for row in selected], marker="o", label=f"lambda={lam:g}")
        axes[1].plot([row["k"] for row in selected], [row["mean_component_coverage"] for row in selected], marker="o", label=f"lambda={lam:g}")
    axes[0].set(xlabel="Prompt budget K", ylabel="Point precision", title="Matched-point precision")
    axes[1].set(xlabel="Prompt budget K", ylabel="GT component coverage", title="Matched-point component coverage")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(out_dir / "point_precision_component_coverage.png", dpi=220)
    plt.close(fig)

    actions: dict[str, int] = defaultdict(int)
    for row in oracle_rows:
        actions[str(row["oracle_action"])] += 1
    ordered = sorted(actions.items(), key=lambda item: (-item[1], item[0]))
    fig, axis = plt.subplots(figsize=(9, max(3.2, 0.34 * len(ordered))))
    axis.barh([item[0] for item in reversed(ordered)], [item[1] for item in reversed(ordered)], color="#2675a8")
    axis.set(xlabel="Number of query episodes", title="Per-episode oracle prompt actions")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_action_distribution.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(
        [row["gt_component_count"] for row in oracle_rows],
        [row["oracle_k"] for row in oracle_rows],
        s=55,
        color="#c73e1d",
        edgecolors="white",
        linewidths=0.8,
    )
    axis.set(xlabel="GT nucleus components (analysis only)", ylabel="Oracle K", title="Prompt budget varies across dense scenes")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_k_vs_gt_components.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-hoc matcher mechanism evidence without feeding GT to the policy.")
    parser.add_argument(
        "--candidates",
        default="runs/research_direction/prompt_policy_oracle/test_grid_dinov2l_samvitb_fixedr3/candidates.csv",
    )
    parser.add_argument(
        "--episodes",
        default="runs/research_direction/prompt_policy_splits/monuseg_strict_official/episodes_test.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/research_direction/mechanisms/dinov2l_samvitb_strict_test",
    )
    parser.add_argument("--fixed-action", default="contrast_topk_K40_lam0.75")
    parser.add_argument("--fg-only-action", default="fg_topk_K40")
    parser.add_argument("--background-action", default="contrast_topk_K40_lam0.75")
    args = parser.parse_args()

    candidates = read_csv(resolve(args.candidates))
    episodes = read_csv(resolve(args.episodes))
    query_masks = {row["query_case_id"]: resolve(row["query_mask_path"]) for row in episodes}
    missing = sorted({row["query_case_id"] for row in candidates} - query_masks.keys())
    if missing:
        raise ValueError(f"Missing query masks for: {missing}")
    mechanism_rows = build_mechanism_rows(candidates, query_masks)
    method_summary = summarize_methods(mechanism_rows)
    oracle_rows, oracle_summary = build_oracle_rows(mechanism_rows, args.fixed_action)
    background_comparison = paired_method_comparison(
        mechanism_rows,
        args.fg_only_action,
        args.background_action,
        ["point_precision", "component_coverage", "dice", "pred_area_pct"],
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "candidate_mechanisms.csv", mechanism_rows)
    write_csv(out_dir / "method_mechanism_summary.csv", method_summary)
    write_csv(out_dir / "oracle_actions.csv", oracle_rows)
    (out_dir / "oracle_summary.json").write_text(json.dumps(oracle_summary, indent=2), encoding="utf-8")
    (out_dir / "background_suppression_paired.json").write_text(
        json.dumps(background_comparison, indent=2),
        encoding="utf-8",
    )
    save_plots(method_summary, oracle_rows, out_dir)
    print(json.dumps(oracle_summary, indent=2))
    print(f"Wrote matcher mechanism evidence to {out_dir}")


if __name__ == "__main__":
    main()
