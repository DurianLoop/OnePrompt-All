from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from adapromptset.evaluation.metrics import dice_score, iou_score
from adapromptset.matching.runtime import checkpoint_record, configure_external_paths
from adapromptset.scripts.run_real_matcher_smoke import resolve_episode_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_SIZE = 518
PRIMARY_FAMILY = "P3_component_grouped_centroids"
PROMPT_FAMILIES = (
    "P1_largest_component_centroid",
    "P2_fixed_multi_centroid_one_group",
    "P3_component_grouped_centroids",
    "P4_union_mask_bounding_box",
    "P5_positive_plus_controlled_negative",
)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260719
RANDOM_CONTROL_SEED = 20260719
MIN_MEAN_MARGIN = 0.05
MIN_WIN_RATE = 0.70
MAX_CATASTROPHIC_RATE = 0.10
CATASTROPHIC_DICE_MIN = 0.20
NEAR_IMAGE_WIDE_RATIO = 0.95
MIN_PRED_AREA_RATIO = 0.0005


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def load_mask(path: Path) -> list[list[bool]]:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).tolist()


def _mask_array(mask: list[list[bool]]) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _mask_area(mask: list[list[bool]]) -> int:
    return int(_mask_array(mask).sum())


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _component_records(mask: list[list[bool]]) -> list[dict[str, Any]]:
    arr = _mask_array(mask)
    height, width = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    records: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if not arr[y, x] or seen[y, x]:
                continue
            pixels: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                pixels.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            centroid = (
                int(round(sum(px for px, _ in pixels) / len(pixels))),
                int(round(sum(py for _, py in pixels) / len(pixels))),
            )
            records.append(
                {
                    "area": len(pixels),
                    "centroid": centroid,
                    "bbox": _bbox_from_points(pixels),
                    "pixels": pixels,
                }
            )
    return sorted(records, key=lambda row: (-row["area"], row["centroid"][1], row["centroid"][0]))


def _boundary_length(mask: list[list[bool]]) -> int:
    arr = _mask_array(mask)
    height, width = arr.shape
    boundary = 0
    for y in range(height):
        for x in range(width):
            if not arr[y, x]:
                continue
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < width and 0 <= ny < height) or not arr[ny, nx]:
                    boundary += 1
                    break
    return boundary


def _near_boundary_negative(mask: list[list[bool]]) -> tuple[int, int]:
    arr = _mask_array(mask)
    components = _component_records(mask)
    if not components:
        return (0, 0)
    x0, y0, x1, y1 = components[0]["bbox"]
    candidates: list[tuple[int, int]] = []
    for radius in (6, 10, 14, 20, 28, 36):
        for x in range(max(0, x0 - radius), min(IMAGE_SIZE - 1, x1 + radius) + 1):
            for y in (max(0, y0 - radius), min(IMAGE_SIZE - 1, y1 + radius)):
                if not arr[y, x]:
                    candidates.append((x, y))
        for y in range(max(0, y0 - radius), min(IMAGE_SIZE - 1, y1 + radius) + 1):
            for x in (max(0, x0 - radius), min(IMAGE_SIZE - 1, x1 + radius)):
                if not arr[y, x]:
                    candidates.append((x, y))
        if candidates:
            return sorted(set(candidates), key=lambda point: (point[1], point[0]))[0]
    ys, xs = np.where(~arr)
    if len(xs) == 0:
        return (0, 0)
    return int(xs[0]), int(ys[0])


def _random_points(
    *,
    seed_text: str,
    count: int,
    labels: list[int],
) -> list[tuple[int, int]]:
    rng = random.Random(seed_text)
    return [(rng.randrange(IMAGE_SIZE), rng.randrange(IMAGE_SIZE)) for _ in range(count)]


def _random_box(*, seed_text: str, width: int, height: int) -> tuple[int, int, int, int]:
    rng = random.Random(seed_text)
    width = max(1, min(width, IMAGE_SIZE))
    height = max(1, min(height, IMAGE_SIZE))
    x0 = rng.randrange(0, IMAGE_SIZE - width + 1)
    y0 = rng.randrange(0, IMAGE_SIZE - height + 1)
    return x0, y0, x0 + width - 1, y0 + height - 1


def _pad_points(points: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if not points:
        points = [(IMAGE_SIZE // 2, IMAGE_SIZE // 2)]
    while len(points) < count:
        points.append(points[-1])
    return points[:count]


def _prompt_spec_from_mask(mask: list[list[bool]], family: str) -> dict[str, Any]:
    components = _component_records(mask)
    if not components:
        raise ValueError("mask has no foreground components")
    top3 = components[:3]
    if family == "P1_largest_component_centroid":
        return {"kind": "points", "groups": [[components[0]["centroid"]]], "labels": [[1]]}
    if family == "P2_fixed_multi_centroid_one_group":
        return {"kind": "points", "groups": [[row["centroid"] for row in top3]], "labels": [[1] * len(top3)]}
    if family == "P3_component_grouped_centroids":
        return {"kind": "points", "groups": [[row["centroid"]] for row in top3], "labels": [[1] for _ in top3]}
    if family == "P4_union_mask_bounding_box":
        points = [(x, y) for row in components for x, y in row["pixels"]]
        return {"kind": "box", "box": _bbox_from_points(points)}
    if family == "P5_positive_plus_controlled_negative":
        return {
            "kind": "points",
            "groups": [[components[0]["centroid"], _near_boundary_negative(mask)]],
            "labels": [[1, 0]],
        }
    raise ValueError(f"unsupported prompt family: {family}")


def _random_control_spec(privileged: dict[str, Any], *, query_case_id: str, family: str) -> dict[str, Any]:
    if privileged["kind"] == "box":
        x0, y0, x1, y1 = privileged["box"]
        return {
            "kind": "box",
            "box": _random_box(
                seed_text=f"{RANDOM_CONTROL_SEED}:box:{query_case_id}:{family}",
                width=x1 - x0 + 1,
                height=y1 - y0 + 1,
            ),
        }
    groups: list[list[tuple[int, int]]] = []
    for group_index, labels in enumerate(privileged["labels"]):
        groups.append(
            _random_points(
                seed_text=f"{RANDOM_CONTROL_SEED}:points:{query_case_id}:{family}:{group_index}",
                count=len(labels),
                labels=labels,
            )
        )
    return {"kind": "points", "groups": groups, "labels": privileged["labels"]}


def _mismatched_control_spec(reference_mask: list[list[bool]], privileged: dict[str, Any], family: str) -> dict[str, Any]:
    reference = _prompt_spec_from_mask(reference_mask, family)
    if privileged["kind"] == "box":
        return reference
    groups: list[list[tuple[int, int]]] = []
    for target_group, reference_group in zip(privileged["labels"], reference["groups"], strict=False):
        groups.append(_pad_points(list(reference_group), len(target_group)))
    while len(groups) < len(privileged["labels"]):
        groups.append(_pad_points(list(reference["groups"][-1]), len(privileged["labels"][len(groups)])))
    return {"kind": "points", "groups": groups[: len(privileged["labels"])], "labels": privileged["labels"]}


def _candidate_anchor_spec(
    candidate_rows: list[dict[str, str]],
    *,
    query_case_id: str,
    family: str,
    privileged: dict[str, Any],
) -> dict[str, Any]:
    if privileged["kind"] == "box":
        return _random_control_spec(privileged, query_case_id=query_case_id, family=f"C3:{family}")
    rows = [
        row
        for row in candidate_rows
        if row["episode_id"].startswith(f"glas_phase2_val_{query_case_id}_")
        and row["positive_limit"] == "20"
    ]
    positives = [(int(row["x"]), int(row["y"])) for row in rows if row["kind"] == "positive"]
    negatives = [(int(row["x"]), int(row["y"])) for row in rows if row["kind"] == "negative"]
    groups: list[list[tuple[int, int]]] = []
    pos_index = 0
    neg_index = 0
    for labels in privileged["labels"]:
        group: list[tuple[int, int]] = []
        for label in labels:
            if label == 1:
                group.append(_pad_points(positives, pos_index + 1)[pos_index])
                pos_index += 1
            else:
                group.append(_pad_points(negatives, neg_index + 1)[neg_index])
                neg_index += 1
        groups.append(group)
    return {"kind": "points", "groups": groups, "labels": privileged["labels"]}


def _validate_spec(spec: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if spec["kind"] == "box":
        x0, y0, x1, y1 = spec["box"]
        if not (0 <= x0 <= x1 < IMAGE_SIZE and 0 <= y0 <= y1 < IMAGE_SIZE):
            issues.append(f"invalid box: {spec['box']}")
        return issues
    for group, labels in zip(spec["groups"], spec["labels"], strict=True):
        if len(group) != len(labels):
            issues.append("point/label count mismatch")
        for x, y in group:
            if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
                issues.append(f"invalid point: {(x, y)}")
        for label in labels:
            if label not in {0, 1}:
                issues.append(f"invalid label: {label}")
    return issues


def _union_masks(masks: list[list[list[bool]]]) -> list[list[bool]]:
    if not masks:
        return [[False for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
    output = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for mask in masks:
        output |= _mask_array(mask)
    return output.tolist()


def _predict_spec(predictor: Any, image_array: np.ndarray, spec: dict[str, Any]) -> tuple[list[list[bool]], float, int]:
    predictor.set_image(image_array)
    masks_out: list[list[list[bool]]] = []
    scores_out: list[float] = []
    if spec["kind"] == "box":
        masks, scores, _ = predictor.predict(
            box=np.asarray(spec["box"], dtype=np.float32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool).tolist(), float(scores[best]), 1
    for points, labels in zip(spec["groups"], spec["labels"], strict=True):
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        masks_out.append(masks[best].astype(bool).tolist())
        scores_out.append(float(scores[best]))
    return _union_masks(masks_out), mean(scores_out) if scores_out else 0.0, len(spec["groups"])


def _mask_area_ratio(mask: list[list[bool]]) -> float:
    return _mask_area(mask) / float(IMAGE_SIZE * IMAGE_SIZE)


def _is_catastrophic(mask: list[list[bool]], target: list[list[bool]], dice: float, failed: bool, invalid: bool) -> bool:
    area_ratio = _mask_area_ratio(mask)
    return failed or invalid or area_ratio <= MIN_PRED_AREA_RATIO or area_ratio >= NEAR_IMAGE_WIDE_RATIO or dice < CATASTROPHIC_DICE_MIN


def _load_sam_predictor(checkpoint: Path, workspace_root: Path, device: str):
    configure_external_paths(workspace_root)
    from segment_anything import SamPredictor, sam_model_registry

    name = checkpoint.name.lower()
    architecture = "vit_l" if "vit_l" in name else "vit_b"
    model = sam_model_registry[architecture](checkpoint=str(checkpoint))
    model.eval().to(device)
    return SamPredictor(model)


def _unique_query_rows(episodes: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in episodes:
        query = row["query_case_id"]
        if query in seen:
            continue
        seen.add(query)
        rows.append(row)
    return rows


def _assert_no_test_access(rows: list[dict[str, str]]) -> None:
    for row in rows:
        joined = " ".join(str(value) for value in row.values()).lower()
        if "testa" in joined or "testb" in joined:
            raise ValueError(f"test split path detected in confirmation manifest row: {row.get('episode_id')}")
        if row.get("query_source_split") != "train":
            raise ValueError(f"non-train query source split detected: {row.get('episode_id')}")


def _morphology(mask: list[list[bool]], grade: str) -> dict[str, Any]:
    components = _component_records(mask)
    areas = [row["area"] for row in components]
    total_area = sum(areas)
    largest = max(areas) if areas else 0
    variance = mean((area - mean(areas)) ** 2 for area in areas) if areas else 0.0
    boundary = _boundary_length(mask)
    centroids = [row["centroid"] for row in components]
    min_dist = 0.0
    if len(centroids) > 1:
        min_dist = min(
            math.hypot(ax - bx, ay - by)
            for idx, (ax, ay) in enumerate(centroids)
            for bx, by in centroids[idx + 1 :]
        )
    return {
        "query_grade": grade,
        "component_count": len(components),
        "foreground_area": total_area,
        "foreground_area_ratio": total_area / float(IMAGE_SIZE * IMAGE_SIZE),
        "largest_component_area": largest,
        "largest_component_area_ratio": largest / total_area if total_area else 0.0,
        "component_size_variance": variance,
        "boundary_irregularity_proxy": boundary / math.sqrt(total_area) if total_area else 0.0,
        "touching_close_component_proxy": min_dist,
        "small_gland_dominant": len(areas) > 0 and median(areas) < 0.02 * IMAGE_SIZE * IMAGE_SIZE,
        "large_gland_dominant": largest > 0.35 * IMAGE_SIZE * IMAGE_SIZE,
    }


def _bootstrap(differences: list[float], *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(differences)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    samples: list[float] = []
    for _ in range(iterations):
        samples.append(mean(differences[rng.randrange(n)] for _ in range(n)))
    ordered = sorted(samples)
    low = ordered[int(0.025 * (iterations - 1))]
    high = ordered[int(0.975 * (iterations - 1))]
    return {"mean": mean(differences), "ci_low": low, "ci_high": high}


def _summarize_segmenter(
    segmenter: str,
    query_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    p3 = [row for row in query_rows if row["segmenter"] == segmenter and row["prompt_family"] == PRIMARY_FAMILY]
    differences = [row["privileged_dice"] - row["matched_control_dice"] for row in p3]
    win_rate = sum(1 for value in differences if value > 0.0) / len(differences)
    catastrophic_rate = sum(1 for row in p3 if row["privileged_catastrophic"]) / len(p3)
    bootstrap = _bootstrap(differences)
    coordinate_pass = all(row["coordinate_valid"] for row in p3)
    cost_pass = all(row["cost_matched"] for row in p3)
    checks = {
        "coordinate_validation": coordinate_pass,
        "cost_matching": cost_pass,
        "mean_paired_dice_margin": bootstrap["mean"] >= MIN_MEAN_MARGIN,
        "query_win_rate": win_rate >= MIN_WIN_RATE,
        "catastrophic_failure_rate": catastrophic_rate <= MAX_CATASTROPHIC_RATE,
        "bootstrap_direction": bootstrap["mean"] > 0.0,
        "no_hidden_oracle": True,
    }
    return {
        "segmenter": segmenter,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "query_count": len(p3),
        "p3_mean_privileged_dice": mean(row["privileged_dice"] for row in p3),
        "p3_mean_matched_control_dice": mean(row["matched_control_dice"] for row in p3),
        "p3_mean_margin": bootstrap["mean"],
        "p3_win_rate": win_rate,
        "p3_catastrophic_failure_rate": catastrophic_rate,
        "p3_bootstrap_ci_low": bootstrap["ci_low"],
        "p3_bootstrap_ci_high": bootstrap["ci_high"],
        "uncertainty_warning": bootstrap["ci_low"] <= 0.0,
        "prompt_family_count": len([row for row in family_rows if row["segmenter"] == segmenter]),
    }


def _final_state(segmenter_summaries: dict[str, dict[str, Any]]) -> str:
    samb_pass = segmenter_summaries["SAM-B"]["status"] == "PASS"
    saml_pass = segmenter_summaries["SAM-L"]["status"] == "PASS"
    if samb_pass and not saml_pass:
        return "PHASE2_GATE_D_PASS_SAMB"
    if saml_pass and not samb_pass:
        return "PHASE2_GATE_D_PASS_SAML"
    if samb_pass and saml_pass:
        saml_dice_gain = segmenter_summaries["SAM-L"]["p3_mean_privileged_dice"] - segmenter_summaries["SAM-B"]["p3_mean_privileged_dice"]
        cat_drop = segmenter_summaries["SAM-B"]["p3_catastrophic_failure_rate"] - segmenter_summaries["SAM-L"]["p3_catastrophic_failure_rate"]
        if saml_dice_gain >= 0.03 and cat_drop >= 0.05:
            return "PHASE2_GATE_D_PASS_BOTH_FREEZE_SAML"
        return "PHASE2_GATE_D_PASS_BOTH_FREEZE_SAMB"
    return "STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED"


def _write_figures(out_dir: Path, figure_dir: Path, query_rows: list[dict[str, Any]], morphology_rows: list[dict[str, Any]]) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    p3 = [row for row in query_rows if row["prompt_family"] == PRIMARY_FAMILY]
    artifacts: dict[str, str] = {}

    def save_current(name: str) -> None:
        path = figure_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        artifacts[name] = str(path)

    plt.figure(figsize=(4.8, 4.0))
    for segmenter in ("SAM-B", "SAM-L"):
        rows = [row for row in p3 if row["segmenter"] == segmenter]
        plt.scatter([row["matched_control_dice"] for row in rows], [row["privileged_dice"] for row in rows], label=segmenter, s=18)
    plt.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    plt.xlabel("Matched control Dice")
    plt.ylabel("P3 privileged Dice")
    plt.legend()
    save_current("paired_privileged_control_dice_scatter.png")

    plt.figure(figsize=(5.0, 3.4))
    for segmenter in ("SAM-B", "SAM-L"):
        rows = [row for row in p3 if row["segmenter"] == segmenter]
        plt.hist([row["privileged_margin"] for row in rows], bins=12, alpha=0.55, label=segmenter)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("P3 Dice margin")
    plt.ylabel("Query count")
    plt.legend()
    save_current("paired_dice_difference_distribution.png")

    plt.figure(figsize=(7.0, 3.8))
    query_ids = sorted({row["query_case_id"] for row in p3})
    for segmenter in ("SAM-B", "SAM-L"):
        values = [next(row["privileged_dice"] for row in p3 if row["segmenter"] == segmenter and row["query_case_id"] == query) for query in query_ids]
        plt.plot(range(len(query_ids)), values, marker="o", linewidth=1.0, label=segmenter)
    plt.xlabel("Query index")
    plt.ylabel("P3 privileged Dice")
    plt.legend()
    save_current("per_query_samb_saml_comparison.png")

    plt.figure(figsize=(6.0, 3.6))
    labels = list(PROMPT_FAMILIES)
    x = np.arange(len(labels))
    width = 0.35
    for idx, segmenter in enumerate(("SAM-B", "SAM-L")):
        values = [
            mean(row["privileged_dice"] for row in query_rows if row["segmenter"] == segmenter and row["prompt_family"] == family)
            for family in labels
        ]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=segmenter)
    plt.xticks(x, [label.split("_")[0] for label in labels])
    plt.ylabel("Mean privileged Dice")
    plt.legend()
    save_current("prompt_family_comparison.png")

    plt.figure(figsize=(5.0, 3.2))
    outcomes = []
    for segmenter in ("SAM-B", "SAM-L"):
        rows = [row for row in p3 if row["segmenter"] == segmenter]
        outcomes.extend((segmenter, "win" if row["privileged_margin"] > 0 else "loss" if row["privileged_margin"] < 0 else "tie") for row in rows)
    counts = Counter(outcomes)
    x = np.arange(2)
    bottoms = np.zeros(2)
    for outcome in ("win", "tie", "loss"):
        values = [counts[(segmenter, outcome)] for segmenter in ("SAM-B", "SAM-L")]
        plt.bar(x, values, bottom=bottoms, label=outcome)
        bottoms += values
    plt.xticks(x, ["SAM-B", "SAM-L"])
    plt.ylabel("Query count")
    plt.legend()
    save_current("win_tie_loss.png")

    plt.figure(figsize=(5.0, 3.2))
    values = [
        sum(1 for row in p3 if row["segmenter"] == segmenter and row["privileged_catastrophic"])
        for segmenter in ("SAM-B", "SAM-L")
    ]
    plt.bar(["SAM-B", "SAM-L"], values)
    plt.ylabel("Catastrophic P3 queries")
    save_current("catastrophic_failure_summary.png")

    morph_by_query = {row["query_case_id"]: row for row in morphology_rows}
    plt.figure(figsize=(5.0, 3.4))
    for segmenter in ("SAM-B", "SAM-L"):
        rows = [row for row in p3 if row["segmenter"] == segmenter]
        plt.scatter([morph_by_query[row["query_case_id"]]["component_count"] for row in rows], [row["privileged_dice"] for row in rows], label=segmenter, s=18)
    plt.xlabel("Component count")
    plt.ylabel("P3 privileged Dice")
    plt.legend()
    save_current("dice_vs_component_count.png")

    plt.figure(figsize=(5.0, 3.4))
    for segmenter in ("SAM-B", "SAM-L"):
        rows = [row for row in p3 if row["segmenter"] == segmenter]
        plt.scatter([morph_by_query[row["query_case_id"]]["foreground_area_ratio"] for row in rows], [row["privileged_dice"] for row in rows], label=segmenter, s=18)
    plt.xlabel("Foreground area ratio")
    plt.ylabel("P3 privileged Dice")
    plt.legend()
    save_current("dice_vs_foreground_area.png")

    plt.figure(figsize=(5.2, 3.4))
    labels = []
    values = []
    for segmenter in ("SAM-B", "SAM-L"):
        for grade in ("benign", "malignant"):
            rows = [row for row in p3 if row["segmenter"] == segmenter and morph_by_query[row["query_case_id"]]["query_grade"] == grade]
            labels.append(f"{segmenter}\n{grade}")
            values.append(mean(row["privileged_dice"] for row in rows) if rows else 0.0)
    plt.bar(labels, values)
    plt.ylabel("Mean P3 privileged Dice")
    save_current("benign_malignant_stratification.png")
    return artifacts


def _write_failure_panels(
    project_root: Path,
    figure_dir: Path,
    query_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> str:
    p3 = [row for row in query_rows if row["prompt_family"] == PRIMARY_FAMILY]
    by_query = {row["query_case_id"]: row for row in metadata_rows}
    selected: dict[str, str] = {}
    selected[min(p3, key=lambda row: row["privileged_margin"])["query_case_id"]] = "lowest P3-control margin"
    selected[min(p3, key=lambda row: row["privileged_margin"])["query_case_id"]] = "largest control advantage"
    catastrophic = [row for row in p3 if row["privileged_catastrophic"]]
    if catastrophic:
        selected[catastrophic[0]["query_case_id"]] = "catastrophic failure"
    selected[max(metadata_rows, key=lambda row: row["component_count"])["query_case_id"]] = "highest component count"
    selected[max(metadata_rows, key=lambda row: row["foreground_area"])["query_case_id"]] = "largest foreground area"
    malignant = [row for row in p3 if by_query[row["query_case_id"]]["query_grade"] == "malignant"]
    if malignant:
        selected[min(malignant, key=lambda row: row["privileged_dice"])["query_case_id"]] = "malignant lowest Dice"

    thumbs: list[Image.Image] = []
    for query_id, reason in list(selected.items())[:6]:
        meta = by_query[query_id]
        image = load_rgb(Path(meta["image_path"])).resize((180, 180), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 179, 32), fill=(255, 255, 255))
        draw.text((4, 3), query_id, fill=(0, 0, 0))
        draw.text((4, 17), reason[:28], fill=(0, 0, 0))
        thumbs.append(image)
    if not thumbs:
        thumbs.append(Image.new("RGB", (180, 180), "white"))
    panel = Image.new("RGB", (360, 180 * math.ceil(len(thumbs) / 2)), "white")
    for index, thumb in enumerate(thumbs):
        panel.paste(thumb, ((index % 2) * 180, (index // 2) * 180))
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "representative_failure_panels.png"
    panel.save(path)
    return str(path)


def _write_tables(
    project_root: Path,
    segmenter_summaries: dict[str, dict[str, Any]],
    family_rows: list[dict[str, Any]],
    morphology_summary: list[dict[str, Any]],
) -> None:
    table_dir = project_root / "paper_phase2" / "tables"
    section_dir = project_root / "paper_phase2" / "sections"
    table_dir.mkdir(parents=True, exist_ok=True)
    section_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GlaS Gate D confirmation uses query-level P3 paired margins against matched controls.}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Segmenter & Margin & Win rate & Cat. rate & Status \\",
        r"\hline",
    ]
    for name, summary in segmenter_summaries.items():
        lines.append(
            f"{name} & {summary['p3_mean_margin']:.3f} & {summary['p3_win_rate']:.3f} & {summary['p3_catastrophic_failure_rate']:.3f} & {summary['status']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (table_dir / "table_gate_d_confirmation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prompt-family viability diagnostics are secondary and do not redefine the primary Gate D criterion.}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Segmenter & Family & Mean Dice & Margin & Calls \\",
        r"\hline",
    ]
    for row in family_rows:
        lines.append(
            f"{row['segmenter']} & {row['prompt_family'].split('_')[0]} & {row['mean_privileged_dice']:.3f} & {row['mean_margin']:.3f} & {row['mean_calls']:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (table_dir / "table_prompt_family_viability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Morphology diagnostics explain failure modes but do not change the overall Gate D decision.}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Group & Queries & Mean P3 margin & Cat. rate \\",
        r"\hline",
    ]
    for row in morphology_summary:
        lines.append(f"{row['group']} & {row['query_count']} & {row['mean_p3_margin']:.3f} & {row['catastrophic_rate']:.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (table_dir / "table_morphology_failures.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    section = [
        r"\section{Gate D Confirmation}",
        r"The Phase 2D confirmation re-evaluates segmenter execution viability on all 30 GlaS strict-validation queries.",
        r"The previous six-case all-case-margin smoke is preserved as exploratory evidence, while this confirmation uses query-level paired margins for the final viability conclusion.",
        r"Privileged prompts are upper-bound/sanity prompts and are not deployable AdaPromptSet actions.",
        r"\input{paper_phase2/tables/table_gate_d_confirmation.tex}",
        r"\input{paper_phase2/tables/table_prompt_family_viability.tex}",
        r"\input{paper_phase2/tables/table_morphology_failures.tex}",
    ]
    (section_dir / "gate_d_confirmation.tex").write_text("\n".join(section) + "\n", encoding="utf-8")


def run_confirmation(
    *,
    project_root: Path = PROJECT_ROOT,
    out_dir: Path | None = None,
    figure_dir: Path | None = None,
    episodes_csv: Path | None = None,
    candidate_csv: Path | None = None,
    device: str = "cuda:0",
    dry_run: bool = False,
) -> int:
    workspace_root = project_root.parent
    out_dir = out_dir or project_root / "reports" / "phase2_glas" / "gate_d_confirmation"
    figure_dir = figure_dir or project_root / "paper_phase2" / "figures" / "gate_d_confirmation"
    episodes_csv = episodes_csv or project_root / "manifests" / "phase2" / "glas" / "episodes_val.csv"
    candidate_csv = candidate_csv or project_root / "reports" / "phase2_glas" / "candidates_redesign_v1" / "candidate_pilot_candidates.csv"
    episodes = _read_csv(episodes_csv)
    _assert_no_test_access(episodes)
    query_episodes = _unique_query_rows(episodes)
    if len(query_episodes) != 30:
        raise ValueError(f"expected 30 strict-validation queries, found {len(query_episodes)}")
    candidate_rows = _read_csv(candidate_csv)
    if dry_run:
        _write_json(out_dir / "gate_d_confirmation_summary.json", {"status": "DRY_RUN", "query_count": len(query_episodes)})
        return 0

    checkpoints = {
        "SAM-B": workspace_root / "assets" / "checkpoints" / "sam_vit_b.pth",
        "SAM-L": workspace_root / "assets" / "checkpoints" / "sam_vit_l_0b3195.pth",
    }
    episode_payloads: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for episode in query_episodes:
        paths = resolve_episode_paths(episode, workspace_root)
        query_image = load_rgb(paths["query_image"])
        query_mask = load_mask(paths["query_mask"])
        support_mask = load_mask(paths["support_mask"])
        metadata_rows.append(
            {
                "query_case_id": episode["query_case_id"],
                "query_grade": episode.get("query_grade", ""),
                "image_path": str(paths["query_image"]),
                "mask_path": str(paths["query_mask"]),
                **_morphology(query_mask, episode.get("query_grade", "")),
            }
        )
        image_array = np.asarray(query_image)
        prompt_specs: dict[str, dict[str, dict[str, Any]]] = {}
        for family in PROMPT_FAMILIES:
            privileged = _prompt_spec_from_mask(query_mask, family)
            prompt_specs[family] = {
                "privileged": privileged,
                "C1_random_coordinate": _random_control_spec(privileged, query_case_id=episode["query_case_id"], family=family),
                "C2_mismatched_reference": _mismatched_control_spec(support_mask, privileged, family),
                "C3_candidate_anchor": _candidate_anchor_spec(candidate_rows, query_case_id=episode["query_case_id"], family=family, privileged=privileged),
            }

        episode_payloads.append(
            {
                "episode": episode,
                "image_array": np.asarray(query_image),
                "query_mask": query_mask,
                "prompt_specs": prompt_specs,
            }
        )

    query_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    prompt_call_rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    segmenter_runtime: dict[str, float] = {}

    for segmenter, checkpoint in checkpoints.items():
        predictor = _load_sam_predictor(checkpoint, workspace_root, device)
        segmenter_start = time.perf_counter()
        for payload in episode_payloads:
            episode = payload["episode"]
            image_array = payload["image_array"]
            query_mask = payload["query_mask"]
            prompt_specs = payload["prompt_specs"]
            for family, variants in prompt_specs.items():
                variant_metrics: dict[str, dict[str, Any]] = {}
                for variant, spec in variants.items():
                    issues = _validate_spec(spec)
                    failed = False
                    try:
                        mask, sam_score, call_count = _predict_spec(predictor, image_array, spec)
                    except Exception:
                        failed = True
                        mask = [[False for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
                        sam_score = 0.0
                        call_count = len(spec.get("groups", [spec.get("box")]))
                    dice = dice_score(mask, query_mask)
                    iou = iou_score(mask, query_mask)
                    catastrophic = _is_catastrophic(mask, query_mask, dice, failed, bool(issues))
                    metric = {
                        "dice": dice,
                        "iou": iou,
                        "sam_score": sam_score,
                        "call_count": call_count,
                        "prompt_count": 4 if spec["kind"] == "box" else sum(len(group) for group in spec["groups"]),
                        "group_count": 1 if spec["kind"] == "box" else len(spec["groups"]),
                        "failed": failed,
                        "coordinate_valid": not issues,
                        "issues": ";".join(issues),
                        "catastrophic": catastrophic,
                        "mask_area_ratio": _mask_area_ratio(mask),
                    }
                    variant_metrics[variant] = metric
                    prompt_call_rows.append(
                        {
                            "segmenter": segmenter,
                            "query_case_id": episode["query_case_id"],
                            "prompt_family": family,
                            "variant": variant,
                            **metric,
                        }
                    )
                privileged = variant_metrics["privileged"]
                c1 = variant_metrics["C1_random_coordinate"]
                c2 = variant_metrics["C2_mismatched_reference"]
                matched_control = (c1["dice"] + c2["dice"]) / 2.0
                cost_matched = (
                    privileged["call_count"] == c1["call_count"] == c2["call_count"]
                    and privileged["group_count"] == c1["group_count"] == c2["group_count"]
                    and privileged["prompt_count"] == c1["prompt_count"] == c2["prompt_count"]
                )
                query_row = {
                    "segmenter": segmenter,
                    "query_case_id": episode["query_case_id"],
                    "query_grade": episode.get("query_grade", ""),
                    "prompt_family": family,
                    "privileged_dice": privileged["dice"],
                    "privileged_iou": privileged["iou"],
                    "matched_control_dice": matched_control,
                    "c1_random_dice": c1["dice"],
                    "c2_mismatched_dice": c2["dice"],
                    "c3_candidate_anchor_dice": variant_metrics["C3_candidate_anchor"]["dice"],
                    "privileged_margin": privileged["dice"] - matched_control,
                    "privileged_win": privileged["dice"] > matched_control,
                    "privileged_catastrophic": privileged["catastrophic"],
                    "coordinate_valid": privileged["coordinate_valid"] and c1["coordinate_valid"] and c2["coordinate_valid"],
                    "cost_matched": cost_matched,
                    "privileged_calls": privileged["call_count"],
                    "matched_control_calls": c1["call_count"],
                    "prompt_count": privileged["prompt_count"],
                    "group_count": privileged["group_count"],
                    "mask_area_bias": privileged["mask_area_ratio"] - _mask_area_ratio(query_mask),
                }
                query_rows.append(query_row)
                matched_rows.append(
                    {
                        "segmenter": segmenter,
                        "query_case_id": episode["query_case_id"],
                        "prompt_family": family,
                        "matched_control_dice": matched_control,
                        "c1_random_dice": c1["dice"],
                        "c2_mismatched_dice": c2["dice"],
                        "c3_candidate_anchor_dice": variant_metrics["C3_candidate_anchor"]["dice"],
                        "cost_matched": cost_matched,
                        }
                    )
        segmenter_runtime[segmenter] = time.perf_counter() - segmenter_start

    family_rows: list[dict[str, Any]] = []
    for segmenter in checkpoints:
        for family in PROMPT_FAMILIES:
            rows = [row for row in query_rows if row["segmenter"] == segmenter and row["prompt_family"] == family]
            family_rows.append(
                {
                    "segmenter": segmenter,
                    "prompt_family": family,
                    "query_count": len(rows),
                    "mean_privileged_dice": mean(row["privileged_dice"] for row in rows),
                    "median_privileged_dice": median(row["privileged_dice"] for row in rows),
                    "mean_privileged_iou": mean(row["privileged_iou"] for row in rows),
                    "mean_margin": mean(row["privileged_margin"] for row in rows),
                    "wins": sum(1 for row in rows if row["privileged_margin"] > 0),
                    "ties": sum(1 for row in rows if row["privileged_margin"] == 0),
                    "losses": sum(1 for row in rows if row["privileged_margin"] < 0),
                    "catastrophic_failures": sum(1 for row in rows if row["privileged_catastrophic"]),
                    "mean_calls": mean(row["privileged_calls"] for row in rows),
                    "mean_mask_area_bias": mean(row["mask_area_bias"] for row in rows),
                }
            )

    segmenter_summaries = {
        segmenter: _summarize_segmenter(segmenter, query_rows, family_rows)
        for segmenter in checkpoints
    }
    final_state = _final_state(segmenter_summaries)
    p3_rows = [row for row in query_rows if row["prompt_family"] == PRIMARY_FAMILY]
    morphology_by_query = {row["query_case_id"]: row for row in metadata_rows}
    morphology_rows: list[dict[str, Any]] = []
    for row in p3_rows:
        morphology_rows.append({**morphology_by_query[row["query_case_id"]], **row})

    morphology_summary: list[dict[str, Any]] = []
    for group_name, predicate in (
        ("benign", lambda row: row["query_grade"] == "benign"),
        ("malignant", lambda row: row["query_grade"] == "malignant"),
        ("high_component_count", lambda row: row["component_count"] >= median(meta["component_count"] for meta in metadata_rows)),
        ("large_foreground", lambda row: row["foreground_area_ratio"] >= median(meta["foreground_area_ratio"] for meta in metadata_rows)),
    ):
        group_rows = [row for row in morphology_rows if predicate(row)]
        if group_rows:
            morphology_summary.append(
                {
                    "group": group_name,
                    "query_count": len({row["query_case_id"] for row in group_rows}),
                    "mean_p3_margin": mean(row["privileged_margin"] for row in group_rows),
                    "mean_p3_dice": mean(row["privileged_dice"] for row in group_rows),
                    "catastrophic_rate": sum(1 for row in group_rows if row["privileged_catastrophic"]) / len(group_rows),
                }
            )

    win_rows = []
    for segmenter in checkpoints:
        rows = [row for row in p3_rows if row["segmenter"] == segmenter]
        win_rows.append(
            {
                "segmenter": segmenter,
                "wins": sum(1 for row in rows if row["privileged_margin"] > 0),
                "ties": sum(1 for row in rows if row["privileged_margin"] == 0),
                "losses": sum(1 for row in rows if row["privileged_margin"] < 0),
            }
        )
    catastrophic_rows = [row for row in p3_rows if row["privileged_catastrophic"]]
    coordinate_audit = {
        "status": "PASS" if all(row["coordinate_valid"] for row in query_rows) else "FAIL",
        "invalid_row_count": sum(1 for row in query_rows if not row["coordinate_valid"]),
        "testA_testB_access": False,
    }
    cost_rows = [
        {
            "segmenter": row["segmenter"],
            "query_case_id": row["query_case_id"],
            "prompt_family": row["prompt_family"],
            "cost_matched": row["cost_matched"],
            "privileged_calls": row["privileged_calls"],
            "matched_control_calls": row["matched_control_calls"],
            "prompt_count": row["prompt_count"],
            "group_count": row["group_count"],
        }
        for row in query_rows
    ]
    runtime_rows = [
        {
            "segmenter": segmenter,
            "elapsed_seconds": round(seconds, 3),
            "gpu_hours_estimate": round(seconds / 3600.0 if device.startswith("cuda") else 0.0, 6),
        }
        for segmenter, seconds in segmenter_runtime.items()
    ]
    total_elapsed = time.perf_counter() - start_all
    total_gpu_hours = total_elapsed / 3600.0 if device.startswith("cuda") else 0.0

    _write_csv(out_dir / "query_level_metrics.csv", query_rows)
    _write_csv(out_dir / "prompt_family_metrics.csv", family_rows)
    _write_csv(out_dir / "matched_control_metrics.csv", matched_rows)
    _write_csv(out_dir / "sam_b_metrics.csv", [row for row in query_rows if row["segmenter"] == "SAM-B"])
    _write_csv(out_dir / "sam_l_metrics.csv", [row for row in query_rows if row["segmenter"] == "SAM-L"])
    _write_csv(out_dir / "win_tie_loss.csv", win_rows)
    _write_csv(out_dir / "catastrophic_failures.csv", catastrophic_rows)
    _write_json(out_dir / "coordinate_audit.json", coordinate_audit)
    _write_csv(out_dir / "cost_audit.csv", cost_rows)
    _write_csv(out_dir / "runtime_audit.csv", runtime_rows)
    _write_csv(out_dir / "morphology_analysis.csv", morphology_rows)
    _write_csv(out_dir / "failure_case_registry.csv", catastrophic_rows[:])
    _write_json(out_dir / "bootstrap_summary.json", {name: summary for name, summary in segmenter_summaries.items()})
    _write_md(
        out_dir / "bootstrap_summary.md",
        [
            "# GlaS Gate D Confirmation Bootstrap",
            "",
            *[
                f"- {name}: margin `{summary['p3_mean_margin']:.6f}`, CI `[{summary['p3_bootstrap_ci_low']:.6f}, {summary['p3_bootstrap_ci_high']:.6f}]`, warning `{summary['uncertainty_warning']}`"
                for name, summary in segmenter_summaries.items()
            ],
        ],
    )
    _write_md(
        out_dir / "morphology_analysis.md",
        [
            "# GlaS Gate D Morphology Diagnostics",
            "",
            "Morphology groups explain viability patterns but do not redefine the primary Gate D pass/fail rule.",
            "",
            *[
                f"- {row['group']}: queries `{row['query_count']}`, mean P3 margin `{row['mean_p3_margin']:.6f}`, catastrophic rate `{row['catastrophic_rate']:.6f}`"
                for row in morphology_summary
            ],
        ],
    )
    figure_artifacts = _write_figures(out_dir, figure_dir, query_rows, metadata_rows)
    figure_artifacts["representative_failure_panels.png"] = _write_failure_panels(project_root, figure_dir, query_rows, metadata_rows)
    _write_tables(project_root, segmenter_summaries, family_rows, morphology_summary)

    summary = {
        "schema_version": "adapromptset.phase2d.gate_d_confirmation.summary.v1",
        "status": final_state,
        "query_count": len(query_episodes),
        "testA_testB_access": False,
        "previous_gate_d_preserved_as": "EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE",
        "segmenter_summaries": segmenter_summaries,
        "final_gate_d_state": final_state,
        "gate_e_open": final_state.startswith("PHASE2_GATE_D_PASS"),
        "device": device,
        "elapsed_seconds": round(total_elapsed, 3),
        "gpu_hours_estimate": round(total_gpu_hours, 6),
        "segmenter_checkpoints": {name: checkpoint_record(path, role="segmenter") for name, path in checkpoints.items()},
        "artifacts": {
            "query_level_metrics": _relative(project_root, out_dir / "query_level_metrics.csv"),
            "prompt_family_metrics": _relative(project_root, out_dir / "prompt_family_metrics.csv"),
            "matched_control_metrics": _relative(project_root, out_dir / "matched_control_metrics.csv"),
            "bootstrap_summary": _relative(project_root, out_dir / "bootstrap_summary.json"),
            "morphology_analysis": _relative(project_root, out_dir / "morphology_analysis.csv"),
            "figures": {name: _relative(project_root, Path(path)) for name, path in figure_artifacts.items()},
        },
        "deviations": [],
        "claim_boundary": "Gate D confirmation evaluates privileged segmenter viability only, not automatic prompting or adaptive policy.",
    }
    _write_json(out_dir / "gate_d_confirmation_summary.json", summary)
    _write_md(
        out_dir / "gate_d_confirmation_summary.md",
        [
            "# GlaS Gate D Confirmation Summary",
            "",
            f"Status: `{final_state}`",
            "",
            "| Segmenter | P3 margin | Win rate | Catastrophic rate | Status |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| {name} | `{summary_row['p3_mean_margin']:.6f}` | `{summary_row['p3_win_rate']:.6f}` | `{summary_row['p3_catastrophic_failure_rate']:.6f}` | `{summary_row['status']}` |"
                for name, summary_row in segmenter_summaries.items()
            ],
            "",
            "GlaS testA/testB were not accessed.",
        ],
    )
    _write_csv(
        out_dir / "compute_ledger.csv",
        [
            {
                "phase": "phase2d_gate_d_confirmation",
                "command": "scripts/run_glas_gate_d_confirmation.py",
                "gpu_hours": f"{total_gpu_hours:.6f}",
                "notes": "SAM-B and SAM-L 30-query Gate D confirmation.",
            }
        ],
    )
    _write_json(
        out_dir / "run_metadata.json",
        {
            "schema_version": "adapromptset.phase2d.run_metadata.v1",
            "episodes_csv": str(episodes_csv),
            "candidate_csv": str(candidate_csv),
            "episodes_csv_sha256": _sha256(episodes_csv),
            "candidate_csv_sha256": _sha256(candidate_csv),
            "device": device,
            "query_count": len(query_episodes),
            "testA_testB_access": False,
        },
    )
    return 0 if final_state.startswith("PHASE2_GATE_D_PASS") else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2D GlaS Gate D 30-query confirmation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--episodes-csv", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_confirmation(
            project_root=args.project_root,
            out_dir=args.out_dir,
            figure_dir=args.figure_dir,
            episodes_csv=args.episodes_csv,
            candidate_csv=args.candidate_csv,
            device=args.device,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
