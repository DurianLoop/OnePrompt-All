from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CandidatePoint:
    rank: int
    x: float
    y: float
    similarity_before: float
    similarity_after: float
    delta: float
    selected_by_threshold: bool
    cluster_id: int = -1


@dataclass
class PromptGroup:
    group_id: int
    cluster_id: int
    sample_size: int
    sample_index: int
    point_count: int
    mean_similarity_after: float
    max_similarity_after: float
    prompt_points: list[dict[str, float]]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return sorted(set(values))


def load_feedback_points(feedback_dir: Path) -> tuple[dict[str, Any], list[CandidatePoint], Image.Image]:
    feedback_json = feedback_dir / "feedback_points.json"
    if not feedback_json.exists():
        raise FileNotFoundError(feedback_json)
    payload = json.loads(feedback_json.read_text(encoding="utf-8"))
    points = [
        CandidatePoint(
            rank=int(row["rank"]),
            x=float(row["x_original"]),
            y=float(row["y_original"]),
            similarity_before=float(row["similarity_before"]),
            similarity_after=float(row["similarity_after"]),
            delta=float(row["delta"]),
            selected_by_threshold=bool(row["selected_by_threshold"]),
        )
        for row in payload["points"]
    ]
    points.sort(key=lambda point: point.rank)
    target = Image.open(resolve_path(payload["target_image"])).convert("RGB")
    return payload, points, target


def select_candidate_points(points: list[CandidatePoint], top_ratio: float, selected_only: bool) -> list[CandidatePoint]:
    if selected_only:
        selected = [point for point in points if point.selected_by_threshold]
        return selected if selected else points[:1]
    safe_ratio = min(max(float(top_ratio), 0.0), 1.0)
    keep = max(1, int(math.ceil(len(points) * safe_ratio)))
    return points[:keep]


def distance(a: CandidatePoint, b: CandidatePoint) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def region_query(points: list[CandidatePoint], index: int, eps: float) -> list[int]:
    return [other for other, point in enumerate(points) if distance(points[index], point) <= eps]


def dbscan(points: list[CandidatePoint], eps: float, min_samples: int) -> list[int]:
    labels = [-2 for _ in points]
    cluster_id = 0
    for index in range(len(points)):
        if labels[index] != -2:
            continue
        neighbors = region_query(points, index, eps)
        if len(neighbors) < min_samples:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        seeds = [neighbor for neighbor in neighbors if neighbor != index]
        cursor = 0
        while cursor < len(seeds):
            current = seeds[cursor]
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -2:
                cursor += 1
                continue
            labels[current] = cluster_id
            current_neighbors = region_query(points, current, eps)
            if len(current_neighbors) >= min_samples:
                for neighbor in current_neighbors:
                    if neighbor not in seeds:
                        seeds.append(neighbor)
            cursor += 1
        cluster_id += 1
    return labels


def cluster_points(points: list[CandidatePoint], cluster_method: str, eps: float, min_samples: int) -> list[CandidatePoint]:
    if cluster_method == "none":
        for point in points:
            point.cluster_id = 0
        return points
    if cluster_method != "dbscan":
        raise ValueError(f"Unsupported cluster method: {cluster_method}")
    labels = dbscan(points, eps=eps, min_samples=min_samples)
    for point, label in zip(points, labels):
        point.cluster_id = int(label)
    return points


def filter_cluster_points_by_anchor(points: list[CandidatePoint], max_anchor_distance: float) -> list[CandidatePoint]:
    if max_anchor_distance <= 0:
        return points
    filtered: list[CandidatePoint] = []
    cluster_ids = sorted({point.cluster_id for point in points if point.cluster_id >= 0})
    for cluster_id in cluster_ids:
        cluster = [point for point in points if point.cluster_id == cluster_id]
        cluster.sort(key=lambda point: point.rank)
        anchor = cluster[0]
        filtered.extend([point for point in cluster if distance(anchor, point) <= max_anchor_distance])
    filtered.extend([point for point in points if point.cluster_id < 0])
    filtered.sort(key=lambda point: point.rank)
    return filtered


def spread_indices(length: int, sample_size: int, sample_index: int) -> list[int]:
    if sample_size == 1:
        return [sample_index % length]
    tail_count = sample_size - 1
    tail_positions = np.linspace(1, length - 1, num=tail_count, dtype=int).tolist()
    offset = sample_index % max(length - 1, 1)
    shifted = [1 + ((pos - 1 + offset) % max(length - 1, 1)) for pos in tail_positions]
    indices = [0]
    for idx in shifted:
        if idx not in indices:
            indices.append(idx)
    cursor = 1
    while len(indices) < sample_size and cursor < length:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return indices[:sample_size]


def sample_prompt_groups(
    points: list[CandidatePoint],
    sample_sizes: list[int],
    samples_per_cluster: int,
) -> list[PromptGroup]:
    groups: list[PromptGroup] = []
    group_id = 1
    cluster_ids = sorted({point.cluster_id for point in points if point.cluster_id >= 0})
    for cluster_id in cluster_ids:
        cluster = [point for point in points if point.cluster_id == cluster_id]
        cluster.sort(key=lambda point: point.rank)
        for sample_size in sample_sizes:
            if len(cluster) < sample_size:
                continue
            for sample_index in range(int(samples_per_cluster)):
                indices = spread_indices(len(cluster), sample_size, sample_index)
                selected = [cluster[idx] for idx in indices]
                sims = [point.similarity_after for point in selected]
                groups.append(
                    PromptGroup(
                        group_id=group_id,
                        cluster_id=cluster_id,
                        sample_size=sample_size,
                        sample_index=sample_index + 1,
                        point_count=len(selected),
                        mean_similarity_after=round(float(np.mean(sims)), 6),
                        max_similarity_after=round(float(np.max(sims)), 6),
                        prompt_points=[
                            {
                                "rank": float(point.rank),
                                "x": round(point.x, 3),
                                "y": round(point.y, 3),
                                "similarity_after": round(point.similarity_after, 6),
                            }
                            for point in selected
                        ],
                    )
                )
                group_id += 1
    return groups


def color_for_id(index: int) -> tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
    ]
    return palette[index % len(palette)]


def draw_cluster_overlay(target: Image.Image, points: list[CandidatePoint]) -> Image.Image:
    image = target.copy()
    draw = ImageDraw.Draw(image)
    for point in points:
        if point.cluster_id < 0:
            fill = (120, 120, 120)
            radius = 4
        else:
            fill = color_for_id(point.cluster_id)
            radius = 6
        draw.ellipse((point.x - radius, point.y - radius, point.x + radius, point.y + radius), fill=fill, outline=(255, 255, 255), width=2)
    draw.rectangle((4, image.height - 34, min(image.width - 4, 430), image.height - 4), fill=(0, 0, 0))
    draw.text((12, image.height - 26), "cluster colors | gray = noise", fill=(255, 255, 255))
    return image


def draw_sample_overlay(target: Image.Image, groups: list[PromptGroup], max_groups: int = 24) -> Image.Image:
    image = target.copy()
    draw = ImageDraw.Draw(image)
    for group in groups[:max_groups]:
        fill = color_for_id(group.group_id)
        coords = [(point["x"], point["y"]) for point in group.prompt_points]
        if len(coords) > 1:
            draw.line(coords, fill=fill, width=2)
        for x, y in coords:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=fill, outline=(255, 255, 255), width=2)
    draw.rectangle((4, image.height - 48, min(image.width - 4, 510), image.height - 4), fill=(0, 0, 0))
    draw.text((12, image.height - 40), f"first {min(max_groups, len(groups))} prompt groups", fill=(255, 255, 255))
    draw.text((12, image.height - 22), "lines connect points from the same sampled prompt", fill=(255, 255, 255))
    return image


def save_stage2_outputs(
    out_dir: Path,
    feedback_dir: Path,
    payload: dict[str, Any],
    points: list[CandidatePoint],
    groups: list[PromptGroup],
    target: Image.Image,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    point_rows = [asdict(point) for point in points]
    group_rows = []
    for group in groups:
        row = asdict(group)
        row["prompt_points"] = json.dumps(group.prompt_points, ensure_ascii=False)
        group_rows.append(row)

    with (out_dir / "cluster_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()) if point_rows else list(CandidatePoint.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(point_rows)

    with (out_dir / "prompt_groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()) if group_rows else list(PromptGroup.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    cluster_ids = sorted({point.cluster_id for point in points if point.cluster_id >= 0})
    noise_count = sum(1 for point in points if point.cluster_id < 0)
    cluster_sizes = {str(cluster_id): sum(1 for point in points if point.cluster_id == cluster_id) for cluster_id in cluster_ids}
    result = {
        "feedback_dir": str(feedback_dir.relative_to(WORKSPACE_ROOT)),
        "target_image": payload["target_image"],
        "selected_only": args.selected_only,
        "top_ratio": args.top_ratio,
        "cluster_method": args.cluster_method,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "sample_sizes": parse_int_list(args.sample_sizes),
        "samples_per_cluster": args.samples_per_cluster,
        "max_anchor_distance": args.max_anchor_distance,
        "candidate_count": len(points),
        "cluster_count": len(cluster_ids),
        "estimated_k": len(cluster_ids),
        "noise_count": noise_count,
        "cluster_sizes": cluster_sizes,
        "prompt_group_count": len(groups),
        "points": point_rows,
        "prompt_groups": [asdict(group) for group in groups],
    }
    (out_dir / "prompt_groups.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_cluster_overlay(target, points).save(out_dir / "cluster_overlay.png")
    draw_sample_overlay(target, groups).save(out_dir / "sample_overlay.png")

    command = (
        "D:\\SAM\\conda_envs\\sam_gpu\\python.exe -B code\\benchmarks\\sample_stage2_prompts.py "
        f"--feedback-dir {args.feedback_dir} --top-ratio {args.top_ratio} "
        f"--cluster-method {args.cluster_method} --eps {args.eps} --min-samples {args.min_samples} "
        f"--sample-sizes {args.sample_sizes} --samples-per-cluster {args.samples_per_cluster} "
        f"--max-anchor-distance {args.max_anchor_distance} "
        f"--out-dir {args.out_dir}"
    )
    if args.selected_only:
        command += " --selected-only"
    summary = [
        "# Matcher Stage 2 Prompt Sampling Summary",
        "",
        f"- Feedback dir: `{result['feedback_dir']}`",
        f"- Selected only: `{args.selected_only}`",
        f"- Top ratio: `{args.top_ratio}`",
        f"- Cluster method: `{args.cluster_method}`",
        f"- Eps: `{args.eps}`",
        f"- Min samples: `{args.min_samples}`",
        f"- Max anchor distance: `{args.max_anchor_distance}`",
        f"- Candidate count: `{len(points)}`",
        f"- Cluster count / estimated K: `{len(cluster_ids)}`",
        f"- Noise count: `{noise_count}`",
        f"- Cluster sizes: `{cluster_sizes}`",
        f"- Prompt group count: `{len(groups)}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Artifacts",
        "",
        "- `cluster_points.csv`",
        "- `prompt_groups.csv`",
        "- `prompt_groups.json`",
        "- `cluster_overlay.png`",
        "- `sample_overlay.png`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster feedback-reranked Matcher points and sample Stage 2 point prompts.")
    parser.add_argument("--feedback-dir", default="runs/research_direction/stage1_feedback/elephant_pair")
    parser.add_argument("--top-ratio", type=float, default=0.5)
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--cluster-method", choices=["none", "dbscan"], default="dbscan")
    parser.add_argument("--eps", type=float, default=55.0)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sample-sizes", default="1,3,5")
    parser.add_argument("--samples-per-cluster", type=int, default=4)
    parser.add_argument("--max-anchor-distance", type=float, default=0.0)
    parser.add_argument("--out-dir", default="runs/research_direction/stage2_sampling/elephant_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback_dir = resolve_path(args.feedback_dir)
    out_dir = resolve_path(args.out_dir)
    payload, all_points, target = load_feedback_points(feedback_dir)
    candidate_points = select_candidate_points(all_points, top_ratio=args.top_ratio, selected_only=args.selected_only)
    clustered_points = cluster_points(candidate_points, cluster_method=args.cluster_method, eps=args.eps, min_samples=args.min_samples)
    clustered_points = filter_cluster_points_by_anchor(clustered_points, max_anchor_distance=args.max_anchor_distance)
    groups = sample_prompt_groups(clustered_points, sample_sizes=parse_int_list(args.sample_sizes), samples_per_cluster=args.samples_per_cluster)
    save_stage2_outputs(
        out_dir=out_dir,
        feedback_dir=feedback_dir,
        payload=payload,
        points=clustered_points,
        groups=groups,
        target=target,
        args=args,
    )
    cluster_count = len({point.cluster_id for point in clustered_points if point.cluster_id >= 0})
    noise_count = sum(1 for point in clustered_points if point.cluster_id < 0)
    print(f"Candidate points: {len(clustered_points)}")
    print(f"Estimated K / clusters: {cluster_count}")
    print(f"Noise points: {noise_count}")
    print(f"Prompt groups: {len(groups)}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
