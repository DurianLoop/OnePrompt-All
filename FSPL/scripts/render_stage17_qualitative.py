from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IDRID_ROOT = Path(r"D:\SAM\code\datasets\IDRID")
DINO_CHECKPOINT = Path(r"D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth")
SAM_CHECKPOINT = Path(r"D:\SAM\assets\checkpoints\sam_vit_b.pth")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render predeclared stage-17 qualitative panels without metric selection.")
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idrid-root", type=Path, default=IDRID_ROOT)
    parser.add_argument("--dino-checkpoint", type=Path, default=DINO_CHECKPOINT)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _overlay(ax, image, mask, positive_points, negative_points, *, title: str, contour_color: str) -> None:
    import numpy as np

    ax.imshow(image)
    if mask.any():
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=contour_color, linewidths=1.0)
    if positive_points:
        points = np.asarray(positive_points)
        ax.scatter(points[:, 0], points[:, 1], c="#00d4ff", edgecolors="black", s=20, marker="o", linewidths=0.5)
    if negative_points:
        points = np.asarray(negative_points)
        ax.scatter(points[:, 0], points[:, 1], c="#ff3b30", edgecolors="black", s=25, marker="x", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.panel_manifest.is_file():
        raise FileNotFoundError(f"missing panel manifest: {args.panel_manifest}")
    if not args.dino_checkpoint.is_file() or not args.sam_checkpoint.is_file():
        raise FileNotFoundError("both frozen checkpoints must exist")
    payload = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    panels = payload.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("panel manifest must contain at least one panel")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import torch

    from fspl.candidates import build_candidate_pools
    from fspl.dataset import discover_idrid_cases
    from fspl.models import (
        IMAGE_SIZE,
        PATCH_SIZE,
        load_binary_mask,
        load_dinov2,
        load_mask_for_features,
        load_rgb,
        load_sam_predictor,
        matching_maps_from_prototypes,
        predict_sam,
        reference_prototypes,
    )
    from fspl.stage1 import _scale_points, image_resample
    from fspl.stage4 import build_spatial_radius_rules, rule_candidate_kwargs

    references = {case.case_id: case for case in discover_idrid_cases(args.idrid_root, "train")}
    queries = {case.case_id: case for case in discover_idrid_cases(args.idrid_root, "test")}
    rule_by_name = {rule.name: rule for rule in build_spatial_radius_rules((2.0, 3.0, 4.0))}
    device = torch.device(args.device)
    matcher = load_dinov2(args.dino_checkpoint, device)
    predictor = load_sam_predictor(args.sam_checkpoint, device)
    contexts = {}
    for reference_case_id in payload["reference_case_ids"]:
        reference = references.get(reference_case_id)
        if reference is None:
            raise ValueError(f"locked reference is absent from IDRiD training pairs: {reference_case_id}")
        image = load_rgb(reference.image_path).resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
        contexts[reference_case_id] = reference_prototypes(
            matcher, image, load_mask_for_features(reference.mask_path), device
        )

    outputs = []
    for panel in panels:
        query_case_id = str(panel["query_case_id"])
        query = queries.get(query_case_id)
        if query is None:
            raise ValueError(f"locked qualitative query is absent from IDRiD test pairs: {query_case_id}")
        query_image = load_rgb(query.image_path)
        query_mask = load_binary_mask(query.mask_path, size=query_image.size)
        reference_panels = panel["reference_panels"]
        figure, axes = plt.subplots(len(reference_panels), 4, figsize=(16, 4 * len(reference_panels)))
        if len(reference_panels) == 1:
            axes = [axes]
        figure.suptitle(
            f"Predeclared IDRiD qualitative panel: {query_case_id} (all fixed references; no metric-based selection)",
            fontsize=12,
        )
        for row_axes, reference_panel in zip(axes, reference_panels, strict=True):
            reference_case_id = str(reference_panel["reference_case_id"])
            reference = references[reference_case_id]
            foreground_prototype, background_prototype = contexts[reference_case_id]
            query_resized = query_image.resize((IMAGE_SIZE, IMAGE_SIZE), image_resample())
            foreground_scores, background_scores = matching_maps_from_prototypes(
                matcher, foreground_prototype, background_prototype, query_resized, device
            )
            _overlay(
                row_axes[0],
                load_rgb(reference.image_path),
                load_binary_mask(reference.mask_path),
                [],
                [],
                title=f"Reference: {reference_case_id}",
                contour_color="#33ff66",
            )
            _overlay(
                row_axes[1],
                query_image,
                query_mask,
                [],
                [],
                title=f"Query ground truth: {query_case_id}",
                contour_color="#ffd60a",
            )
            for ax, rule_name, label in (
                (row_axes[2], "top5_n3", "Matched Top-5"),
                (row_axes[3], str(reference_panel["selected_policy_rule"]), "Frozen policy"),
            ):
                rule = rule_by_name.get(rule_name)
                if rule is None:
                    raise ValueError(f"panel requests an unknown locked action: {rule_name}")
                pools = build_candidate_pools(
                    foreground_scores, background_scores, patch_size=PATCH_SIZE, **rule_candidate_kwargs(rule)
                )
                positive_points, negative_points = _scale_points(pools, query_image.size)
                prediction, _ = predict_sam(predictor, query_image, positive_points, negative_points)
                _overlay(
                    ax,
                    query_image,
                    prediction,
                    positive_points,
                    negative_points,
                    title=f"{label}: {rule_name}",
                    contour_color="#00d084",
                )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        image_path = args.output_dir / f"{query_case_id}_panel.png"
        figure.savefig(image_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append({"query_case_id": query_case_id, "image_path": str(image_path)})
    summary = {
        "panel_manifest": str(args.panel_manifest),
        "rendering_rule": "all predeclared references and queries; no utility or Dice used for selection",
        "outputs": outputs,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
