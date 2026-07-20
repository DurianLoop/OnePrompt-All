from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        payload[key.strip()] = value.strip().strip("'\"")
    return payload


def _dataset_roots(project_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    config_dir = project_root / "configs" / "data"
    if not config_dir.is_dir():
        return roots
    for path in sorted(config_dir.glob("*.yaml")):
        payload = _parse_simple_yaml(path)
        name = payload.get("name")
        root = payload.get("root")
        if name and root:
            roots[name] = Path(root)
    return roots


def _resolve_data_path(dataset_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    root_relative = dataset_root / path
    if root_relative.exists():
        return root_relative
    return dataset_root.parent / path


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _profile_row(dataset_root: Path, row: dict[str, str]) -> dict[str, Any]:
    image_path = _resolve_data_path(dataset_root, row["image_path"])
    mask_path = _resolve_data_path(dataset_root, row["mask_path"])
    with Image.open(image_path) as image:
        width, height = image.size
        image_mode = image.mode
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
    if mask.ndim == 3:
        foreground = np.any(mask != 0, axis=2)
        labels = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)
        nonzero_labels = [label for label in labels if np.any(label != 0)]
    else:
        foreground = mask != 0
        labels = np.unique(mask)
        nonzero_labels = [label for label in labels if int(label) != 0]
    mask_area = int(np.count_nonzero(foreground))
    pixels = int(width * height)
    return {
        "dataset": row.get("dataset", ""),
        "case_id": row.get("case_id", ""),
        "split": row.get("split", ""),
        "role": row.get("role", row.get("split", "")),
        "image_width": int(width),
        "image_height": int(height),
        "image_mode": image_mode,
        "mask_area_pixels": mask_area,
        "mask_area_fraction": mask_area / pixels if pixels else 0.0,
        "mask_unique_nonzero_count": len(nonzero_labels),
        "manifest_instance_count": int(row["instance_count"]) if row.get("instance_count", "").isdigit() else None,
    }


def _manifest_paths(project_root: Path) -> list[Path]:
    return sorted((project_root / "manifests").glob("*_manifest.csv"))


def _summarize_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    instance_values = [
        float(row["manifest_instance_count"])
        if row["manifest_instance_count"] is not None
        else float(row["mask_unique_nonzero_count"])
        for row in rows
    ]
    return {
        "dataset": dataset,
        "case_count": len(rows),
        "width": _numeric_summary([float(row["image_width"]) for row in rows]),
        "height": _numeric_summary([float(row["image_height"]) for row in rows]),
        "mask_area_pixels": _numeric_summary([float(row["mask_area_pixels"]) for row in rows]),
        "mask_area_fraction": _numeric_summary([float(row["mask_area_fraction"]) for row in rows]),
        "instance_count": _numeric_summary(instance_values),
        "roles": sorted({row["role"] for row in rows}),
    }


def build_data_profile_audit(project_root: Path = PROJECT_ROOT, max_rows: int | None = None) -> dict[str, Any]:
    roots = _dataset_roots(project_root)
    profile_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for manifest_path in _manifest_paths(project_root):
        rows = _read_csv(manifest_path)
        if max_rows is not None:
            rows = rows[:max_rows]
        if not rows:
            continue
        dataset = rows[0].get("dataset", manifest_path.stem.replace("_manifest", ""))
        dataset_root = roots.get(dataset)
        if dataset_root is None:
            issues.append(
                {
                    "dataset": dataset,
                    "case_id": "",
                    "reason": "missing_dataset_root_config",
                    "path": manifest_path.relative_to(project_root).as_posix(),
                }
            )
            continue
        for row in rows:
            try:
                profile_rows.append(_profile_row(dataset_root, row))
            except Exception as exc:  # noqa: BLE001 - report per-file profiling failures.
                issues.append(
                    {
                        "dataset": dataset,
                        "case_id": row.get("case_id", ""),
                        "reason": type(exc).__name__,
                        "path": row.get("image_path", ""),
                    }
                )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in profile_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    return {
        "schema_version": "adapromptset.data_profile_audit.v1",
        "status": "PASS" if profile_rows and not issues else "FAIL",
        "dataset_count": len(by_dataset),
        "row_count": len(profile_rows),
        "datasets": [_summarize_dataset(dataset, rows) for dataset, rows in sorted(by_dataset.items())],
        "rows": profile_rows,
        "issues": issues,
        "rule": "Profile every manifest image/mask pair for dimensions, foreground mask area, and instance/label count distribution.",
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Data Profile Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This report summarizes image dimensions, mask area, and instance count distributions from manifest image/mask pairs.",
        "",
        "| Dataset | Cases | Width mean | Height mean | Mask area mean | Mask fraction mean | Instance count mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in payload["datasets"]:
        lines.append(
            "| "
            f"{dataset['dataset']} | "
            f"{dataset['case_count']} | "
            f"{dataset['width']['mean']:.2f} | "
            f"{dataset['height']['mean']:.2f} | "
            f"{dataset['mask_area_pixels']['mean']:.2f} | "
            f"{dataset['mask_area_fraction']['mean']:.6f} | "
            f"{dataset['instance_count']['mean']:.2f} |"
        )
    lines.extend(["", "## Issues", ""])
    if payload["issues"]:
        lines.extend(
            f"- `{issue['dataset']}` `{issue['case_id']}`: {issue['reason']} ({issue['path']})"
            for issue in payload["issues"]
        )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_data_profile_audit(
    project_root: Path = PROJECT_ROOT,
    max_rows: int | None = None,
) -> dict[str, Path]:
    payload = build_data_profile_audit(project_root, max_rows=max_rows)
    outputs = {
        "json": project_root / "reports" / "data_profile_audit.json",
        "markdown": project_root / "reports" / "data_profile_audit.md",
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(outputs["markdown"], payload)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image dimensions, mask area, and instance distributions.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    export_data_profile_audit(args.project_root, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
