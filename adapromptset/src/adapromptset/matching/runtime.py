from __future__ import annotations

import hashlib
import importlib.util
import sys
import platform
from pathlib import Path
from typing import Any


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def infer_dino_architecture(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    if "vitl" in name:
        return "large"
    if "vitb" in name:
        return "base"
    raise ValueError(f"cannot infer DINOv2 architecture from checkpoint name: {checkpoint.name}")


def infer_sam_architecture(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    for architecture in ("vit_b", "vit_l", "vit_h"):
        if architecture in name:
            return architecture
    raise ValueError(f"cannot infer SAM architecture from checkpoint name: {checkpoint.name}")


def checkpoint_record(path: Path, *, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.is_file(),
        "sha256": "",
        "bytes": 0,
    }
    if path.is_file():
        record["sha256"] = compute_sha256(path)
        record["bytes"] = path.stat().st_size
    if role == "matcher" and path.name:
        record["architecture"] = infer_dino_architecture(path)
    if role == "segmenter" and path.name:
        record["architecture"] = infer_sam_architecture(path)
    return record


def configure_external_paths(workspace_root: Path) -> list[str]:
    paths: list[str] = []
    for source_root in (workspace_root / "code", workspace_root / "Matcher"):
        text = str(source_root)
        paths.append(text)
        if source_root.exists() and text not in sys.path:
            sys.path.insert(0, text)
    return paths


def dependency_status() -> dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "torchvision": importlib.util.find_spec("torchvision") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
        "dinov2": importlib.util.find_spec("dinov2") is not None,
        "segment_anything": importlib.util.find_spec("segment_anything") is not None,
    }


def runtime_versions() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
    }
    if importlib.util.find_spec("torch") is None:
        return payload
    import torch

    payload["torch"] = getattr(torch, "__version__", "")
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_version"] = getattr(torch.version, "cuda", "") or ""
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
    return payload


def build_feature_cache_key(
    *,
    dataset: str,
    episode_id: str,
    support_image_sha256: str,
    support_mask_sha256: str,
    query_image_sha256: str,
    matcher_checkpoint_sha256: str,
    matcher_name: str,
) -> dict[str, str]:
    """Build a cache key that intentionally excludes query mask / query GT."""
    return {
        "dataset": dataset,
        "episode_id": episode_id,
        "support_image_sha256": support_image_sha256,
        "support_mask_sha256": support_mask_sha256,
        "query_image_sha256": query_image_sha256,
        "matcher_checkpoint_sha256": matcher_checkpoint_sha256,
        "matcher_name": matcher_name,
    }


def cache_key_digest(key: dict[str, str]) -> str:
    canonical = "\n".join(f"{name}={key[name]}" for name in sorted(key))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
