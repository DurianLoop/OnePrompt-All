from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "adapromptset.run_metadata.v1"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_config_hash(payload: Any) -> str:
    """Return a stable SHA256 for JSON-serializable config content."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": _relative_path(path, root),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _git_value(project_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def collect_git_metadata(project_root: Path) -> dict[str, str]:
    status = _git_value(project_root, ["status", "--porcelain"])
    status_lines = [] if status in {"", "UNKNOWN"} else status.splitlines()
    return {
        "commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "branch": _git_value(project_root, ["branch", "--show-current"]),
        "is_dirty": str(status not in {"", "UNKNOWN"}).lower(),
        "dirty_file_count": str(len(status_lines)),
        "dirty_file_sample": "\n".join(status_lines[:20]),
    }


def collect_environment_metadata() -> dict[str, str]:
    payload = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }
    try:
        import torch  # type: ignore

        payload.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": str(torch.cuda.is_available()).lower(),
                "cuda_version": str(getattr(torch.version, "cuda", None)),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }
        )
    except Exception:
        payload.update({"torch": "UNAVAILABLE", "cuda_available": "unknown", "cuda_version": "", "gpu_name": ""})
    return payload


def collect_run_metadata(
    project_root: Path,
    run_name: str,
    config_paths: Iterable[Path],
    input_paths: Iterable[Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    configs = [_file_record(path.resolve(), root) for path in config_paths]
    inputs = [_file_record(path.resolve(), root) for path in input_paths]
    config_hash = canonical_config_hash(configs)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name).strip("_") or "run"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "run_id": f"{run_slug}_{config_hash[:12]}",
        "timestamp_utc": timestamp,
        "project_root": str(root),
        "config_hash": config_hash,
        "configs": configs,
        "inputs": inputs,
        "git": collect_git_metadata(root),
        "environment": collect_environment_metadata(),
        "extra": dict(extra or {}),
    }


def write_run_metadata(
    out_dir: Path,
    project_root: Path,
    run_name: str,
    config_paths: Iterable[Path],
    input_paths: Iterable[Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = collect_run_metadata(
        project_root=project_root,
        run_name=run_name,
        config_paths=config_paths,
        input_paths=input_paths,
        extra=extra,
    )
    out_path = out_dir / "run_metadata.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
