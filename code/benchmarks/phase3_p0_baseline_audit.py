from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = WORKSPACE_ROOT / "runs" / "phase3_p0_baselines"
REPORT = WORKSPACE_ROOT / "docs" / "phase3_p0_baseline_integration.md"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_rows() -> list[dict[str, Any]]:
    ckpt = WORKSPACE_ROOT / "assets" / "checkpoints"
    adapter_zoo = ckpt / "medical_adapter_zoo"
    rows = [
        {
            "baseline": "SAM-Med2D",
            "repo": rel(WORKSPACE_ROOT / "SAM-Med2D"),
            "repo_exists": int((WORKSPACE_ROOT / "SAM-Med2D").exists()),
            "weights": rel(ckpt / "sam-med2d_b.pth"),
            "weights_exists": int((ckpt / "sam-med2d_b.pth").exists()),
            "status": "code_ready_weight_missing",
            "nearest_use": "box/point prompt medical SAM baseline once sam-med2d_b.pth is downloaded",
            "next_command": "Download official sam-med2d_b.pth from the README Google Drive/Baidu link, then place it at assets/checkpoints/sam-med2d_b.pth.",
        },
        {
            "baseline": "Med-SA Adapter Zoo OpticDisc",
            "repo": rel(WORKSPACE_ROOT / "Medical-SAM-Adapter"),
            "repo_exists": int((WORKSPACE_ROOT / "Medical-SAM-Adapter").exists()),
            "weights": rel(adapter_zoo / "OpticDisc_Fundus_SAM_1024.pth"),
            "weights_exists": int((adapter_zoo / "OpticDisc_Fundus_SAM_1024.pth").exists()),
            "status": "code_and_weight_ready_adapter_needed",
            "nearest_use": "fundus-domain adapter sanity baseline; not identical to IDRID lesion segmentation",
            "next_command": "Implement Medical-SAM-Adapter runner for SAM + adapter checkpoint, then evaluate on IDRID as domain-near baseline.",
        },
        {
            "baseline": "Med-SA Adapter Zoo Microscope",
            "repo": rel(WORKSPACE_ROOT / "Medical-SAM-Adapter"),
            "repo_exists": int((WORKSPACE_ROOT / "Medical-SAM-Adapter").exists()),
            "weights": rel(adapter_zoo / "White Blood Cell_MicroScope_sam_1024.pth"),
            "weights_exists": int((adapter_zoo / "White Blood Cell_MicroScope_sam_1024.pth").exists()),
            "status": "code_and_weight_ready_adapter_needed",
            "nearest_use": "microscope-domain adapter sanity baseline for MonuSeg-like histology imagery",
            "next_command": "Implement Medical-SAM-Adapter runner for SAM + adapter checkpoint, then evaluate on MonuSeg as domain-near baseline.",
        },
        {
            "baseline": "nnU-Net v2",
            "repo": rel(WORKSPACE_ROOT / "nnUNet"),
            "repo_exists": int((WORKSPACE_ROOT / "nnUNet").exists()),
            "weights": rel(WORKSPACE_ROOT / "runs" / "nnunet" / "nnUNet_raw" / "Dataset901_OCSAMPhase3"),
            "weights_exists": int((WORKSPACE_ROOT / "runs" / "nnunet" / "nnUNet_raw" / "Dataset901_OCSAMPhase3" / "dataset.json").exists()),
            "status": "code_and_dataset_ready_training_required",
            "nearest_use": "supervised upper-reference baseline; Phase3 2D dataset has been converted to nnU-Net raw format",
            "next_command": "Install nnunetv2, set nnUNet_raw/preprocessed/results env vars, run planning/preprocessing and train 2d configuration.",
        },
        {
            "baseline": "Medical SAM2 / MedSAM2",
            "repo": rel(WORKSPACE_ROOT / "MedSAM2"),
            "repo_exists": int((WORKSPACE_ROOT / "MedSAM2").exists()),
            "weights": rel(ckpt / "medsam2.pth"),
            "weights_exists": int((ckpt / "medsam2.pth").exists()),
            "status": "not_cloned_project_page_identified",
            "nearest_use": "2D/3D/video medical SAM2 baseline for memory/propagation claim",
            "next_command": "Resolve the exact official repo/checkpoint from the Medical-SAM2 project page before cloning; then add video/volume protocol.",
        },
    ]
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    ready = [row for row in rows if row["repo_exists"] and row["weights_exists"]]
    code_ready = [row for row in rows if row["repo_exists"]]
    lines = [
        "# Phase3 P0 Baseline 接入审计",
        "",
        "本报告记录 P0 强基线从“口头计划”推进到“本地可审计状态”的结果。",
        "",
        f"- 已有代码仓库：{len(code_ready)} / {len(rows)}",
        f"- 已有代码和权重：{len(ready)} / {len(rows)}",
        "",
        "| Baseline | 状态 | 仓库 | 权重 | 下一步 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline']} | {row['status']} | {row['repo_exists']} | {row['weights_exists']} | {row['next_command']} |"
        )
    lines.extend(
        [
            "",
            "## 对当前论文路线的影响",
            "",
            "- `SAM-Med2D` 是最该补的 promptable medical SAM 强基线，但官方权重还没落到本地。",
            "- `Med-SA Adapter Zoo` 已经有两个可用 adapter 权重，下一步可以先做 domain-near sanity baseline。",
            "- `nnU-Net` 已经有代码和 Phase3 raw dataset，但它是监督训练 baseline，必须先训练，不能直接和 promptable SAM 混为一谈。",
            "- `MedSAM2 / Medical SAM2` 仍需要确认精确官方仓库和权重入口，适合支撑跨切片/视频 memory claim。",
            "",
            "## 下一步最短路径",
            "",
            "1. 先写 Medical-SAM-Adapter runner，接入已下载的 OpticDisc / Microscope adapter。",
            "2. 手动或脚本下载 `sam-med2d_b.pth` 后写 SAM-Med2D runner。",
            "3. 跑 nnU-Net 2D planning/preprocessing/training，作为监督参考上界。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = audit_rows()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_ROOT / "p0_baseline_readiness.csv", rows)
    write_report(rows)
    print(f"Wrote {REPORT}")
    print(f"Wrote {OUT_ROOT / 'p0_baseline_readiness.csv'}")


if __name__ == "__main__":
    main()
