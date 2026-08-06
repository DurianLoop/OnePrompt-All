from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = WORKSPACE_ROOT / "runs" / "phase2_lite"
MILESTONE5_ROOT = WORKSPACE_ROOT / "runs" / "phase3_milestone5"
OUT_ROOT = WORKSPACE_ROOT / "runs" / "phase3_acceptance_1_to_5"
REPORT_PATH = WORKSPACE_ROOT / "docs" / "phase3_acceptance_1_to_5.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def summarize_rows(rows: list[dict[str, str]], dice_key: str = "dice") -> dict[str, Any]:
    dice = [fnum(row.get(dice_key)) for row in rows]
    return {
        "rows": len(rows),
        "mean_dice": round(mean(dice), 4) if dice else "",
        "datasets": ",".join(sorted({row.get("dataset", "") for row in rows if row.get("dataset")})),
        "models": ",".join(sorted({row.get("model", "") for row in rows if row.get("model")})),
    }


def phase2_manifest_summary() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    manifest = read_csv(PHASE2_ROOT / "manifests" / "phase2_lite_40_cases.csv")
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        by_dataset[row["dataset"]].append(row)
    rows: list[dict[str, Any]] = []
    for dataset, items in sorted(by_dataset.items()):
        rows.append(
            {
                "source": "phase2_lite_manifest",
                "dataset": dataset,
                "case_count": len(items),
                "splits": ",".join(sorted({r.get("split", "") for r in items})),
                "difficulty_buckets": ",".join(sorted({r.get("difficulty_bucket", "") for r in items})),
                "status": "already_evaluated",
                "note": "已有 SAM ViT-B / MedSAM / SAM2 Tiny 的统一 prompt 评测结果。",
            }
        )
    return manifest, rows


def count_files(root: Path, suffixes: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def dataset_inventory_rows() -> list[dict[str, Any]]:
    datasets = WORKSPACE_ROOT / "code" / "datasets"
    medsam_demo = WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_download" / "test_demo"
    rows = [
        {
            "dataset_or_source": "IDRID",
            "path": rel(datasets / "IDRID"),
            "image_like_files": count_files(datasets / "IDRID", (".jpg", ".jpeg", ".png", ".tif", ".tiff")),
            "mask_like_files": count_files(datasets / "IDRID", (".png", ".tif", ".tiff")),
            "domain": "retinal fundus lesions",
            "phase3_use": "扩展 Phase2-lite 到更多眼底病例；适合测试小目标、多实例和低对比病灶。",
        },
        {
            "dataset_or_source": "MonuSeg",
            "path": rel(datasets / "MonuSeg"),
            "image_like_files": count_files(datasets / "MonuSeg", (".jpg", ".jpeg", ".png", ".tif", ".tiff")),
            "mask_like_files": count_files(datasets / "MonuSeg", (".png", ".tif", ".tiff")),
            "domain": "histopathology nuclei",
            "phase3_use": "扩展核分割病例；适合测试密集实例和边界粘连。",
        },
        {
            "dataset_or_source": "MedSAM demo npz",
            "path": rel(medsam_demo),
            "image_like_files": count_files(medsam_demo / "imgs", (".npz",)),
            "mask_like_files": count_files(medsam_demo / "gts", (".npz",)) + count_files(medsam_demo / "segs", (".npz",)),
            "domain": "CXR / dermoscopy / endoscopy / fundus / mammography / microscope / OCT / US / CT / MR",
            "phase3_use": "作为跨模态 sanity set，不适合作为最终主表，但适合快速验收 domain generalization 管线。",
        },
        {
            "dataset_or_source": "FSS-1000",
            "path": rel(datasets / "FSS-1000"),
            "image_like_files": count_files(datasets / "FSS-1000", (".jpg", ".jpeg", ".png")),
            "mask_like_files": count_files(datasets / "FSS-1000", (".png",)),
            "domain": "few-shot natural images",
            "phase3_use": "非医学，只能作为 Matcher/support 机制的工程回归测试，不进入医学主表。",
        },
    ]
    return rows


def medsam_demo_case_rows() -> list[dict[str, Any]]:
    root = WORKSPACE_ROOT / "assets" / "checkpoints" / "medsam_download" / "test_demo"
    imgs = sorted((root / "imgs").glob("*.npz")) if (root / "imgs").exists() else []
    rows: list[dict[str, Any]] = []
    for img in imgs:
        stem = img.stem
        rows.append(
            {
                "case_id": f"medsam_demo_{stem}",
                "domain": stem.replace("_demo", "").replace("2DBox_", "").replace("3DBox_", ""),
                "image_path": rel(img),
                "gt_path": rel(root / "gts" / img.name) if (root / "gts" / img.name).exists() else "",
                "seg_path": rel(root / "segs" / img.name) if (root / "segs" / img.name).exists() else "",
                "status": "ready_for_adapter",
                "note": "需要写 npz loader 后才能进入统一 predict/evaluate。",
            }
        )
    return rows


def baseline_rows() -> list[dict[str, Any]]:
    ckpt = WORKSPACE_ROOT / "assets" / "checkpoints"
    candidates = [
        ("SAM ViT-B", WORKSPACE_ROOT / "sam", ckpt / "sam_vit_b.pth", "done_phase2_lite", "已进入统一接口和 Phase2-lite。"),
        ("SAM ViT-H", WORKSPACE_ROOT / "sam", ckpt / "sam_vit_h_4b8939.pth", "checkpoint_ready", "权重存在；可作为更强 SAM backbone 的下一轮对照。"),
        ("MedSAM ViT-B", WORKSPACE_ROOT / "MedSAM", ckpt / "medsam_vit_b.pth", "done_phase2_lite", "已进入统一接口和 Phase2-lite。"),
        ("Lite-MedSAM", WORKSPACE_ROOT / "MedSAM", ckpt / "medsam_download" / "lite_medsam.pth", "checkpoint_ready", "权重存在；适合速度/轻量化对照。"),
        ("SAM2 Tiny", WORKSPACE_ROOT / "sam2", ckpt / "sam2.1_hiera_tiny.pt", "done_phase2_lite", "已进入统一接口和 Phase2-lite。"),
        ("Matcher", WORKSPACE_ROOT / "Matcher", ckpt / "best_backbone_idrid_finetuned.pth", "repo_and_ckpt_ready", "已有仓库和 DINOv2/IDRID backbone；需要纳入同一 metric 主表。"),
        ("SAM3", WORKSPACE_ROOT / "sam3", ckpt / "sam3_modelscope" / "sam3.pt", "demo_ready_not_box_point_baseline", "权重存在；更适合作为 text/concept 分割对照，不直接等价 SAM box/point。"),
        ("SAM-Med2D", WORKSPACE_ROOT / "SAM-Med2D", ckpt / "sam_med2d.pth", "missing", "P0 强基线，需要下载/接入。"),
        ("Med-SA / Medical SAM Adapter", WORKSPACE_ROOT / "Med-SA", ckpt / "med_sa.pth", "missing", "P0 强基线，需要下载/接入。"),
        ("MedSAM2 / Medical SAM2", WORKSPACE_ROOT / "MedSAM2", ckpt / "medsam2.pth", "missing", "P0 视频/3D/跨切片方向强基线，需要下载/接入。"),
        ("nnU-Net", WORKSPACE_ROOT / "nnUNet", WORKSPACE_ROOT / "runs" / "nnunet", "missing", "监督式医学分割强参照；需要训练 split 和推理脚本。"),
    ]
    rows: list[dict[str, Any]] = []
    for name, repo_path, weight_path, status, note in candidates:
        rows.append(
            {
                "baseline": name,
                "repo_path": rel(repo_path),
                "repo_exists": int(repo_path.exists()),
                "weight_or_output_path": rel(weight_path),
                "weight_or_output_exists": int(weight_path.exists()),
                "status": status,
                "priority": "P0" if status == "missing" else "P1" if "checkpoint_ready" in status or "repo_and" in status else "done",
                "note": note,
            }
        )
    return rows


def acceptance_rows() -> list[dict[str, Any]]:
    support = read_csv(MILESTONE5_ROOT / "support_guided_prompt_v1.csv")
    uncertainty = read_csv(MILESTONE5_ROOT / "perturbation_uncertainty_proxy_v1.csv")
    portfolio = read_csv(MILESTONE5_ROOT / "cross_model_calibrated_portfolio_v4.csv")
    data_inventory = dataset_inventory_rows()
    baselines = baseline_rows()
    checked_sources = [row for row in data_inventory if row["image_like_files"] or row["mask_like_files"]]
    ready_or_done_baselines = [row for row in baselines if row["status"] != "missing"]
    return [
        {
            "step": 1,
            "acceptance": "Support Prompt v1",
            "status": "pass_engineering",
            "evidence": rel(MILESTONE5_ROOT / "support_guided_prompt_v1.csv"),
            "rows": len(support),
            "mean_dice": summarize_rows(support)["mean_dice"],
            "next_gap": "当前 support 只用历史 Dice proxy；论文版需要用视觉相似度/病灶先验替代 oracle-like 历史表现。",
        },
        {
            "step": 2,
            "acceptance": "Perturbation Uncertainty v1",
            "status": "pass_proxy",
            "evidence": rel(MILESTONE5_ROOT / "perturbation_uncertainty_proxy_v1.csv"),
            "rows": len(uncertainty),
            "mean_dice": summarize_rows(uncertainty)["mean_dice"],
            "next_gap": "当前是 auto-family consensus proxy；下一版要真实生成 jitter prompts 并重跑模型。",
        },
        {
            "step": 3,
            "acceptance": "Cross-model Calibration",
            "status": "pass_engineering",
            "evidence": rel(MILESTONE5_ROOT / "cross_model_calibrated_portfolio_v4.csv"),
            "rows": len(portfolio),
            "mean_dice": summarize_rows(portfolio)["mean_dice"],
            "next_gap": "需要扩大数据后验证置信区间和显著性，目前还不能写 SOTA claim。",
        },
        {
            "step": 4,
            "acceptance": "扩数据准备",
            "status": "pass_preparation",
            "evidence": rel(OUT_ROOT / "dataset_expansion_inventory.csv"),
            "rows": len(checked_sources),
            "mean_dice": "",
            "next_gap": "已形成扩展清单；还需把 MedSAM demo npz 和更多 IDRID/MonuSeg 病例写入统一 manifest 并跑主表。",
        },
        {
            "step": 5,
            "acceptance": "强基线比较",
            "status": "pass_audit_not_scientific_claim",
            "evidence": rel(OUT_ROOT / "strong_baseline_readiness.csv"),
            "rows": len(ready_or_done_baselines),
            "mean_dice": "",
            "next_gap": "已完成 readiness audit；科学验收必须补 SAM-Med2D / Med-SA / MedSAM2 / nnU-Net 中至少两个。",
        },
    ]


def write_report(rows: list[dict[str, Any]], manifest: list[dict[str, str]], manifest_summary: list[dict[str, Any]]) -> None:
    metrics = read_csv(PHASE2_ROOT / "metrics" / "phase2_lite_metrics.csv")
    ok_metrics = [row for row in metrics if row.get("status") == "ok"]
    by_dataset = Counter(row.get("dataset", "") for row in manifest)
    by_model = defaultdict(list)
    for row in ok_metrics:
        by_model[row["model"]].append(fnum(row.get("dice")))
    lines = [
        "# Phase3 五个验收标准推进报告",
        "",
        "这份报告把当前阶段从第 1 个验收标准推进到第 5 个验收标准。需要强调：这里完成的是可复现工程验收和研究路线验收，不伪造未运行强基线的指标。",
        "",
        "## 1. 验收总表",
        "",
        "| Step | 验收标准 | 状态 | 证据文件 | 行数 | Mean Dice | 仍需补齐 |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['acceptance']} | {row['status']} | `{row['evidence']}` | {row['rows']} | {row['mean_dice']} | {row['next_gap']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 当前 Phase2-lite 基础",
            "",
            f"- 已评估 manifest：`{rel(PHASE2_ROOT / 'manifests' / 'phase2_lite_40_cases.csv')}`。",
            f"- 当前病例数：{len(manifest)}，数据集分布：{dict(by_dataset)}。",
            f"- 当前有效 metric 行：{len(ok_metrics)}。",
            "",
            "| 模型 | Mean Dice | 有效行数 |",
            "| --- | ---: | ---: |",
        ]
    )
    for model, values in sorted(by_model.items()):
        lines.append(f"| {model} | {mean(values):.4f} | {len(values)} |")
    lines.extend(
        [
            "",
            "## 3. 扩数据验收结论",
            "",
            "扩数据不是简单堆样本，而是要覆盖不同医学边界失败模式：眼底小病灶、病理密集核、超声/内镜/OCT 等低对比和伪影场景。当前已经生成数据库存和 MedSAM demo 适配清单。",
            "",
            "| 来源 | 病例/文件概况 | 进入 Phase3 的用途 |",
            "| --- | --- | --- |",
        ]
    )
    inventory = read_csv(OUT_ROOT / "dataset_expansion_inventory.csv")
    for row in inventory:
        lines.append(
            f"| {row['dataset_or_source']} | image-like={row['image_like_files']}, mask-like={row['mask_like_files']} | {row['phase3_use']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 强基线验收结论",
            "",
            "第 5 个验收标准目前完成了强基线 readiness audit：能清楚区分已经可跑、已有权重待接入、仍缺仓库/权重的模型。论文主张仍必须等 P0 强基线补齐后再收紧。",
            "",
            "| Baseline | 状态 | 仓库 | 权重/输出 | 下一步 |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in read_csv(OUT_ROOT / "strong_baseline_readiness.csv"):
        lines.append(
            f"| {row['baseline']} | {row['status']} | {row['repo_exists']} | {row['weight_or_output_exists']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 现在可以站住的结论",
            "",
            "OCSAM-X 已经从单一 prompt selector 推进为一个可复现实验链条：support-guided prompt selection、prompt-induced uncertainty proxy、cross-model calibrated portfolio，再加上扩数据和强基线 readiness audit。",
            "",
            "但当前还不是顶会级最终结论。真正的 Phase3 连接点是：在更大医学数据、多强基线、真实 prompt jitter uncertainty 下，证明这种 support + uncertainty + cross-model calibration 的组合能稳定优于固定 prompt、单模型 SAM/MedSAM/SAM2，以及至少部分近年医学 SAM adapter baseline。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest, manifest_summary = phase2_manifest_summary()
    write_csv(OUT_ROOT / "phase2_manifest_summary.csv", manifest_summary)
    write_csv(OUT_ROOT / "dataset_expansion_inventory.csv", dataset_inventory_rows())
    write_csv(OUT_ROOT / "medsam_demo_extension_candidates.csv", medsam_demo_case_rows())
    write_csv(OUT_ROOT / "strong_baseline_readiness.csv", baseline_rows())
    rows = acceptance_rows()
    write_csv(OUT_ROOT / "acceptance_1_to_5_checklist.csv", rows)
    write_report(rows, manifest, manifest_summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
