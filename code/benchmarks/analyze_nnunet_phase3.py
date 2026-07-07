from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = WORKSPACE_ROOT / "runs" / "nnunet" / "nnUNet_raw" / "Dataset901_OCSAMPhase3"
RESULT_ROOT = (
    WORKSPACE_ROOT
    / "runs"
    / "nnunet"
    / "nnUNet_results"
    / "Dataset901_OCSAMPhase3"
    / "nnUNetTrainer_250epochs__nnUNetPlans__2d"
    / "fold_0"
)
PRED_ROOT = WORKSPACE_ROOT / "runs" / "nnunet" / "predictions" / "Dataset901_OCSAMPhase3_fold0_250epochs"
PHASE3_METRICS = WORKSPACE_ROOT / "runs" / "phase3_full_with_medsa" / "metrics_with_medsa.csv"
OUT_ROOT = WORKSPACE_ROOT / "runs" / "nnunet" / "analysis" / "Dataset901_OCSAMPhase3_fold0_250epochs"
REPORT = WORKSPACE_ROOT / "docs" / "phase3_nnunet_250epochs_analysis.md"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int, int]:
    inter = int(np.logical_and(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    dice = 1.0 if pred_pixels + gt_pixels == 0 else 2.0 * inter / (pred_pixels + gt_pixels)
    union = int(np.logical_or(pred, gt).sum())
    iou = 1.0 if union == 0 else inter / union
    return dice, iou, inter, pred_pixels, gt_pixels


def summarize(rows: Iterable[dict[str, object]], key: str = "dice") -> dict[str, object]:
    vals = [float(row[key]) for row in rows]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def evaluate_test_predictions() -> list[dict[str, object]]:
    split_records = read_json(RAW_ROOT / "ocsam_split_manifest.json")
    records = {row["case_id"]: row for row in split_records}
    rows: list[dict[str, object]] = []
    for pred_path in sorted(PRED_ROOT.glob("OCSAM_*.png")):
        record = records[pred_path.stem]
        label_path = RAW_ROOT / record["label"]
        dice, iou, inter, pred_pixels, gt_pixels = dice_iou(load_mask(pred_path), load_mask(label_path))
        rows.append(
            {
                "case_id": pred_path.stem,
                "source_case_id": record["source_case_id"],
                "dataset": record["dataset"],
                "split": record["split"],
                "dice": round(dice, 6),
                "iou": round(iou, 6),
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "pred_gt_ratio": round(pred_pixels / max(1, gt_pixels), 6),
                "intersection": inter,
                "prediction": str(pred_path.relative_to(WORKSPACE_ROOT)),
                "label": str(label_path.relative_to(WORKSPACE_ROOT)),
            }
        )
    return rows


def summarize_promptable_same_cases(test_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_ids = {str(row["source_case_id"]) for row in test_rows}
    metric_rows = [
        row
        for row in read_csv(PHASE3_METRICS)
        if row.get("case_id") in test_ids and row.get("status") == "ok" and row.get("dice")
    ]
    summaries: list[dict[str, object]] = []
    for model in sorted({row["model"] for row in metric_rows}):
        model_rows = [row for row in metric_rows if row["model"] == model]
        for protocol in sorted({row["protocol"] for row in model_rows}):
            vals = [float(row["dice"]) for row in model_rows if row["protocol"] == protocol]
            if vals:
                summaries.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "rows": len(vals),
                        "mean_dice": round(statistics.mean(vals), 6),
                        "median_dice": round(statistics.median(vals), 6),
                    }
                )
    summaries.sort(key=lambda row: (str(row["model"]), -float(row["mean_dice"])))
    best = []
    for model in sorted({row["model"] for row in summaries}):
        best.append(max([row for row in summaries if row["model"] == model], key=lambda row: float(row["mean_dice"])))
    return summaries, best


def parse_training_log() -> dict[str, object]:
    log_files = sorted(RESULT_ROOT.glob("training_log*.txt"))
    if not log_files:
        return {}
    log_path = log_files[-1]
    log = log_path.read_text(encoding="utf-8", errors="replace")
    epoch_re = re.compile(
        r"Epoch (\d+).*?train_loss ([\-0-9.]+).*?val_loss ([\-0-9.]+).*?Pseudo dice \[np\.float32\(([0-9.]+)\)\]",
        re.S,
    )
    epochs = [
        {
            "epoch": int(match.group(1)),
            "train_loss": float(match.group(2)),
            "val_loss": float(match.group(3)),
            "pseudo_dice": float(match.group(4)),
        }
        for match in epoch_re.finditer(log)
    ]
    ema_records = []
    current_epoch = None
    for line in log.splitlines():
        epoch_match = re.search(r"Epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        ema_match = re.search(r"New best EMA pseudo Dice: ([0-9.]+)", line)
        if ema_match and current_epoch is not None:
            ema_records.append({"epoch": current_epoch, "ema_pseudo_dice": float(ema_match.group(1))})
    validation_summary_path = RESULT_ROOT / "validation" / "summary.json"
    validation_summary = read_json(validation_summary_path) if validation_summary_path.exists() else {}
    return {
        "log_file": str(log_path.relative_to(WORKSPACE_ROOT)),
        "num_epochs_logged": len(epochs),
        "first_epoch": epochs[0] if epochs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "best_instant_pseudo_dice": max(epochs, key=lambda row: row["pseudo_dice"]) if epochs else None,
        "best_ema": ema_records[-1] if ema_records else None,
        "internal_validation_dice": validation_summary.get("foreground_mean", {}).get("Dice"),
        "internal_validation_iou": validation_summary.get("foreground_mean", {}).get("IoU"),
    }


def md_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_report(
    test_rows: list[dict[str, object]],
    prompt_best: list[dict[str, object]],
    log_summary: dict[str, object],
) -> None:
    all_summary = summarize(test_rows)
    iou_summary = summarize(test_rows, "iou")
    bad_cases = sorted(test_rows, key=lambda row: float(row["dice"]))[:5]
    good_cases = sorted(test_rows, key=lambda row: float(row["dice"]), reverse=True)[:5]
    prompt_rows = [
        {
            "model": row["model"],
            "best_protocol": row["protocol"],
            "mean_dice_same20": row["mean_dice"],
            "median_dice": row["median_dice"],
        }
        for row in prompt_best
    ]
    nnunet_row = {
        "model": "nnU-Net 2D",
        "best_protocol": "supervised_250ep_best_ckpt",
        "mean_dice_same20": all_summary["mean"],
        "median_dice": all_summary["median"],
    }
    comparison_rows = sorted(prompt_rows + [nnunet_row], key=lambda row: float(row["mean_dice_same20"]), reverse=True)

    lines = [
        "# Phase3 nnU-Net 250-epoch 结果分析",
        "",
        "## 结论摘要",
        "",
        f"- 本次实际完成的是 `nnUNetTrainer_250epochs` 的 fold0 训练与 test prediction。",
        f"- 训练内部 validation Dice: `{log_summary.get('internal_validation_dice'):.4f}`；held-out test Dice: `{all_summary['mean']:.4f}`。",
        "- 当前 held-out test 的 20 个 case 全部来自 `MonuSeg`，因为原始转换脚本按 manifest 顺序做 80/20 split，前 50 个是 IDRID，后 50 个是 MonuSeg。这个结果可以作为 MonuSeg 小测试集参考，但还不能作为跨数据集最终强基线。",
        f"- 同 20 个 case 上，promptable SAM 系列最好约 `0.40 Dice`，nnU-Net 达到 `{all_summary['mean']:.4f}`，监督式模型明显更强。",
        "- 主要失败来自小目标/小前景 case 的面积过预测，后续 OCSAM-X 可以把 `前景面积校准` 和 `小目标置信抑制` 作为明确改进点。",
        "",
        "## 训练概况",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Trainer | nnUNetTrainer_250epochs |",
        f"| Fold | 0 |",
        f"| Logged epochs | {log_summary.get('num_epochs_logged')} |",
        f"| First epoch pseudo Dice | {log_summary.get('first_epoch', {}).get('pseudo_dice')} |",
        f"| Last epoch pseudo Dice | {log_summary.get('last_epoch', {}).get('pseudo_dice')} |",
        f"| Best instant pseudo Dice | {log_summary.get('best_instant_pseudo_dice', {}).get('pseudo_dice')} @ epoch {log_summary.get('best_instant_pseudo_dice', {}).get('epoch')} |",
        f"| Best EMA pseudo Dice | {log_summary.get('best_ema', {}).get('ema_pseudo_dice')} @ epoch {log_summary.get('best_ema', {}).get('epoch')} |",
        f"| Internal validation Dice | {log_summary.get('internal_validation_dice')} |",
        f"| Internal validation IoU | {log_summary.get('internal_validation_iou')} |",
        "",
        "## Held-out Test 结果",
        "",
        "| Split | Cases | Mean Dice | Median Dice | Min Dice | Max Dice | Mean IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| MonuSeg test | {all_summary['n']} | {all_summary['mean']} | {all_summary['median']} | {all_summary['min']} | {all_summary['max']} | {iou_summary['mean']} |",
        "",
        "## 同 20 Case 对照",
        "",
        *md_table(comparison_rows, ["model", "best_protocol", "mean_dice_same20", "median_dice"]),
        "",
        "## 失败样本",
        "",
        *md_table(
            [
                {
                    "case_id": row["case_id"],
                    "source_case_id": row["source_case_id"],
                    "dice": row["dice"],
                    "pred_gt_ratio": row["pred_gt_ratio"],
                    "pred_pixels": row["pred_pixels"],
                    "gt_pixels": row["gt_pixels"],
                }
                for row in bad_cases
            ],
            ["case_id", "source_case_id", "dice", "pred_gt_ratio", "pred_pixels", "gt_pixels"],
        ),
        "",
        "## 表现较好样本",
        "",
        *md_table(
            [
                {
                    "case_id": row["case_id"],
                    "source_case_id": row["source_case_id"],
                    "dice": row["dice"],
                    "pred_gt_ratio": row["pred_gt_ratio"],
                }
                for row in good_cases
            ],
            ["case_id", "source_case_id", "dice", "pred_gt_ratio"],
        ),
        "",
        "## 对 Phase3 的影响",
        "",
        "1. nnU-Net 已经证明：在 MonuSeg 这类监督式细胞核分割上，任务专用训练模型明显强于当前零训练 promptable SAM 系列。",
        "2. 这并不否定 OCSAM-X；它说明我们的论文 claim 不能写成全面超过监督式 SOTA，而应该强调 promptable、少交互、跨模型选择、面积校准和不确定性控制。",
        "3. 当前 split 不合理，必须重做 stratified split 或 dataset-wise protocol：IDRID 和 MonuSeg 都要有 train/val/test，而不是 test 只有 MonuSeg。",
        "4. 下一步最有价值的是做一个 `nnU-Net prediction -> OCSAM failure oracle` 分析：哪些 case SAM 低、nnU-Net 高，说明 prompt/模型选择还能补；哪些 case 两者都低，说明数据或任务定义需要清洗。",
        "",
        "## 产物路径",
        "",
        "- Test metrics: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/test_metrics.csv`",
        "- Same-20 promptable summary: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/promptable_same20_summary.csv`",
        "- Training curve summary: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/training_curve_summary.json`",
        "- Predictions: `runs/nnunet/predictions/Dataset901_OCSAMPhase3_fold0_250epochs`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_rows = evaluate_test_predictions()
    write_csv(OUT_ROOT / "test_metrics.csv", test_rows)
    summary = {
        "all": summarize(test_rows),
        "iou": summarize(test_rows, "iou"),
        "by_dataset": {dataset: summarize([row for row in test_rows if row["dataset"] == dataset]) for dataset in sorted({row["dataset"] for row in test_rows})},
    }
    (OUT_ROOT / "test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prompt_summary, prompt_best = summarize_promptable_same_cases(test_rows)
    write_csv(OUT_ROOT / "promptable_same20_summary.csv", prompt_summary)
    write_csv(OUT_ROOT / "promptable_same20_best_by_model.csv", prompt_best)
    log_summary = parse_training_log()
    (OUT_ROOT / "training_curve_summary.json").write_text(json.dumps(log_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(test_rows, prompt_best, log_summary)
    print(f"Wrote {OUT_ROOT / 'test_metrics.csv'}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
