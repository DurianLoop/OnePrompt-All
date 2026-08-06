from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PHASE3_METRICS = WORKSPACE_ROOT / "runs" / "phase3_full" / "metrics" / "phase2_lite_metrics.csv"
MEDSA_METRICS = WORKSPACE_ROOT / "runs" / "phase3_medsa" / "metrics" / "phase2_lite_metrics.csv"
OUT_ROOT = WORKSPACE_ROOT / "runs" / "phase3_full_with_medsa"
REPORT = WORKSPACE_ROOT / "docs" / "phase3_medsa_adapter_results.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean(vals: list[float]) -> float:
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def summarize(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec.update({"rows": len(items), "mean_dice": round(mean([fnum(r["dice"]) for r in items]), 4)})
        out.append(rec)
    return out


def write_report(rows: list[dict[str, str]]) -> None:
    medsa = [r for r in rows if r.get("model", "").startswith("medsa_") and r.get("status") == "ok"]
    summary_model = summarize(rows, ["model"])
    summary_protocol = summarize(medsa, ["model", "dataset", "protocol"])
    lines = [
        "# Phase3 Med-SA Adapter 主表结果",
        "",
        "本报告把已下载的两个 Medical-SAM-Adapter Zoo 权重真正跑进了 Phase3 100-case 主表。",
        "",
        "- `medsa_opticdisc`: `OpticDisc_Fundus_SAM_1024.pth`",
        "- `medsa_microscope`: `White Blood Cell_MicroScope_sam_1024.pth`",
        "- 评估 prompt: `gt_box`, `auto_box`, `auto_medical_box_v1`, `auto_medical_point_v1`",
        "",
        "## 全模型均值",
        "",
        "| Model | Rows | Mean Dice |",
        "| --- | ---: | ---: |",
    ]
    for row in summary_model:
        lines.append(f"| {row['model']} | {row['rows']} | {row['mean_dice']} |")
    lines.extend(["", "## Med-SA 细分结果", "", "| Model | Dataset | Prompt | Rows | Mean Dice |", "| --- | --- | --- | ---: | ---: |"])
    for row in summary_protocol:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['protocol']} | {row['rows']} | {row['mean_dice']} |")
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "两个 adapter 已经完成工程接入；但它们来自 OpticDisc/WBC 任务，和 IDRID lesion、MonuSeg nuclei 并非完全同分布，所以当前只能作为 domain-near Med-SA baseline，而不是最终强结论。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_csv(PHASE3_METRICS) + read_csv(MEDSA_METRICS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_ROOT / "metrics_with_medsa.csv", rows)
    write_csv(OUT_ROOT / "summary_by_model.csv", summarize(rows, ["model"]))
    write_csv(OUT_ROOT / "summary_by_model_dataset_protocol.csv", summarize(rows, ["model", "dataset", "protocol"]))
    write_report(rows)
    print(f"Wrote {OUT_ROOT / 'metrics_with_medsa.csv'}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
