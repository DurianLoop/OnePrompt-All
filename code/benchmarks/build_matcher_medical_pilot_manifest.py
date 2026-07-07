from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import phase2_lite as p2


DEFAULT_OUT_DIR = p2.WORKSPACE_ROOT / "runs" / "research_direction" / "matcher_medical_pilot"


@dataclass(frozen=True)
class PilotPair:
    pilot_id: str
    dataset: str
    split: str
    support_case_id: str
    support_image_path: str
    support_mask_path: str
    support_box_xyxy: str
    support_width: int
    support_height: int
    support_target_area_ratio: float
    query_case_id: str
    query_image_path: str
    query_mask_path: str
    query_width: int
    query_height: int
    query_target_area_ratio: float
    target_hint: str


def target_hint(dataset: str) -> str:
    if dataset == "IDRID":
        return "retinal lesion foreground from IDRID combined mask"
    if dataset == "MonuSeg":
        return "nuclei foreground from MonuSeg mask"
    return "foreground target from support mask"


def support_box(record: p2.CaseRecord) -> str:
    return json.dumps(
        [record.bbox_x0, record.bbox_y0, record.bbox_x1, record.bbox_y1],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_records(dataset: str, rows: list[p2.CaseRecord], requested: int) -> list[PilotPair]:
    if requested <= 0 or len(rows) < 2:
        return []
    support = rows[0]
    selected_queries = rows[1 : requested + 1]
    pairs: list[PilotPair] = []
    for index, query in enumerate(selected_queries, start=1):
        pairs.append(
            PilotPair(
                pilot_id=f"{dataset.lower()}_{index:02d}",
                dataset=dataset,
                split=query.split,
                support_case_id=support.case_id,
                support_image_path=support.image_path,
                support_mask_path=support.mask_path,
                support_box_xyxy=support_box(support),
                support_width=support.width,
                support_height=support.height,
                support_target_area_ratio=support.target_area_ratio,
                query_case_id=query.case_id,
                query_image_path=query.image_path,
                query_mask_path=query.mask_path,
                query_width=query.width,
                query_height=query.height,
                query_target_area_ratio=query.target_area_ratio,
                target_hint=target_hint(dataset),
            )
        )
    return pairs


def collect_records(idrid_cases: int, monuseg_cases: int) -> list[PilotPair]:
    selected: list[PilotPair] = []
    idrid = [
        p2.build_case_record("IDRID", case_key, split, image_path, mask_path)
        for case_key, split, image_path, mask_path in p2.collect_idrid_cases(idrid_cases + 1)
    ]
    monuseg = [
        p2.build_case_record("MonuSeg", case_key, split, image_path, mask_path)
        for case_key, split, image_path, mask_path in p2.collect_monuseg_cases(monuseg_cases + 1)
    ]
    selected.extend(build_records("IDRID", idrid, idrid_cases))
    selected.extend(build_records("MonuSeg", monuseg, monuseg_cases))
    return selected


def write_outputs(out_dir: Path, records: list[PilotPair], args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "medical_matcher_pilot_manifest.csv"
    json_path = out_dir / "medical_matcher_pilot_manifest.json"
    summary_path = out_dir / "summary.md"

    fields = list(PilotPair.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    dataset_counts = {dataset: sum(1 for row in records if row.dataset == dataset) for dataset in sorted({r.dataset for r in records})}
    payload = {
        "case_count": len(records),
        "dataset_counts": dataset_counts,
        "requested": {
            "idrid_cases": args.idrid_cases,
            "monuseg_cases": args.monuseg_cases,
        },
        "manifest_csv": p2.rel(csv_path),
        "records": [asdict(row) for row in records],
        "usage": {
            "stage1": "Use support_box_xyxy/support_mask_path to extract a medical target prototype, then match query_image_path.",
            "stage2": "Send top-k query matched points to SAM-family point prompt segmentation.",
            "stage3": "Compare query predictions with query_mask_path using Dice/IoU and record accept/reject feedback.",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Matcher Medical Pilot Manifest",
        "",
        f"- Cases: `{len(records)}`",
        f"- Dataset counts: `{dataset_counts}`",
        f"- CSV: `{p2.rel(csv_path)}`",
        f"- JSON: `{p2.rel(json_path)}`",
        "",
        "## 为什么这是下一步",
        "",
        "这个 manifest 把自然场景里已经跑通的 feedback/learning matcher 迁移到医学图像：每一行都有 support image/mask/reference box 和 query image/mask。support 端负责定义“我要找什么”，query 端负责验证 matcher 是否能找到相似目标。",
        "",
        "## 字段解释",
        "",
        "- `support_*`: 参考图、参考 mask 和由参考 mask 外接得到的 reference box。",
        "- `query_*`: 需要被匹配/分割的目标图和真值 mask。",
        "- `target_hint`: 给实验记录看的目标语义，不作为模型输入的唯一依据。",
        "",
        "## 下一条命令",
        "",
        "`python -B code/benchmarks/build_matcher_medical_pilot_manifest.py --out-dir runs/research_direction/matcher_medical_pilot --idrid-cases 3 --monuseg-cases 2`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 5-case medical support/query manifest for Matcher pilot experiments.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--idrid-cases", type=int, default=3)
    parser.add_argument("--monuseg-cases", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = p2.WORKSPACE_ROOT / out_dir
    records = collect_records(args.idrid_cases, args.monuseg_cases)
    payload = write_outputs(out_dir.resolve(), records, args)
    print(f"Wrote {payload['case_count']} medical pilot rows to {payload['manifest_csv']}")


if __name__ == "__main__":
    main()
