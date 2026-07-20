from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve(text: str) -> Path:
    path = Path(text)
    return (ROOT / path).resolve() if not path.is_absolute() else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MoNuSAC reference-query manifest for dense prompt experiments.")
    parser.add_argument("--processed-dir", default="code/datasets/MoNuSAC/processed")
    parser.add_argument("--support-case", default="train_000")
    parser.add_argument("--query-split", choices=["train", "test"], default="test")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--out", default="runs/research_direction/monusac_prompt_policy/monusac_screening_manifest.csv")
    args = parser.parse_args()
    root, out = resolve(args.processed_dir), resolve(args.out)
    support_image = root / "train" / "images" / f"{args.support_case}.png"
    support_mask = root / "train" / "masks" / f"{args.support_case}.png"
    query_images = sorted((root / args.query_split / "images").glob("*.png"))
    query_images = [path for path in query_images if path.stem != args.support_case]
    if args.case_limit > 0:
        query_images = query_images[: args.case_limit]
    rows = []
    for image in query_images:
        case_id = image.stem
        rows.append({
            "pilot_id": f"{args.support_case}__to__{case_id}",
            "dataset": "MoNuSAC",
            "domain": "pathology_nuclei_all",
            "support_case_id": args.support_case,
            "support_split": "train",
            "support_image_path": str(support_image.relative_to(ROOT)).replace("\\", "/"),
            "support_mask_path": str(support_mask.relative_to(ROOT)).replace("\\", "/"),
            "query_case_id": case_id,
            "query_split": args.query_split,
            "query_image_path": str(image.relative_to(ROOT)).replace("\\", "/"),
            "query_mask_path": str((root / args.query_split / "masks" / f"{case_id}.png").relative_to(ROOT)).replace("\\", "/"),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
