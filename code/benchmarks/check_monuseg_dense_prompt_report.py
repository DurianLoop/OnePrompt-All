from __future__ import annotations

import csv
import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    report_dir = WORKSPACE_ROOT / "docs" / "reports" / "monuseg_dense_prompt_t1_t5"
    run_dir = WORKSPACE_ROOT / "runs" / "research_direction" / "monuseg_dense_prompt_t1_t5"
    html_path = report_dir / "index.html"
    candidates_path = run_dir / "candidates.csv"
    html = html_path.read_text(encoding="utf-8")
    srcs = re.findall(r'src="([^"]+)"', html)
    missing = [src for src in srcs if not (report_dir / src).resolve().exists()]
    rows = list(csv.DictReader(candidates_path.open("r", newline="", encoding="utf-8")))
    counts = {
        "round1": sum(row["round_name"] == "round1" for row in rows),
        "round2": sum(row["round_name"] == "round2" for row in rows),
        "round3": sum(row["round_name"] == "round3" for row in rows),
    }
    print(f"html_images={len(srcs)}")
    print(f"missing_images={len(missing)}")
    print(f"candidate_rows={len(rows)}")
    print(f"round1_rows={counts['round1']}")
    print(f"round2_rows={counts['round2']}")
    print(f"round3_rows={counts['round3']}")
    if missing:
        print("missing_sample=" + "; ".join(missing[:5]))
        raise SystemExit(1)
    if len(rows) != 340 or counts != {"round1": 140, "round2": 80, "round3": 120}:
        raise SystemExit(1)
    print("check=ok")


if __name__ == "__main__":
    main()
