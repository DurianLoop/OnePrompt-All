from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / ".cache" / "matcher_p0_p4_progress_check"


def main() -> None:
    script = ROOT / "code" / "benchmarks" / "build_matcher_p0_p4_progress.py"
    expanded_root = ROOT / "runs" / "research_direction" / "matcher_medical_expanded"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--expanded-root",
            str(expanded_root),
            "--out-dir",
            str(OUT_DIR),
        ],
        cwd=ROOT,
        check=True,
    )

    payload_path = OUT_DIR / "p0_p4_progress.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for key in [
        "p0_review_package",
        "p1_domain_policy",
        "p2_feedback_simulation",
        "p3_expansion_plan",
        "p4_model_upgrade",
        "point_details",
    ]:
        assert key in payload, key

    assert payload["p0_review_package"]["pending_review_count"] > 0
    assert payload["p1_domain_policy"]["IDRID"]["learned_metrics"]["f1"] >= 0
    assert payload["p2_feedback_simulation"]["rounds"]
    assert payload["p3_expansion_plan"]["planned_cases"] >= 20
    assert payload["p4_model_upgrade"]["loco"]["scalar_threshold"]["f1"] >= 0

    detail_by_case = {item["case_name"]: item for item in payload["point_details"]}
    assert "idrid_11" in detail_by_case
    assert "monuseg_04" in detail_by_case
    assert len(detail_by_case["idrid_11"]["top_points"]) >= 5
    assert detail_by_case["idrid_11"]["reference_image"]
    assert detail_by_case["idrid_11"]["query_image"]

    for name in [
        "summary.md",
        "point_details.csv",
        "domain_policy_summary.csv",
        "model_upgrade_loco.csv",
    ]:
        assert (OUT_DIR / name).exists(), name

    print(f"[ok] P0-P4 progress package generated at {OUT_DIR}")


if __name__ == "__main__":
    main()
