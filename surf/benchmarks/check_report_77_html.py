from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "reports" / "7.7" / "index.html"


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    HTMLParser().feed(text)
    assert "给老师" not in text
    assert "老师方向" not in text
    assert 'id="background"' in text
    assert 'id="closed-loop"' in text
    assert 'id="domain-policy"' in text
    assert 'id="lightbox"' in text
    assert "openLightbox" in text
    assert "assets/idrid_11_match_composite.png" in text
    assert "assets/monuseg_04_match_composite.png" in text
    assert "assets/idrid_11_detail_strong.png" in text
    assert "assets/monuseg_04_detail_strong.png" in text
    assert "assets/idrid_11_separated_evidence.png" in text
    assert "assets/monuseg_04_separated_evidence.png" in text
    assert "assets/case_split_50_qualitative_grid.png" in text
    assert "train_domain_proposal_policy.py" in text
    assert "train_case_split_proposal_policy.py" in text
    assert "build_case_split_qualitative_figures.py" in text
    assert "case_split_50_summary.csv" in text
    assert "case_split_50_predictions.csv" in text
    assert "overall_5fold" in text
    assert "domain_5fold" in text
    assert "idrid_11" in text
    assert "monuseg_04" in text

    missing: list[str] = []
    anchors = 0
    for raw in re.findall(r'(?:src|href)="([^"]+)"', text):
        value = html.unescape(raw)
        if value.startswith("#"):
            anchors += 1
            continue
        if "://" in value or value.startswith("mailto:"):
            continue
        path = (REPORT.parent / value).resolve()
        if not path.exists():
            missing.append(value)
    if missing:
        raise AssertionError("Missing report links:\n" + "\n".join(missing))
    print(f"[ok] report html parsed with {anchors} anchors and no missing local links")


if __name__ == "__main__":
    main()
