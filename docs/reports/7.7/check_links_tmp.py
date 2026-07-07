from __future__ import annotations

import pathlib
import re


path = pathlib.Path("docs/reports/7.7/index.html")
text = path.read_text(encoding="utf-8")
base = path.parent
missing: list[str] = []
for url in re.findall(r'(?:src|href)="([^"]+)"', text):
    if url.startswith("#") or url.startswith("http"):
        continue
    if not (base / url).resolve().exists():
        missing.append(url)
print("missing", len(missing))
for item in missing:
    print(item)
raise SystemExit(1 if missing else 0)
