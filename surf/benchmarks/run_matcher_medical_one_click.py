from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(WORKSPACE_ROOT):
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    return str(resolved)


def html_rel(path_text: str, html_dir: Path) -> str:
    path = resolve_path(path_text)
    try:
        return str(path.relative_to(html_dir.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_command(command: list[str], *, cwd: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def command_text(command: list[str]) -> str:
    return " ".join(command)


def add_step(steps: list[dict[str, Any]], name: str, commands: list[list[str]]) -> None:
    steps.append(
        {
            "name": name,
            "command_count": len(commands),
            "commands": [command_text(command) for command in commands],
        }
    )


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "medical_matcher_pilot_manifest.csv"


def build_steps(args: argparse.Namespace, rows: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    out_dir = resolve_path(args.out_dir)
    manifest = manifest_path(out_dir)
    run_dir = out_dir / "run"
    eval_dir = out_dir / "eval"
    casebook_dir = out_dir / "casebook"
    steps: list[dict[str, Any]] = []

    add_step(
        steps,
        "build_manifest",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/build_matcher_medical_pilot_manifest.py",
            "--out-dir",
            rel(out_dir),
            "--idrid-cases",
            str(args.idrid_cases),
            "--monuseg-cases",
            str(args.monuseg_cases),
        ]],
    )
    add_step(
        steps,
        "run_stage1",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/run_matcher_medical_pilot.py",
            "--manifest",
            rel(manifest),
            "--out-dir",
            rel(run_dir),
            "--run-stage1",
        ]],
    )
    add_step(
        steps,
        "run_stage2_sampling",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/run_matcher_medical_pilot.py",
            "--manifest",
            rel(manifest),
            "--out-dir",
            rel(run_dir),
            "--run-stage2-sampling",
            "--cluster-eps",
            str(args.cluster_eps),
            "--min-samples",
            str(args.min_samples),
            "--sample-sizes",
            args.sample_sizes,
            "--samples-per-cluster",
            str(args.samples_per_cluster),
        ]],
    )
    proposal_commands: list[list[str]] = []
    stage3_commands: list[list[str]] = []
    for row in rows or []:
        case_id = row["pilot_id"]
        proposal_commands.append([
            sys.executable,
            "-B",
            "code/benchmarks/generate_stage2_sam_proposals.py",
            "--stage2-dir",
            rel(run_dir / case_id / "stage2_sampling"),
            "--checkpoint",
            args.sam_checkpoint,
            "--model-type",
            args.sam_model_type,
            "--device",
            args.device,
            "--max-groups",
            str(args.max_groups),
            "--out-dir",
            rel(run_dir / case_id / "stage2_proposals"),
        ])
        stage3_commands.append([
            sys.executable,
            "-B",
            "code/benchmarks/filter_stage3_proposals.py",
            "--proposals-dir",
            rel(run_dir / case_id / "stage2_proposals"),
            "--min-score",
            str(args.min_score),
            "--min-area-pct",
            str(args.min_area_pct),
            "--max-area-pct",
            str(args.max_area_pct),
            "--max-bbox-area-pct",
            str(args.max_bbox_area_pct),
            "--min-cluster-coverage",
            str(args.min_cluster_coverage),
            "--min-prompt-coverage",
            str(args.min_prompt_coverage),
            "--nms-iou",
            str(args.nms_iou),
            "--target-area-pct",
            str(args.target_area_pct),
            "--target-bbox-area-pct",
            str(args.target_bbox_area_pct),
            "--out-dir",
            rel(run_dir / case_id / "stage3_filtering"),
        ])
    add_step(steps, "run_sam_proposals", proposal_commands)
    add_step(steps, "run_stage3_filtering", stage3_commands)
    add_step(
        steps,
        "run_eval",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/evaluate_matcher_medical_pilot.py",
            "--manifest",
            rel(manifest),
            "--run-dir",
            rel(run_dir),
            "--out-dir",
            rel(eval_dir),
        ]],
    )
    add_step(
        steps,
        "run_casebook",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/build_matcher_medical_casebook.py",
            "--manifest",
            rel(manifest),
            "--run-dir",
            rel(run_dir),
            "--out-dir",
            rel(casebook_dir),
            "--positive-dice",
            str(args.positive_dice),
        ]],
    )
    sensitivity_commands: list[list[str]] = []
    for value in parse_float_list(args.sensitivity_dice):
        suffix = f"dice{int(round(value * 1000)):03d}"
        sensitivity_commands.append([
            sys.executable,
            "-B",
            "code/benchmarks/build_matcher_medical_casebook.py",
            "--manifest",
            rel(manifest),
            "--run-dir",
            rel(run_dir),
            "--out-dir",
            rel(out_dir / f"casebook_{suffix}"),
            "--positive-dice",
            str(value),
        ])
    add_step(steps, "run_casebook_sensitivity", sensitivity_commands)
    add_step(
        steps,
        "build_review_gallery",
        [[
            sys.executable,
            "-B",
            "code/benchmarks/run_matcher_medical_one_click.py",
            "--out-dir",
            rel(out_dir),
            "--build-gallery-only",
        ]],
    )
    return steps


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return values or [0.15, 0.30, 0.50]


def write_plan(out_dir: Path, args: argparse.Namespace, steps: list[dict[str, Any]], executed: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_case_count": int(args.idrid_cases) + int(args.monuseg_cases),
        "idrid_cases": int(args.idrid_cases),
        "monuseg_cases": int(args.monuseg_cases),
        "dry_run": bool(args.dry_run),
        "steps": steps,
        "executed": executed,
        "artifacts": {
            "manifest": rel(manifest_path(out_dir)),
            "run_dir": rel(out_dir / "run"),
            "eval_dir": rel(out_dir / "eval"),
            "casebook_dir": rel(out_dir / "casebook"),
            "review_gallery": rel(out_dir / "review_gallery.html"),
            "summary": rel(out_dir / "one_click_summary.md"),
        },
    }
    (out_dir / "one_click_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Matcher Medical One-Click Plan",
        "",
        f"- Requested cases: `{payload['requested_case_count']}`",
        f"- IDRID / MonuSeg: `{args.idrid_cases}` / `{args.monuseg_cases}`",
        f"- Dry run: `{args.dry_run}`",
        "",
        "## Steps",
        "",
    ]
    for step in steps:
        lines.append(f"- `{step['name']}`: {step['command_count']} command(s)")
    if executed:
        lines.extend(["", "## Executed", ""])
        for item in executed:
            lines.append(f"- `{item['step']}` returncode `{item['returncode']}`")
    (out_dir / "one_click_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_steps(steps: list[dict[str, Any]], *, out_dir: Path) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    for step in steps:
        for command_text_value in step["commands"]:
            command = split_stored_command(command_text_value)
            result = run_command(command)
            item = {"step": step["name"], **result}
            executed.append(item)
            write_execution_log(out_dir, executed)
            if result["returncode"] != 0:
                raise RuntimeError(f"Step {step['name']} failed: {result['stderr_tail']}")
    return executed


def split_stored_command(text: str) -> list[str]:
    # Commands are generated by this script and only use space-free option values except paths;
    # regenerate via JSON would be cleaner, but this keeps the plan readable and reproducible.
    import shlex

    return shlex.split(text, posix=False)


def write_execution_log(out_dir: Path, executed: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "one_click_execution_log.json").write_text(json.dumps(executed, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command_list(name: str, commands: list[list[str]], out_dir: Path, executed: list[dict[str, Any]]) -> None:
    for command in commands:
        result = run_command(command)
        item = {"step": name, **result}
        executed.append(item)
        write_execution_log(out_dir, executed)
        if result["returncode"] != 0:
            raise RuntimeError(f"Step {name} failed: {result['stderr_tail']}")


def casebook_rows(out_dir: Path) -> list[dict[str, str]]:
    path = out_dir / "casebook" / "medical_casebook_rows.csv"
    return read_csv(path) if path.exists() else []


def proposal_overlay_path(row: dict[str, str]) -> str:
    if row.get("overlay_path"):
        return row["overlay_path"]
    mask_path = row.get("mask_path", "")
    if not mask_path:
        return ""
    return mask_path.replace("proposal_masks", "proposal_overlays")


def safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_review_gallery(out_dir: Path) -> Path:
    rows = casebook_rows(out_dir)
    html_path = out_dir / "review_gallery.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["case_name"], []).append(row)
    accept_count = sum(1 for row in rows if str(row.get("label")) == "1")
    kept_count = sum(1 for row in rows if str(row.get("stage3_keep")) == "1")
    high_drop_count = sum(1 for row in rows if str(row.get("label")) == "1" and str(row.get("stage3_keep")) != "1")
    cards: list[str] = []
    for case_name, case_rows in grouped.items():
        case_rows.sort(key=lambda item: int(item["group_id"]))
        items = []
        for row in case_rows:
            overlay_path = proposal_overlay_path(row)
            overlay = html_rel(overlay_path, html_path.parent) if overlay_path else ""
            label = "accept" if str(row.get("label")) == "1" else "reject"
            stage3 = "kept" if str(row.get("stage3_keep")) == "1" else "drop"
            dice = safe_float(row, "dice")
            iou = safe_float(row, "iou")
            score = safe_float(row, "feedback_learning_score")
            stage3_conflict = "conflict" if label == "accept" and stage3 == "drop" else ""
            items.append(
                f"""
                <article class="proposal {stage3_conflict}" data-label="{html.escape(label)}" data-stage3="{html.escape(stage3)}" data-dataset="{html.escape(row.get('dataset', ''))}">
                  <button class="image-button" type="button" data-full-src="{html.escape(overlay)}" data-title="{html.escape(case_name)} · group {html.escape(row['group_id'])}">
                    <img data-proposal-image="1" loading="lazy" src="{html.escape(overlay)}" alt="{html.escape(case_name)} group {html.escape(row['group_id'])}">
                  </button>
                  <div class="meta">
                    <strong>{html.escape(case_name)} · group {html.escape(row['group_id'])}</strong>
                    <span>{html.escape(row.get('dataset', ''))} · Dice {dice:.3f} · IoU {iou:.3f}</span>
                    <span>score {score:.3f} · label <b class="{label}">{label}</b> · Stage3 {stage3}</span>
                    <select data-case="{html.escape(case_name)}" data-group="{html.escape(row['group_id'])}">
                      <option value="">review pending</option>
                      <option value="accept">accept</option>
                      <option value="reject">reject</option>
                      <option value="uncertain">uncertain</option>
                    </select>
                  </div>
                </article>
                """
            )
        best = max(case_rows, key=lambda item: safe_float(item, "dice"))
        cards.append(
            f"""
            <section class="case">
              <div class="case-head">
                <h2>{html.escape(case_name)}</h2>
                <span>{html.escape(case_rows[0].get('dataset', ''))}</span>
                <span>best Dice {safe_float(best, 'dice'):.3f} · group {html.escape(best['group_id'])}</span>
              </div>
              <div class="grid">{''.join(items)}</div>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matcher Medical Proposal Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --paper: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #d8dee8;
      --blue: #2458a6;
      --green: #16794c;
      --red: #b23b3b;
      --amber: #a15c00;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 "Segoe UI", Arial, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; background: rgba(255,255,255,.94); border-bottom: 1px solid var(--line); padding: 14px 22px; backdrop-filter: blur(8px); }}
    h1 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .metric {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; min-width: 122px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric b {{ display: block; font-size: 20px; line-height: 1.15; margin-top: 2px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; align-items: center; }}
    button {{ border: 1px solid var(--line); background: var(--paper); color: var(--ink); padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    button.active {{ border-color: var(--blue); color: var(--blue); background: #eef4ff; }}
    main {{ padding: 18px 22px 40px; }}
    .case {{ margin-bottom: 26px; }}
    .case-head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; margin: 0 0 10px; }}
    h2 {{ font-size: 16px; margin: 0; }}
    .case-head span {{ color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 12px; }}
    .proposal {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(18, 26, 38, .04); }}
    .proposal.conflict {{ border-color: #d6a64f; }}
    .image-button {{ width: 100%; display: block; border: 0; border-radius: 0; padding: 0; background: #edf1f5; }}
    .image-button:focus-visible {{ outline: 3px solid #8ab4f8; outline-offset: -3px; }}
    .proposal img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #edf1f5; transition: transform .18s ease; }}
    .image-button:hover img {{ transform: scale(1.015); }}
    .meta {{ display: grid; gap: 4px; padding: 10px; }}
    .meta span {{ color: var(--muted); }}
    .accept {{ color: var(--green); }}
    .reject {{ color: var(--red); }}
    select {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px; background: #fff; color: var(--ink); }}
    .hidden {{ display: none; }}
    textarea {{ width: min(980px, 100%); min-height: 160px; margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-family: Consolas, monospace; }}
    .lightbox {{ position: fixed; inset: 0; z-index: 10; display: none; place-items: center; background: rgba(9, 15, 24, .82); padding: 24px; }}
    .lightbox.open {{ display: grid; }}
    .lightbox-panel {{ width: min(1280px, 96vw); max-height: 94vh; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 24px 70px rgba(0,0,0,.32); }}
    .lightbox-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .lightbox-head strong {{ font-size: 14px; }}
    .lightbox img {{ width: 100%; max-height: calc(94vh - 54px); object-fit: contain; display: block; background: #eef2f6; }}
  </style>
</head>
<body>
  <header>
    <h1>Matcher Medical Proposal Review</h1>
    <div class="summary">
      <div class="metric"><span>cases</span><b>{len(grouped)}</b></div>
      <div class="metric"><span>proposals</span><b>{len(rows)}</b></div>
      <div class="metric"><span>accept labels</span><b>{accept_count}</b></div>
      <div class="metric"><span>Stage3 kept</span><b>{kept_count}</b></div>
      <div class="metric"><span>accept but dropped</span><b>{high_drop_count}</b></div>
    </div>
    <div class="toolbar">
      <button class="active" data-filter="all">All</button>
      <button data-filter="accept">Pseudo accept</button>
      <button data-filter="reject">Pseudo reject</button>
      <button data-filter="kept">Stage3 kept</button>
      <button data-filter="drop">Stage3 dropped</button>
      <button data-filter="IDRID">IDRID</button>
      <button data-filter="MonuSeg">MonuSeg</button>
      <button id="exportBtn">Export review JSON</button>
    </div>
    <textarea id="exportBox" class="hidden" placeholder="exported labels will appear here"></textarea>
  </header>
  <main>
    {''.join(cards)}
  </main>
  <div class="lightbox" id="lightbox" aria-hidden="true">
    <div class="lightbox-panel" role="dialog" aria-modal="true" aria-labelledby="lightboxTitle">
      <div class="lightbox-head">
        <strong id="lightboxTitle">proposal</strong>
        <button type="button" id="lightboxClose">Close</button>
      </div>
      <img id="lightboxImage" alt="">
    </div>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll('[data-filter]'));
    const proposals = Array.from(document.querySelectorAll('.proposal'));
    buttons.forEach(button => button.addEventListener('click', () => {{
      buttons.forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      const filter = button.dataset.filter;
      proposals.forEach(card => {{
        const visible = filter === 'all' || card.dataset.label === filter || card.dataset.stage3 === filter || card.dataset.dataset === filter;
        card.classList.toggle('hidden', !visible);
      }});
    }}));
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxTitle = document.getElementById('lightboxTitle');
    function openLightbox(src, title) {{
      lightboxImage.src = src;
      lightboxImage.alt = title;
      lightboxTitle.textContent = title;
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
    }}
    function closeLightbox() {{
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      lightboxImage.removeAttribute('src');
    }}
    Array.from(document.querySelectorAll('.image-button')).forEach(button => {{
      button.addEventListener('click', () => openLightbox(button.dataset.fullSrc, button.dataset.title));
    }});
    document.getElementById('lightboxClose').addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', event => {{
      if (event.target === lightbox) closeLightbox();
    }});
    document.addEventListener('keydown', event => {{
      if (event.key === 'Escape') closeLightbox();
    }});
    document.getElementById('exportBtn').addEventListener('click', () => {{
      const labels = Array.from(document.querySelectorAll('select')).map(select => ({{
        case_name: select.dataset.case,
        group_id: Number(select.dataset.group),
        review_label: select.value || 'pending'
      }})).filter(item => item.review_label !== 'pending');
      const box = document.getElementById('exportBox');
      box.value = JSON.stringify({{labels}}, null, 2);
      box.classList.remove('hidden');
      box.focus();
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def execute_pipeline(args: argparse.Namespace) -> None:
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    executed: list[dict[str, Any]] = []

    build_command = build_steps(args, rows=[])[0]["commands"][0]
    run_command_list("build_manifest", [split_stored_command(build_command)], out_dir, executed)
    rows = read_csv(manifest_path(out_dir))
    steps = build_steps(args, rows=rows)

    for step in steps[1:]:
        if step["name"] == "build_review_gallery":
            gallery = build_review_gallery(out_dir)
            executed.append({"step": "build_review_gallery", "returncode": 0, "output": rel(gallery)})
            write_execution_log(out_dir, executed)
            continue
        run_command_list(step["name"], [split_stored_command(cmd) for cmd in step["commands"]], out_dir, executed)
    write_plan(out_dir, args, steps, executed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click expanded medical Matcher pipeline.")
    parser.add_argument("--out-dir", default="runs/research_direction/matcher_medical_expanded")
    parser.add_argument("--idrid-cases", type=int, default=12)
    parser.add_argument("--monuseg-cases", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-gallery-only", action="store_true")
    parser.add_argument("--cluster-eps", type=float, default=120.0)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--sample-sizes", default="1,2,3")
    parser.add_argument("--samples-per-cluster", type=int, default=4)
    parser.add_argument("--sam-checkpoint", default="assets/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-groups", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=0.80)
    parser.add_argument("--min-area-pct", type=float, default=0.01)
    parser.add_argument("--max-area-pct", type=float, default=80.0)
    parser.add_argument("--max-bbox-area-pct", type=float, default=80.0)
    parser.add_argument("--min-cluster-coverage", type=float, default=0.20)
    parser.add_argument("--min-prompt-coverage", type=float, default=1.0)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--target-area-pct", type=float, default=25.0)
    parser.add_argument("--target-bbox-area-pct", type=float, default=30.0)
    parser.add_argument("--positive-dice", type=float, default=0.30)
    parser.add_argument("--sensitivity-dice", default="0.15,0.30,0.50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    if args.build_gallery_only:
        gallery = build_review_gallery(out_dir)
        print(f"Wrote {rel(gallery)}")
        return
    if args.dry_run:
        steps = build_steps(args, rows=[])
        write_plan(out_dir, args, steps, executed=[])
        print(f"Wrote dry-run plan to {rel(out_dir / 'one_click_plan.json')}")
        return
    execute_pipeline(args)
    print(f"Wrote one-click outputs to {rel(out_dir)}")


if __name__ == "__main__":
    main()
