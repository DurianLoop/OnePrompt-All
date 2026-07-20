# Evidence-Supported Fixed-Rule Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale FSPL Markdown draft with an English conference-style paper whose claims, tables, figures, and limitations are exactly supported by the completed fixed spatial-layout experiments.

**Architecture:** The manuscript remains a single readable Markdown source in `paper/`, but a separate evidence manifest records every numerical source and every prohibited claim.  A small validator will compare the manuscript's reported fixed-layout numbers against authoritative JSON artifacts, preventing accidental reuse of stale policy or cross-domain results.  Generated tables and qualitative PNGs stay in `artifacts/` and are referenced rather than manually regenerated.

**Tech Stack:** Markdown, Python standard library JSON parsing, pytest, existing FSPL artifacts, frozen DINOv2 and SAM experimental outputs.

## Global Constraints

- Write only the bounded claim: at a matched budget of five positive and three negative points, fixed spatial diversification improves paired MoNuSAC Dice over matched Top-5 in the five completed replications.
- Use `artifacts/fixed_layout_effects.json` as the authority for the five mean gains, 95% paired bootstrap intervals, and win/tie/loss counts.
- Use `artifacts/stage18_dinol_samh_monusac/summary.json` and `case_action_utilities.csv` only to substantiate the SAM-H row and its 303 episodes per rule.
- Report the clean adaptive branch as an external negative: `artifacts/stage17_clean_adaptive_layout/summary.json` gives 0.0725 policy Dice on once-read IDRiD, below the 0.0958 Top-5 rule; do not retune or relabel it as a success.
- Do not claim cross-domain transfer, SOTA, clinical value, superiority to trained medical segmenters, or successful adaptive prompt selection.
- Exclude the old MoNuSeg table and the obsolete policy-oracle numbers from the main result section because they are not part of the current verified evidence gate.
- Use the three rendered predeclared panels in `artifacts/stage17_clean_adaptive_layout/qualitative_rendered/` without selecting cases by Dice or utility.
- Preserve Windows-compatible commands: `D:\conda\python.exe`, `PYTHONPATH=D:\SAM;D:\SAM\Matcher;D:\SAM\FSPL`, and `KMP_DUPLICATE_LIB_OK=TRUE` for torch-dependent checks.
- Do not edit any evaluation result, checkpoint, action CSV, or immutable experiment summary while preparing the manuscript.

---

## File Structure

- Modify: `paper/spatial_layout_matching_prompts_draft.md` — the conference-style manuscript source; replaces stale single-SAM and positive-policy prose with the verified scope.
- Create: `paper/fixed_rule_evidence_manifest.md` — a compact source-of-truth mapping from each manuscript number and figure to an artifact path.
- Create: `scripts/validate_fixed_rule_manuscript.py` — a deterministic check that extracts the five replication table rows from the manuscript and compares them to `fixed_layout_effects.json` at four-decimal reporting precision.
- Create: `tests/test_validate_fixed_rule_manuscript.py` — unit tests for compliant table-row extraction and mismatched-number rejection.
- Reference only: `artifacts/fixed_layout_effects.json`, `artifacts/evidence_tables.md`, `artifacts/stage17_clean_adaptive_layout/summary.json`, `artifacts/stage17_clean_adaptive_layout/qualitative_rendered/render_summary.json`, and the three panel PNGs.

### Task 1: Create the evidence manifest

**Files:**
- Create: `paper/fixed_rule_evidence_manifest.md`

**Interfaces:**
- Consumes: `artifacts/fixed_layout_effects.json`, `artifacts/evidence_tables.md`, `artifacts/stage17_clean_adaptive_layout/summary.json`, and `docs/2026-07-16-prompt-geometry-analysis.md`.
- Produces: a Markdown table with the exact source path, reported statistic, and permitted interpretation for every manuscript result.

- [x] **Step 1: Record the five fixed-layout rows from their JSON authority**

Create a table headed `Fixed-layout replication authority` containing these five labels: `DINOv2-L + SAM-B / 20260716`, `DINOv2-L + SAM-B / 20260717`, `DINOv2-B + SAM-B / 20260716`, `DINOv2-L + SAM-L / 20260716`, and `DINOv2-L + SAM-H / 20260716`.  Each row must point to `artifacts/fixed_layout_effects.json` and state the rule contrast as `spread5_n3 - top5_n3`.

- [x] **Step 2: Record the negative-policy and qualitative authorities**

Add one row for `stage17_clean_adaptive_layout/summary.json` stating that the selector is a negative external-transfer result, and one row for `qualitative_rendered/render_summary.json` stating that all three predeclared queries and fixed references are displayed without metric-based selection.

- [x] **Step 3: Verify the manifest against source artifacts**

Run:

```powershell
D:\conda\python.exe -c "import json; p=json.load(open('artifacts/fixed_layout_effects.json',encoding='utf-8')); print(len(p['replicates']))"
```

Expected: `5`.

Run:

```powershell
D:\conda\python.exe -c "import json; p=json.load(open('artifacts/stage17_clean_adaptive_layout/summary.json',encoding='utf-8')); print(p['method_status'])"
```

Expected: `clean-development adaptive-layout policy; external IDRiD test evaluated once after action set was locked`; no manuscript claim may reverse it.

### Task 2: Add a manuscript-number validator with tests

**Files:**
- Create: `scripts/validate_fixed_rule_manuscript.py`
- Create: `tests/test_validate_fixed_rule_manuscript.py`

**Interfaces:**
- Consumes: a Markdown manuscript path and `artifacts/fixed_layout_effects.json`.
- Produces: `validate_replication_table(manuscript_text: str, replicates: dict[str, dict[str, object]]) -> list[str]`, returning an empty list only when every row is present and its Top-5, spread, and gain values match four-decimal artifact reporting.

- [x] **Step 1: Write failing tests for a compliant and a stale SAM-H row**

Use a five-row Markdown fixture with `DINOv2-L + SAM-H / 20260716 | 0.4289 | 0.4439 | +0.0149` for the passing case.  In a second fixture replace `+0.0149` with `+0.0200` and assert that the returned errors mention the SAM-H label and `gain`.

- [x] **Step 2: Run the focused tests and confirm failure before implementation**

Run:

```powershell
D:\conda\python.exe -m pytest -q -p no:cacheprovider tests\test_validate_fixed_rule_manuscript.py
```

Expected: import failure because `validate_fixed_rule_manuscript` is not yet defined.

- [x] **Step 3: Implement deterministic extraction and comparison**

Implement a standard-library-only parser that recognizes the exact labels in Task 1, parses the three numeric fields from `| label | top5 | spread | gain |`, formats artifact values with `:.4f`, and returns actionable mismatch strings.  Add a CLI requiring `--manuscript-path` and optionally accepting `--effects-path`, exiting nonzero if the error list is nonempty.

- [x] **Step 4: Run the focused tests and the real-manuscript validator**

Run:

```powershell
D:\conda\python.exe -m pytest -q -p no:cacheprovider tests\test_validate_fixed_rule_manuscript.py
D:\conda\python.exe -B -m scripts.validate_fixed_rule_manuscript --manuscript-path paper\spatial_layout_matching_prompts_draft.md
```

Expected: all tests pass and the real manuscript prints a validation success line.

### Task 3: Rewrite the manuscript around verified fixed-layout evidence

**Files:**
- Modify: `paper/spatial_layout_matching_prompts_draft.md`

**Interfaces:**
- Consumes: the manifest from Task 1, the five-row effects report, the geometry analysis, and rendered qualitative artifacts.
- Produces: a coherent English conference-style Markdown paper with `Abstract`, `Introduction`, `Method`, `Experimental Protocol`, `Results`, `Discussion and Limitations`, and `Reproducibility` sections.

- [x] **Step 1: Replace stale scope statements**

Replace every statement that says experiments use only SAM ViT-B, that the offline policy is the method contribution, or that MoNuSeg cross-dataset results establish the paper's result.  State that the main comparison uses a frozen matcher and frozen SAM, with compatibility replications over DINOv2-B/L and SAM-B/L/H.

- [x] **Step 2: Write the evidence-bounded abstract and contributions**

The abstract must name the matched point budget, identify the primary DINOv2-L + SAM-B result as 0.4196 to 0.4526, state that all five completed paired intervals are positive, and report the adaptive selector as an external negative rather than a method success.  Contributions must describe deterministic spatial diversification, paired replication, and transparent negative-result reporting.

- [x] **Step 3: Replace the results section with four evidence-aligned subsections**

Write: (a) matched-budget primary result, (b) five-row matcher/segmenter replication table with all exact values, CIs, and win/tie/loss counts, (c) prompt-geometry mechanism result, and (d) clean adaptive-policy external negative.  Use only the numerical values in the manifest; no oracle or target-test model selection may appear.

- [x] **Step 4: Add the predeclared qualitative protocol and figure references**

Reference `IDRiD_68_panel.png`, `IDRiD_75_panel.png`, and `IDRiD_81_panel.png`.  Caption them as three predeclared queries, all fixed references, no metric-based selection.  Describe what is visible without inferring a score advantage from individual cases.

- [x] **Step 5: Tighten limitations and reproducibility**

Explicitly exclude cross-domain, clinical, SOTA, and adaptive-transfer claims.  Point to the fixed effects JSON, evidence table, stage-18 validator output, qualitative render summary, and deterministic test command.

### Task 4: Perform source, claim, and rendering verification

**Files:**
- Modify only if needed: `paper/spatial_layout_matching_prompts_draft.md` and `paper/fixed_rule_evidence_manifest.md`

**Interfaces:**
- Consumes: the rewritten manuscript, validator CLI, existing rendered panels, and artifact reports.
- Produces: a verified Markdown manuscript and a concise completion record in the evidence manifest.

- [x] **Step 1: Run the manuscript-number validator**

Run:

```powershell
D:\conda\python.exe -B -m scripts.validate_fixed_rule_manuscript --manuscript-path paper\spatial_layout_matching_prompts_draft.md
```

Expected: success with five validated replication rows.

- [x] **Step 2: Search for prohibited claims and stale numbers**

Run:

```powershell
rg -n -i "sota|clinical|cross-dataset improvement|adaptive.*transfer|oracle|0\.4640|0\.4444|MoNuSeg" paper\spatial_layout_matching_prompts_draft.md
```

Expected: no positive cross-domain/adaptive/SOTA claim; any remaining `MoNuSeg` mention must be a limitation stating that it is excluded from the verified main evidence.

- [x] **Step 3: Recheck the three qualitative panels**

Inspect `IDRiD_68_panel.png`, `IDRiD_75_panel.png`, and `IDRiD_81_panel.png`.  Confirm each presents three rows, one per fixed reference, with reference, ground truth, matched Top-5, and frozen-policy columns; do not edit or select panels by outcome.

- [x] **Step 4: Run the full focused regression suite**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda\python.exe -m pytest -q -p no:cacheprovider tests\test_statistics.py tests\test_analyze_fixed_layout_effects.py tests\test_qualitative.py tests\test_build_stage17_qualitative_panels.py tests\test_render_stage17_qualitative.py tests\test_build_evidence_tables.py tests\test_validate_fixed_rule_manuscript.py
```

Expected: every focused test passes.

## Self-Review

- Spec coverage: Task 1 establishes numerical authority; Task 2 prevents stale tables; Task 3 converts only that authority into manuscript text; Task 4 verifies exact numbers, forbidden claims, figures, and regression tests.
- Placeholder scan: all tasks specify their files, source artifacts, validation commands, and expected outcomes.
- Scope check: all tasks serve the single manuscript deliverable; no new segmentation method or post-hoc experiment is introduced.
