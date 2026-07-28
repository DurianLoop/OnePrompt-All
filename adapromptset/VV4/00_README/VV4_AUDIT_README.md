# VV4 Audit README

**Generated**: 2026-07-28 12:16:57

---

## What is VV4?

VV4 is the comprehensive audit archive for the V4 research paper
("Reference Reliability Estimation for Visual Prompting Pipelines").
It contains the paper's original files, code, results, claim forensics,
reproduction results, salvage assessment, and final decisions.

---

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `00_README/` | This README and navigation guide |
| `01_V4_PAPER_ORIGINAL/` | Original V4 paper (LaTeX, PDF, Markdown, figures) |
| `02_CODE_ORIGINAL/` | Exact V4 execution code |
| `05_RESULTS_ORIGINAL/` | Original V4 results (JSON, CSV, predictions) |
| `06_REPORTS_AND_DECISIONS/` | Phase reports, claim registries, decisions |
| `07_CLAIM_FORENSICS/` | Claim extraction, hallucination audit, table/figure forensics |
| `08_REPRODUCTION/` | Artifact replay results |
| `09_V4_SALVAGE/` | Section salvage map, reconstruction decision |
| `10_REBUILT_PAPER/` | New paper outline and contributions |
| `99_MANIFESTS/` | Full file manifests with SHA256 |

---

## Quick Facts

| Field | Value |
|-------|-------|
| Canonical V4 | `paper/phaseU/full_draft.tex` |
| VV4 Final State | `VV4_PARTIALLY_RECONSTRUCTABLE` |
| Total files archived | 220 |
| Total size | 324.3 MB |
| Total claims | 92 |
| Phase AB status | PHASEAB_PAPER_CANDIDATE_READY (12/15 gates) |
| Recommended venue | Workshop / Benchmark track |

---

## Key Decisions

| Decision | Verdict |
|----------|---------|
| V4 reconstructable? | VV4_PARTIALLY_RECONSTRUCTABLE |
| RAMP retained? | NO — zero added value (Phase W) |
| RRN retained? | APPENDIX_ONLY — AUROC=0.573 < 0.65 |
| Anti-correlation retained? | PARTIALLY — SAM-B only, not universal |
| Theory retained? | YES — as hypotheses, not theorems |
| SAM-Med2D retained? | NO — checkpoint unloadable |
| New paper method | Prediction Consistency + coverage-based abstention |
| New paper title | "When References Disagree: Prediction Consistency for Selective Visual Prompting" |

---

## Key Files

| File | Description |
|------|-------------|
| `VV4_FINAL_AUDIT_REPORT.md` | Complete final audit report (30 questions answered) |
| `VV4_FINAL_AUDIT_REPORT.json` | Machine-readable audit report |
| `09_V4_SALVAGE/v4_reconstruction_decision.md` | WP12: V4 reconstruction decision |
| `09_V4_SALVAGE/v4_section_salvage_map.csv` | WP11: Section salvage map |
| `10_REBUILT_PAPER/outline/new_paper_outline.md` | WP13: New paper outline |
| `10_REBUILT_PAPER/outline/new_paper_contributions.md` | WP13: New paper contributions |
| `07_CLAIM_FORENSICS/extracted_claims/v4_claim_registry.csv` | WP3: All 92 V4 claims |
| `07_CLAIM_FORENSICS/hallucination_audit/hallucination_registry.csv` | WP10: A-I classification |
| `07_CLAIM_FORENSICS/hallucination_audit/hallucination_report.md` | WP5: High-risk claim audit |
| `08_REPRODUCTION/reproduction_registry.csv` | WP8: 60 reproduction results |
| `06_REPORTS_AND_DECISIONS/claim_registries/v4_vs_later_phases.csv` | WP9: Cross-phase comparison |
| `99_MANIFESTS/all_files_manifest.csv` | Full file manifest with SHA256 |

---

## Prohibited Terms in New Paper

- `universal` — anti-correlation is SAM-B specific
- `holdout-confirmed` — no confirmatory cohort
- `SOTA` / `state-of-the-art` — no proper comparison
- `training-free` — DINOv2 is trained
- `calibration-free` — IDRID ECE=0.477
- `identical conditions` — backbone mismatch
- `Theorem` / `Lemma` — use 'Geometric Hypotheses'

