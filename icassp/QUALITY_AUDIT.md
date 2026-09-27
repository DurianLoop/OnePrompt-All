# Quality audit — 2026-08-09

## Evidence integrity

`D:\conda\python.exe -B icassp\scripts\verify_evidence.py` passed all 16 checks. The generated report is `evidence/verification_report.json`.

## FoB transfer diagnostic

The copied FoB source and all compatibility changes are recorded in `baselines/fob_source/COMPATIBILITY.md`. On the materialized public MoNuSeg source-train images, a frozen fold-0 run trained 1,000 steps on 115 train-only episodes, selected from steps 250/500/750/1000 using 30 inner-validation episodes, and evaluated the selected step 250 exactly once on 40 outer-test episodes. Its Dice was 0.0047703. Query masks were loaded only after prompt creation and SAM-H prediction to calculate metrics. This is a reproducible negative-transfer diagnostic; because it is one fold and uses SAM ViT-H rather than the paper's five-fold DINOv2-L + SAM ViT-B protocol, it is not a direct comparative baseline.

## PanNuke zero-shot cohort

The PanNuke zero-shot protocol was frozen before download. Its materialization script produced 2,540 non-empty fold-1 query image/union-mask pairs and five SHA-256-selected non-empty fold-2 supports, all at 256 by 256. A one-pair frozen DINOv2-L + SAM-B interface smoke passed before the full run. The complete frozen cohort ran all 12,700 support--query episodes and generated raw N0/N5 records before gate loading. Fixed N5 degraded N0 by 0.0127 Dice (95% CI [-0.0143, -0.0112]); the five-fold majority gate improved N0 by 0.0047 (95% CI [0.0040, 0.0054]). The saved gates had mixed sklearn origins, so folds 0--1 were deserialized only with 1.7.2 and folds 2--4 only with 1.9.0; their complete outputs were then deterministically joined. Gate inputs exclude Dice and ground-truth masks. This supports only a field-level, zero-shot statement because the public PanNuke release exposes no patient identifiers. Evidence is in `evidence/pannuke_zero_shot_summary.json`, `evidence/pannuke_zero_shot_per_query.csv`, and `evidence/pannuke_gate_environment_audit.json`.

The existing MoNuSAC patient aggregate is also evidence-locked and is now reported directly: 25 held-out patients give +0.00762 Dice, bootstrap 95% CI [0.00181, 0.01462], 14/5/6 win/tie/loss. Its two-sided patient sign test is p=.115 and is stated in the manuscript; it is directional corroboration rather than a standalone rank-test confirmation.

As a post-hoc descriptive robustness check, the pre-registered five support strata were reported separately rather than selected. All five preserve a gate-vs-N0 CI lower bound above zero (0.0025--0.0044) and a fixed-N5-vs-N0 CI upper bound below zero. The strata do not alter the cohort-level inference or authorize a support-optimized claim; see `evidence/pannuke_support_sensitivity_summary.json`.

Public tissue-code metadata was also joined post hoc to the immutable query-level results. Across 19 strata (17--816 tiles each), the gate has 15 positive, four indeterminate, and no negative paired-bootstrap CIs; this is reported with its smaller strata rather than collapsed into an “all tissues significant” claim. See `evidence/pannuke_tissue_sensitivity_summary.json`.

An auxiliary 37-query nested component analysis prevents an instance-quality overclaim. The learned gate changes object recall by -0.00216 (95% CI [-0.00564, 0.00003]) and merge error by +0.00271 ([-0.00022, 0.00562]); neither supports a positive instance-separation statement. The manuscript therefore limits its efficacy claim to the predeclared union-mask Dice endpoint. See `evidence/nested_instance_metrics_summary.json`.

## End-to-end inference latency

The frozen 256 by 256 PanNuke support--query path was benchmarked after five warmups on an NVIDIA GeForce RTX 5090 D v2 (PyTorch 2.11.0+cu128). Thirty CUDA-synchronized measurements include support/query image and support-mask read, both DINOv2-L features, matching/point construction, SAM ViT-B image embedding, and N0/N5 dual decodes; query annotations were never opened. Full episode median/p90 was 240.7/389.4 ms. With an already cached support representation it was 178.9/309.3 ms. This is a declared single-GPU inference measurement, not system throughput. See `evidence/end_to_end_latency_summary.json` and `evidence/end_to_end_latency_per_episode.csv`.

## Manuscript build

`paper/main.tex` compiled successfully to `paper/main.pdf` (3 pages) after two `pdflatex --disable-installer` passes. The build log has no undefined citation or reference warnings. The current local MiKTeX installation does not contain `IEEEtran.cls`, so this PDF uses a generic two-column draft layout for content and visual QA only. It must be rebuilt using the official ICASSP 2027 template before submission.

## Visual inspection

All three pages were rendered and reviewed. The title, equations, table, primary plot, qualitative figure, captions, and bibliography are readable; no clipping or overlapping text was observed. The qualitative figure is intentionally evidence-dense and should be rebalanced after conversion to the official layout.

## Source regression audit

The MoNuSAC episode resolver now preserves a declared dataset-relative path even for a non-existent dry-run fixture, rather than silently falling back to `D:\SAM\processed\...`. The focused regression passed 4/4 and the complete suite passed **356/356** with:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda\python.exe -B -m pytest adapromptset\tests -q -p no:cacheprovider
```

The environment variable is required only because the workspace has a concurrently loaded long-running Python process with a conflicting OpenMP runtime. The resolver repair does not generate, transform, or aggregate any frozen paper evidence file.
