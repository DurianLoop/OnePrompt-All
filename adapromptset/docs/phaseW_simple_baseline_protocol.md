# Phase W — Simple Baseline Protocol

**Date**: 2026-07-27

## Methods to Compare (per todo.md Section 6.1)

| ID | Method | Description |
|----|--------|-------------|
| B0 | Random | Random risk ranking |
| B1 | Matching-map entropy | Entropy-based risk |
| B2 | Candidate Jaccard | Feature similarity-based risk |
| B3 | Prompt-coordinate overlap | Prompt dispersion-based risk |
| B4 | Prediction-consistency Jaccard | Mask agreement-based risk |
| B5 | M2-Instability | Instability risk model |
| B6 | M2-Catastrophic | Catastrophic risk model |
| B7 | Dual-head M2 | Combined instability + catastrophic |
| B8 | Full RAMP | Complete RAMP pipeline |

## Pipelines (per todo.md Section 6.2)

- DINOv2-L + SAM-B
- DINOv2-B + SAM-B
- DINOv2-L + SAM-L
- DINOv2-L + SAM2-Tiny
- DINOv2-L + SAM2-Small

## Metrics (per todo.md Section 6.4)

- AUROC, AUPRC
- CFR@80, Dice@80
- AURC, excess AURC
- ECE
- Paired bootstrap CI
- Query-level win/tie/loss