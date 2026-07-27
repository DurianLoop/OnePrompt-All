# Phase AB Baseline Provenance Protocol

## Requirements for Main Table Results

Each baseline must have:
- Code commit hash
- Config file
- Checkpoint (if applicable)
- Prediction manifest
- Metric computation script
- Query-level results
- Reproduction command

## SAM-Med2D

Status: REMOVE_FROM_MAIN_TABLE

Reason: Checkpoint not verifiable. Missing:
- Official source link
- SHA256 hash
- Load log
- Inference log
- Per-query predictions

## External Baseline Table Structure

### Table A: Shared Backbone (same segmenter)
- Top-K + SAM-B
- CycleSAM + SAM-B

### Table B: Official Default (different backbone)
- GF-SAM + SAM-H
- MAUP + SAM-H
- H-SAM + SAM-L

Note: Explicitly annotate backbone differences. Do not claim "identical conditions."

## Main Risk Estimation Table

Comparators:
- Random Ranking
- Matching Entropy
- Candidate Jaccard
- Prompt Overlap
- Prediction Consistency (our method)
- M2-Instability (learned baseline)
- M2-Catastrophic (supervised upper bound, diagnostic only)