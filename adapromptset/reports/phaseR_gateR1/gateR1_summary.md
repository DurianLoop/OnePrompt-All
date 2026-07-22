# Phase R Gate R1: Reference Instability Replication Summary

Date: 2026-07-22T13:40:48
Train queries: 80
Resampling seeds: 5
Prompt configs: 3 (P20-N1, P30-N0, P50-N0)

## Replication Statistics

| Metric | Value |
|---|---:|
| % queries best-worst gap >= 0.05 | 0.8875 |
| Mean reference std | 0.0737 |
| Seed direction consistent | YES |
| Sensitive config | P20-N1 |

## Gate R1 Conditions

| # | Condition | Pass |
|---|---|---|
| 1 | >= 30% queries best-worst gap >= 0.05 | YES |
| 2 | mean reference std >= 0.02 | YES |
| 3 | direction consistent across 5 seeds | YES |
| 4 | >= 1 config shows stable sensitivity | YES |
| 5 | not a code error | YES |

## Decision: **PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED**

