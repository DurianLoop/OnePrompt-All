# Phase W — External Baseline Protocol

**Date**: 2026-07-27

## Two-Table Structure (per todo.md Section 10.1)

### Table A — Shared-Backbone Comparison
Only methods with same segmenter backbone.

| Method | Backbone | Dice | CFR | Reproducible? |
|--------|----------|------|-----|---------------|
| Top-K (ours) | SAM-B | 0.446 | 0.143 | Yes |
| CycleSAM | SAM-B | 0.412 | 0.198 | Yes |
| SAM-Med2D | SAM-B | 0.438 | 0.187 | TBD (WP5) |

### Table B — Official-Default Comparison
Different backbones, shared protocol only.

| Method | Backbone | Dice | CFR | Notes |
|--------|----------|------|-----|-------|
| GF-SAM | SAM-H | 0.388 | 0.231 | Different backbone |
| MAUP | SAM-H | 0.279 | 0.312 | Different backbone |
| H-SAM | SAM-L | 0.405 | 0.215 | Different backbone |

## Required Documentation per Method
- Official commit
- Checkpoint
- Task alignment
- Input format
- Reference construction
- Prompt budget
- SAM calls
- Per-query predictions