# Null Model Registry

**Date**: 2026-07-27
**Phase**: AB Integrity Closure

## Null Model Definitions

### Null 1: Mean-Preserving Random Beta
- **Definition**: For each query, draw a random beta-distributed pseudo-CFR with the same mean Dice as observed data
- **Preserves**: Per-query mean Dice
- **Randomizes**: CFR structure, correlation structure
- **Input**: query-level mean Dice values
- **Seed**: 42
- **Phase V result**: r_null = -0.583 (L+SAM-B)
- **Phase W result**: r_null = 0.0 (BUGGY — function returned 0.0)
- **Applicable claim**: "Is the observed anti-correlation stronger than expected from mechanical constraints?"

### Null 2: Query-Shuffled Null
- **Definition**: Shuffle CFR values across queries, preserving marginal distribution
- **Preserves**: Marginal CFR distribution
- **Randomizes**: Query-level pairing
- **Input**: query-level CFR values
- **Seed**: 42
- **Applicable claim**: "Is the observed correlation due to query-level pairing?"

### Null 3: Reference-Shuffled Null
- **Definition**: Shuffle reference assignments within each query
- **Preserves**: Per-query reference distribution
- **Randomizes**: Reference-to-outcome mapping
- **Applicable claim**: "Is the instability driven by specific reference choices?"

## Resolution

1. Phase V Null 1 is CORRECT (r_null = -0.583)
2. Phase W null model is BUGGY (returned 0.0)
3. Corrected Δr is POSITIVE for SAM-B pipelines
4. Anti-correlation is a MECHANICAL artefact, not structural
5. Anti-correlation removed from main text per AA7 decision