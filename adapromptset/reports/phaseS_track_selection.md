# Phase S Track Selection

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`
Directive: §10 Track Selection and Auto-Routing

## Track A Result

**State**: `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID` (7/7 conditions passed)

### Evidence Summary

| Dimension | Finding |
|---|---|
| Phenomenon stability | V_ref is 13.8%-33.1% of total variance; 80.5%-90.2% queries gap≥0.05 |
| Corrected metrics | V_ref(k=1) > V_ref(k=3) > V_ref(k=6)=0 (monotone, explainable) |
| Tail risk linkage | P50-N0: ρ(gap, catastrophic)=-0.395, p=0.011 |
| Predictability | prediction_jaccard_mean |ρ|=0.883 (train); far exceeds 0.25 threshold |
| Metric audit | Phase R cond3 was invalid (mixed variance axes); corrected VR=100% |

### Caveat

The strongest predictor (`prediction_jaccard_mean`) requires multiple SAM calls to
compute (it measures prediction-mask Jaccard across references). Track B must
evaluate truly test-time-safe features that are computable WITHOUT running SAM:
- `matching_map_disagreement` (from DINOv2 features only)
- `candidate_jaccard_mean` (from candidate pools only)
- `prompt_overlap_mean` (from positive point sets only)

These are computable at test time with zero SAM calls (just DINOv2 forward
passes, which are already needed for matching).

## Scoring (directive §10)

| Dimension | Weight | Score (0-10) | Rationale |
|---|---:|---:|---|
| Scientific phenomenon stability | 20 | 9 | Instability is real, large, and replicates across seeds/configs |
| Method novelty potential | 20 | 7 | Reliability-weighted consensus is novel for SAM point prompting |
| Validation effect size | 15 | 8 | V_ref is 13-33% of total; ΔDice from consensus +0.046 |
| Predictability | 10 | 9 | |ρ|=0.883 for prediction_jaccard; need to verify test-time-safe features |
| Cross-dataset potential | 10 | 6 | GlaS/DDR not yet tested; morphology may differ |
| Related-work differentiation | 10 | 7 | No prior work on reference reliability for SAM prompting |
| Compute cost | 5 | 8 | Lightweight models (<5M params); 1 SAM call at deployment |
| Paper narrative completeness | 10 | 7 | Benchmark + method + holdout narrative feasible |
| **Weighted total** | 100 | **770/1000** | |

## Routing Decision

Per directive §10 auto-rules:
> "Track A strong + reliability predictable → Track B"

Track A is strong (7/7 passed, weighted score 770/1000). Reliability is
predictable (|ρ|=0.883 >> 0.30 threshold, even with the caveat that the
strongest feature needs SAM calls — the test-time-safe features will be
evaluated in Track B).

### Selected Track: **Track B — Learned Reference Reliability and Robust Aggregation**

### Rationale

1. Track A confirmed reference instability is real and significant.
2. The metric correction audit showed Phase R's cond3 was invalid; consensus
   actually eliminates within-query reference variance by construction.
3. The remaining scientific question: can we do BETTER than uniform consensus
   by learning per-reference reliability?
4. Track B directly addresses this: learn `û(q, R_r)` from test-time-safe
   features, then produce weighted consensus `C_weighted = Σ w_r C_r`.
5. If Track B passes, we have a method paper (benchmark + reliability method).
6. If Track B fails but has utility-prediction signal, we have a benchmark
   paper with diagnostic value.

### Paused Tracks

- **Track C** (Set-Conditioned Robust Prompting): PAUSED. Only启动s if Track B
  has stable positive signal but plateaus. Track B must run first.
- **Track D** (Direct Reference-Conditioned Segmentation): PAUSED. Only启动s if
  Track B/C fail AND prompt interface is confirmed as the bottleneck. Track A
  showed the prompt interface IS a bottleneck (instability is real), but Track
  B may resolve it without bypassing prompts.

## Next Action

Execute Track B (§7):
1. Generate train-only labels (utility, pairwise, harmfulness) from R1 data.
2. Compute test-time-safe features (reference-only, reference-query, leave-one-ref-out).
3. Train model ladder M0→M4 (M5 only if positive signal).
4. Evaluate on validation with weighted consensus.
5. Run Gate B (§7.6) with 6+ conditions.

Compute budget: 35 GPU-hours cap for Track B.
