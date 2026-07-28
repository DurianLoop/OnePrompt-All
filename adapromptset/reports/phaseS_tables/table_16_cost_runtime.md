# Table 16 — Cost / Runtime

## 1. Compute Budget

| Resource | Used | Cap | Remaining |
|---|---|---|---|
| GPU-hours | **0.00** | 210 | **210.00** |
| CPU runtime (s) | ~2200 (all phases combined) | — | — |
| Total SAM calls | **6779** | — | — |

**All phases ran on CPU; SAM was executed on CPU. No GPU resources were consumed.**

## 2. SAM Calls Per Phase / Track

| Track | Phase | SAM calls | Notes |
|---|---|---|---|
| Track A | benchmark | **0** | CPU-only, reused Phase R artefacts |
| Track B | Phase 2 (val) | **123** | 41 val queries × 3 configs × 1 |
| Track C | Phase 1 (train grid) | **6432** | 80 train queries × 3 configs × 27 combos |
| Track C | Phase 2 (val) | **117** | 41 val queries × 3 configs × 1 |
| Track C | Phase 3 (gate eval) | **110** | seed consistency, refcount, LOO |
| Track D | — | **0** | not started |
| Holdout | — | 0 | BLOCKED |
| Cross-dataset | — | 0 | BLOCKED |
| Formal test | — | 0 | LOCKED_UNUSED |
| **Total** | | **6779** | |

## 3. Runtime (Seconds) Per Phase

| Phase | Runtime (s) | Notes |
|---|---|---|
| Track A benchmark + audit | reuses Phase R artefacts | ~0 incremental |
| Track B Phase 2 | ~150 | 123 SAM calls on CPU |
| Track C Phase 1 (train grid) | 1164 | 27-combo grid |
| Track C Phase 2 (val) | 195 | 117 SAM calls |
| Track C Phase 3 (gate eval) | ~200 | seed consistency, refcount, LOO |
| Total Phase S incremental | ~2200 | all CPU |

## 4. Cost Decomposition

| Component | Calls | Runtime (s) | Notes |
|---|---|---|---|
| SAM forward (CPU) | 6779 | ~1700 | dominant cost |
| Matcher (reference matching) | n/a (Phase R reused) | ~0 | reuses Phase R matched candidates |
| Track B model training (M4_gbm) | n/a | ~50 | GBM inner CV |
| Track C grid evaluation | n/a | ~150 | aggregation logic |
| Bootstrap CI estimation | n/a | ~30 | 1000 bootstrap samples |
| Audit / I/O | n/a | ~270 | writing artefacts |

## 5. Cost-Effectiveness

| Metric | Value | Comment |
|---|---|---|
| GPU-hours per Gate A condition | 0 / 7 = 0 | CPU-only |
| SAM calls per Gate A condition | 0 | reused Phase R |
| SAM calls per Gate B condition (5/6 passed requirements) | 123 / 5 = ~25 | efficient |
| SAM calls per Gate C condition (5/8 passed) | 6659 / 5 = ~1332 | within budget |
| Total GPU-hours | 0 | far under 210 cap |

## 6. Phase R Cost (Recap, for Reference)

| Phase R resource | Used | Notes |
|---|---|---|
| SAM calls | 3756 | Phase R matching + consensus evaluation |
| GPU-hours | 0 | also CPU-only |
| Total runtime | ~5400 s | full Phase R |

Phase S Track A benchmark **reuses** Phase R's matched candidates and SAM outputs — no new SAM calls are needed for the benchmark itself.

## 7. Reference

- `reports/phaseS_compute_ledger.csv`
- `reports/phaseS_trackA_benchmark/cost_scaling.csv`
- `reports/phaseS_trackA_metric_audit/method_cost_decomposition.csv`
- `reports/phaseS_trackB/run_metadata.json`
- `reports/phaseS_trackC/run_metadata.json`
- `reports/phaseR_compute_ledger.csv`
