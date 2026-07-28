# Table 03 — Reference Resampling Setup

## 1. Reference Subset Sizes

| k | Meaning | Use |
|---|---|---|
| 1 | Single reference | Worst-case reference instability; baseline single-ref |
| 3 | Small consensus | Aggregation with few references |
| 6 | Larger consensus | Aggregation benefit assessment |

## 2. Configurations

| Config | Description | V_ref_frac |
|---|---|---|
| P20-N1 | 20 candidate positives, 1 negative prompt | 0.3312 |
| P30-N0 | 30 candidate positives, 0 negatives (primary config) | 0.2456 |
| P50-N0 | 50 candidate positives, 0 negatives | 0.1376 |

P30-N0 is the **primary configuration** for Track B/C evaluation (best baseline Dice).

## 3. Resampling Seeds

| Set | # seeds | Purpose |
|---|---|---|
| Train | 10 seeds per (config, k) | Reference-subset resampling for benchmark + training of Track B/C models |
| Validation | fixed seeds (deterministic) | Held-out reference subsets for Track A benchmark reporting and Track B/C val metrics |

- Train seeds are independent across (config, k) cells.
- Validation seeds are fixed (committed in preregistration `2de8b4d`) to prevent seed gardening.

## 4. Validation Query Counts

| Set | # queries | Notes |
|---|---|---|
| Validation queries (Track A) | 41 | Used for V_ref_frac, mean_ref_std, best-worst gap, CVaR, CFR reporting |

## 5. Determinism

- All metrics reported on the **fixed validation seeds** to be reproducible.
- Train seeds used only for (a) bootstrap CI estimation, (b) Track B/C model fitting.
- No seed selection on validation — first fixed seed set is the reported set.

## 6. SAM Calls Per Cell

| Phase | # queries × # configs × k | SAM calls |
|---|---|---|
| Track A benchmark | reuses Phase R artefacts (no new SAM calls) | 0 |
| Track B Phase 2 (val) | 41 × 3 × 1 | 123 |
| Track C Phase 1 (train grid) | 80 × 3 × 27 | 6432 |
| Track C Phase 2 (val) | 41 × 3 × 1 | 117 |
| Track C Phase 3 (gate eval) | — | 110 |
| **Total SAM calls** | | **6779** |

## 7. Compute

| Resource | Used | Remaining |
|---|---|---|
| GPU-hours | 0.00 | 210.00 |
| CPU runtime (s) | ~2200 total | — |

All phases ran on CPU; SAM was executed on CPU. No GPU resources were consumed.

## 8. Config Selection Rationale (Phase S Track Selection)

- P20-N1 has the highest V_ref_frac (0.3312): maximum reference instability — used to stress-test the benchmark.
- P30-N0 is the primary operating point for Track B/C (best Dice/variance trade-off).
- P50-N0 has the lowest V_ref_frac (0.1376): used to confirm the predictability signal holds when instability is smaller.

## Reference

- `reports/phaseS_trackA_benchmark/benchmark_summary.md`
- `reports/phaseS_trackB/run_metadata.json`
- `reports/phaseS_trackC/run_metadata.json`
- `reports/phaseS_protocol_snapshot.json`
- Preregistration commit: `2de8b4d`
