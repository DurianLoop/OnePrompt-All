# Phase S — Final Research Summary

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`
Commit: 3e63733
Directive: `D:\SAM\adapromptset\research.md` (Phase S)

## 1. 历史研究问题如何演化

本项目经历了四个阶段的问题演化：

- **Phase X (action bank saturation)**：验证旧 12-action bank 是否过窄。结论：bank 过窄，但其高 oracle 仅是探索性 in-reference 上界，不可部署。
- **Phase Y (fresh-query oracle)**：在 fresh split (train=80/val=41/holdout=40) 上验证 search oracle 是否稳定。结论：stable cross-reference oracle gap 仅 0.0061，Search Oracle 对 reference 过拟合。
- **Phase R (reference instability)**：确认 reference instability（88.75% queries best-worst gap ≥0.05，mean ref std≈0.0737），mean consensus 将 Dice 从 0.4297 提升至 0.4756，但预注册 variance gate 未通过（cond3 variance reduction = -0.0506）。
- **Phase S (multi-track research)**：在严格控制数据泄漏的前提下，对 4 条路线（A/B/C/D）进行系统探索。Track A 修正了 Phase R 的 metric 定义错误并建立 benchmark；Track B/C 尝试方法改进但均失败；Track D 未达到启动条件。

## 2. Phase R metric 是否有定义问题

**YES — Phase R cond3 存在严重的定义错误。**

Phase R cond3 比较：
- `var_single`：在 (query × reference) pair 轴上的方差 = `V_total = V_query + V_ref`
- `var_consensus`：在 query 轴上的方差 = `V_query`（consensus 消除了 reference 维度）

两者在不同轴上，比较是 apples-to-oranges。`VR = 1 - V_query / (V_query + V_ref)` 接近 0（reported -0.0506），但这不是 reference variance 的减少量。

正确比较：`VR_correct = 1 - V_ref(consensus) / V_ref(single) = 1 - 0 / 0.013340 = 1.0`（consensus 通过构造消除 within-query reference variance）。

详见 `reports/phaseS_trackA_metric_audit/variance_code_trace.md` 和 `phaseR_interpretation_addendum.md`。

## 3. Corrected Instability Metrics

| Metric | Definition | P30-N0 (k=1) |
|---|---|---|
| `V_query` | `Var_q[E_r[D(q,r)]]` — case difficulty | 0.040968 |
| `V_ref` | `(1/N) Σ_q Var_r[D(q,r)]` — reference-induced | 0.013340 |
| `V_ref_frac` | `V_ref / (V_query + V_ref)` | 0.2456 |
| `MAD_ref` | `(1/N) Σ_q E_{i,j}|D(q,S_i)-D(q,S_j)|` | 0.113918 |
| `J_pred` | `E_{S_i,S_j} IoU(M_hat(q,S_i), M_hat(q,S_j))` | 0.607517 |
| `CVaR_0.1` | Mean of bottom-10% Dice | 0.046793 (k=1) / 0.073764 (k=6) |
| `CFR` | `P(Dice < 0.1)` — catastrophic failure rate | 0.1423 (k=1) / 0.0732 (k=6) |
| `worst_ref_gap` | `max_r D(q,r) - min_r D(q,r)` | 0.244623 |

## 4. Track A/B/C/D 结果

### Track A — Metric Correction and Stability Benchmark
- Status: `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID`
- Gate A: **7/7 PASSED**
- 0 GPU-hours, 0 SAM calls (CPU-only, reused Phase R artefacts)
- Key: V_ref_frac 0.1376–0.3312; 80.5%–90.2% queries gap≥0.05; prediction Spearman 0.8832

### Track B — Learned Reference Reliability
- Status: `PHASES_TRACK_B_NO_GAIN`
- Gate B: **1/6 conditions PASSED, 5/6 requirements PASSED**
- 0 GPU-hours, 123 SAM calls
- Best model: M4_gbm (T=0.25, ε=0.0, λ=0.0)
- Utility prediction Spearman: train 0.9163, val 0.4781
- Delta Dice: -0.0024 (weighted 0.4732 vs uniform 0.4756)
- Pairwise dev reduction: 60.09% (only condition passed)
- Decision: cond1 (Dice+0.005) and cond2 (std -10%) both FAILED → NO_GAIN

### Track C — Set-Conditioned Robust Prompting
- Status: `PHASES_TRACK_C_NO_GAIN`
- Gate C: **5/8 conditions PASSED**
- 0 GPU-hours, 6659 SAM calls (6432 train + 117 val + 110 gate eval)
- Methods: C3 (pixel-wise agreement gate), C4 (stable candidate scoring)
- Hyperparameter grid: 27 combos (α∈{0,1,5}, β∈{0,0.5,1}, γ∈{0,0.5,1})
- Best: (α=0, β=0, γ=0) = B2_mean baseline; ALL non-zero combos degrade Dice
- Delta Dice vs Track B: +0.0047 (FAILED ≥0.005)
- Decision: cond1 and cond2 both FAILED → NO_GAIN

### Track D — Direct Reference-Conditioned Segmentation
- Status: `START_CONDITION_NOT_FULLY_MET` (NOT STARTED)
- 0 GPU-hours, 0 SAM calls
- Start condition 3 NOT met: SAM refinement HELPS (consensus Dice 0.48 vs raw matching map <0.3)
- Track D not started

## 5. 每条路线的假设与 Gate

| Track | Hypothesis | Gate | Result |
|---|---|---|---|
| A | Reference instability is real, measurable, and predictable from test-time-safe features | 7 conditions (§6.5) | **7/7 PASS** |
| B | Learned reliability weighting reduces variance and improves Dice | 6 conditions (§7.6) | **1/6 PASS** (NO_GAIN) |
| C | Set-conditioned modeling captures interactions beyond per-reference scoring | 8 conditions (§8.4) | **5/8 PASS** (NO_GAIN) |
| D | Direct reference-conditioned segmentation bypasses prompt bottleneck | 4 start conditions (§9.1) | **3/4 met** (NOT STARTED) |

## 6. 支持与否证的假设

### Supported (验证)
- H1: Reference instability is real and substantial (V_ref_frac 0.1376–0.3312).
- H2: Reference instability is predictable from test-time-safe features (Spearman 0.8832 train, 0.4781 val).
- H3: Deterministic mean consensus improves Dice over single-reference (0.4756 vs 0.4297, +0.046).
- H4: Consensus eliminates within-query reference variance by construction (VR_correct = 1.0).

### Falsified (证伪)
- H5: Learned reliability weighting improves Dice over uniform mean. (FALSIFIED: -0.0024)
- H6: Learned reliability weighting reduces within-query reference std by ≥10%. (FALSIFIED: 2.83%)
- H7: Pixel-wise agreement gating improves over mean consensus. (FALSIFIED: all α>0 degrade Dice)
- H8: Stable candidate scoring improves over baseline candidate selection. (FALSIFIED: all β>0 or γ>0 degrade Dice)
- H9: Set-conditioned methods improve CVaR. (FALSIFIED: -0.0031)

## 7. Holdout 与跨数据集状态

- **Holdout**: BLOCKED. Directive §11 requires Track B/C/D PASS; none passed.
- **Cross-dataset**: BLOCKED. Directive §12 requires holdout PASS.
- **Formal test**: LOCKED_UNUSED. Directive §22 prohibits access.

## 8. 最强自动方法、deterministic baseline、upper bound

| Category | Method | Val Dice | Notes |
|---|---|---|---|
| Strongest automatic method | M4_gbm weighted consensus (Track B) | 0.4732 | Spearman 0.4781, but delta -0.0024 vs uniform |
| Strongest deterministic baseline | B2_mean consensus k=6 P30-N0 | 0.4756 | Best overall; no learning required |
| Upper bound (exploratory) | Search Oracle (Phase Y) | ~0.4338 (cross-ref) | Overfit to reference; not deployable |
| Single-reference (lower bound) | B1_random | 0.4297 | Phase R baseline |

## 9. Mean Dice、Reference Variance、CVaR、Catastrophic Failure

(B2_mean, P30-N0, k=6, validation n=41)

| Metric | Value |
|---|---|
| Mean Dice | 0.4756 |
| V_ref (within-query) | 0.0133 |
| V_ref_frac | 0.2456 (at k=1) / 0 (at k=6 by construction) |
| Mean ref std (k=1) | 0.0893 |
| CVaR_0.1 | 0.0738 |
| CFR (Dice<0.1) | 0.0732 |
| Worst-subset Dice | 0.0444 |
| Best-worst gap (k=1) | 0.2446 |

## 10. Cost / Runtime

| Stage | GPU-hours | SAM calls | Runtime (s) |
|---|---|---|---|
| Track A (audit + benchmark) | 0.00 | 0 | — |
| Track B (phase 1+2+3) | 0.00 | 123 | 616 |
| Track C (phase 1+2+3) | 0.00 | 6659 | 1570 |
| **Total** | **0.00** | **6779** | **~2200** |
| Cap | 210 | — | — |
| Remaining | 210 | — | — |

All phases ran on CPU. SAM-B inference was executed via the shared CPU path.

## 11. Ablations

### Track C α/β/γ ablation (train, 80 queries × 3 configs, 27 combos)

| α | β | γ | Mean Dice |
|---|---|---|---|
| 0.0 | 0.0 | 0.0 | **0.5038** (best, = B2_mean) |
| 1.0 | 0.0 | 0.0 | 0.4992 |
| 5.0 | 0.0 | 0.0 | 0.4954 |
| 0.0 | 0.5 | 0.0 | 0.4934 |
| 0.0 | 1.0 | 0.0 | 0.4884 |
| 0.0 | 0.0 | 0.5 | 0.4945 |
| 0.0 | 0.0 | 1.0 | 0.4934 |
| 5.0 | 1.0 | 1.0 | 0.4789 (worst) |

Key: ALL non-zero hyperparameters DECREASE Dice monotonically. C3 (agreement gate) and C4 (candidate scoring) provide no benefit.

### Track B T/ε/λ ablation note

Track B selected T=0.25, ε=0.0, λ=0.0 via inner CV. Higher temperature T>0.5 flattened weights toward uniform (no benefit). Non-zero ε (weight floor) slightly helped prevent collapse but did not change Dice direction. λ (risk objective) had no effect because CVaR gradient was dominated by query difficulty.

## 12. Failure Taxonomy

| Failure Mode | Frequency | Track | Root Cause |
|---|---|---|---|
| Catastrophic failure (Dice<0.1) | 7.32% (k=6) / 14.23% (k=1) | A | SAM-B produces empty/wrong mask from point prompt |
| LOO instability (drop >0.05) | 7/30 LOO runs | C | Removing one reference shifts consensus enough to flip SAM output |
| Reliability weighting no-gain | 100% of configs | B | Even correct utility prediction (Spearman 0.478) doesn't improve Dice — bottleneck is downstream of weighting |
| Set-conditioned degradation | 26/27 combos | C | Agreement gating and candidate scoring remove information rather than adding it |
| Prompt-interface bottleneck | structural | D | SAM refinement HELPS (0.48 vs <0.3), so "point prompt loses" premise is false |
| Train-val generalization gap | 0.037 (Track C) | C | Within acceptable bound (<0.05), but indicates mild overfitting risk |

## 13. Novelty Audit

详见 `reports/phaseS_novelty_audit.md`。

核心 novelty：
1. Corrected V_ref decomposition（修正 Phase R 的 axis-mixing 错误）
2. Tail-risk metric suite for reference instability（CVaR, CFR, worst-subset）
3. Preregistered multi-track protocol with claim registry
4. Honest negative results documenting the prompt-interface bottleneck

## 14. 与相关工作的差异

| Work | Phase S Difference |
|---|---|
| GF-SAM | We benchmark instability, not propose a new matcher |
| SynPo | We use real references, no synthetic prompts |
| AlignSAM | We use frozen SAM-B, no fine-tuning |
| ProSAM | We do not learn prompts; fixed candidates from matching |
| MAUP | Single-scale DINOv2-L, no multi-scale anchors |
| SAMIX | We aggregate reference evidence, not mix prompts |
| Few-shot seg | We do not train a segmenter; reference-conditioned prompting only |
| Visual ref seg | We benchmark the *instability* of this paradigm |

## 15. 最适合的投稿方向

Per directive §21: **Benchmark + failure-boundary paper**.

理由：
- Track A PASSED (instability benchmark valid)
- No method track (B/C/D) PASSED
- Negative results are scientifically valuable
- Metric correction is a methodological contribution

## 16. 推荐论文标题和核心主张

**Title**: "Reference-Induced Instability in Visual-Reference Segmentation: A Benchmark and Failure-Boundary Analysis"

**Core claims** (from claim registry):
1. "Valid references induce substantial, measurable segmentation variability under the frozen matching-and-prompting pipeline, and this instability is associated with test-time-safe disagreement features."
2. "Reference utility is predictable from test-time-safe features, but weighted consensus does not reduce variance; the utility signal is a scientific finding without a deployable stability gain."

## 17. 不允许的主张 (Permanently Prohibited)

1. ❌ "Reference instability solved" / "eliminated"
2. ❌ "Universal across datasets"
3. ❌ "SOTA" / "state-of-the-art segmentation"
4. ❌ "Formal test success"
5. ❌ "ProSAM/MAUP/SAMIX fully outperformed"
6. ❌ "Reference weighting is model-agnostic"
7. ❌ "Mean Dice alone proves robustness"
8. ❌ "Reference instability is purely random"
9. ❌ "The method works without references"
10. ❌ "Submission-ready" / "publication-completed"

## 18. Formal Test 是否值得

**NOT YET.** Per directive §21, benchmark paper requires ≥2–3 datasets first. Current benchmark is MoNuSAC-only. Formal test (locked) should only be accessed after:
1. Cross-dataset validation (GlaS or DDR) confirms instability generalizes.
2. Holdout=40 confirms benchmark metrics on unseen internal data.
3. A frozen benchmark protocol is preregistered for formal test.

Current status: LOCKED_UNUSED.

## 19. Artifact 路径与 Git Commits

### Git commits (branch `phaseS-multitrack-reference-robust-research`)
| Commit | Description |
|---|---|
| 2de8b4d | Preregistration (11 docs + 7 reports) |
| abd1c1b | Track A metric audit |
| bba99c5 | Track A benchmark PASSED |
| 50c82c4 | Track selection → Track B |
| 4a0e6ef | Track B NO_GAIN |
| 3e63733 | Track C NO_GAIN (current HEAD) |

### Key artifacts
- `reports/phaseS_trackA_benchmark/` — Track A benchmark (9 files)
- `reports/phaseS_trackA_metric_audit/` — Metric audit (10 files)
- `reports/phaseS_trackB/` — Track B results (11 files)
- `reports/phaseS_trackC/` — Track C results (10 files)
- `reports/phaseS_tables/` — 20 tables (table_01..table_20)
- `reports/phaseS_novelty_audit.md` — Novelty audit
- `reports/phaseS_mock_review.md` — Mock review
- `reports/phaseS_final_research_summary.md` — This file
- `reports/phaseS_final_research_summary.json` — Machine-readable summary
- `reports/phaseS_program_status.json` — Program status
- `reports/phaseS_gate_status.md` — Gate status
- `reports/phaseS_compute_ledger.csv` — Compute ledger
- `docs/phaseS_claim_registry.md` — Claim registry
- `docs/protocol_history.md` — Protocol history
- `docs/decision_log.md` — Decision log

## 20. One-Command Reproduction

```bash
git checkout 3e63733
cd D:\SAM\adapromptset
python src/adapromptset/scripts/run_phaseS_trackA.py --all
python src/adapromptset/scripts/run_phaseS_trackB.py --all
python src/adapromptset/scripts/run_phaseS_trackC.py --all
```

Track D was not started (start condition not met), so no reproduction command for it.

## 21. Unresolved Blockers

1. **Single dataset**: Only MoNuSAC tested. Directive §21 requires ≥2–3 datasets for benchmark paper. GlaS (55/30) and DDR (train/valid) are available but locked pending holdout PASS.
2. **No holdout validation**: Holdout=40 never used (no method track passed). Benchmark claims rest on validation n=41 only.
3. **No cross-dataset validation**: Blocked by holdout. Cannot confirm instability generalizes.
4. **CPU-only scale**: 0 GPU-hours used; 6779 SAM calls on CPU. Absolute Dice numbers (0.4756) are low; relative comparisons hold but absolute performance claims are limited.
5. **No head-to-head with related methods**: ProSAM, MAUP, SAMIX not run under same protocol. Novelty audit is conceptual, not empirical.
6. **Track C limited to C3/C4**: C1 (DeepSets) and C2 (Set Attention) not implemented. Higher-capacity set models untested.
7. **Prompt-interface bottleneck**: Identified but not resolved. Track D would require a new direct-segmentation method, but start condition 3 is not met.

## 22. 最终唯一状态

```
PHASES_BENCHMARK_PAPER_READY
```

Rationale:
- Track A PASSED (7/7) → instability benchmark valid.
- Track B/C/D all failed/not started → no method paper.
- Per directive §20, Track A PASS + no method PASS → `PHASES_BENCHMARK_PAPER_READY`.
- Allowed paper type (§21): Benchmark + failure-boundary paper.
- Allowed claims: Track A instability benchmark claim + Track B Utility-Only claim.
- Holdout BLOCKED, cross-dataset BLOCKED, formal test LOCKED_UNUSED.
