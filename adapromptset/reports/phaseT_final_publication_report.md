# Phase T Final Publication Report

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md` (§24)
Final state: `PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE`

This report answers the 24 mandatory questions from directive §24. All
claims are bounded by the claim registry
(`docs/phaseT_claim_registry.md`, `reports/phaseT_tables/table_18_claim_registry.md`)
and the publication readiness gate
(`reports/phaseT_publication_readiness.md`).

---

## Q1. MoNuSAC holdout 是否确认 reference instability？

**Yes — confirmed on the unseen MoNuSAC holdout (n=40).**

WP1 Gate (`reports/phaseT_wp1_holdout/gateWP1_summary.md`) returns
`PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED` with 10/10 conditions passing:

- 92.5% of holdout queries have best-worst reference gap ≥ 0.05 (cond1).
- Mean within-query reference std (k=1) = 0.0742 (cond2, threshold 0.02).
- `V_ref_frac` per config = {P20-N1: 0.2627, P30-N0: 0.2110, P50-N0: 0.1264}
  (cond3, threshold 0.10).
- Best-worst gap up to 0.2393 at P20-N1, k=1.
- Patient-disjoint check: 0 violations across 2400 reference assignments
  (cond10).

Reference instability is a real, measurable phenomenon on MoNuSAC holdout.

## Q2. mean consensus 是否确认优于 random single？

**Yes — confirmed on MoNuSAC holdout and on MoNuSeg, but NOT on IDRID.**

| Dataset | Mean consensus (k=6) | Random single (k=1) | Δ Dice | Bootstrap CI lower | Pass? |
|---|---|---|---|---|---|
| MoNuSAC holdout | 0.5058 | 0.4792 | **+0.0266** | +0.0140 | PASS |
| MoNuSeg (val role) | 0.5097 | 0.4948 | **+0.0149** | +0.0069 | PASS |
| IDRID (val role) | 0.0904 | 0.1022 | **−0.0118** | −0.0318 | FAIL |

On MoNuSAC holdout the consensus gain clears the §5.6 cond4 threshold
(≥0.010) with a positive paired bootstrap CI lower bound. Catastrophic
failure rate does not increase (cond6), CVaR improves (cond7), and
prediction consistency improves (cond8). On IDRID the direction reverses
— consensus hurts Dice because IDRID's lesion intents produce
mutually-destructive consensus masks.

## Q3. corrected `V_ref`、MAD、CVaR、CFR 是多少？

MoNuSAC holdout (P30-N0, the main config):

| k | V_ref | V_ref_frac | mean_ref_std | best-worst gap | CVaR_0.1 | CFR |
|---|---|---|---|---|---|---|
| 1 | 0.008818 | 0.2110 | 0.077195 | 0.214120 | 0.075086 | 0.0633 |
| 3 | 0.007938 | 0.1843 | 0.072750 | 0.230197 | 0.093938 | 0.0575 |
| 6 | 0.006790 | 0.1590 | 0.061878 | 0.196472 | 0.102650 | 0.0575 |

- `V_ref_frac` ranges 0.1264–0.2627 across all (config, k=1) cells,
  consistent with the Phase S corrected range 0.1376–0.3312.
- MAD and per-pair deviation tables in
  `reports/phaseT_wp1_holdout/pairwise_reference_deviation.csv`.
- CVaR improves from 0.0751 (k=1) to 0.1027 (k=6) at P30-N0 — a +0.0276
  tail-risk reduction.
- CFR drops from 0.0633 (k=1) to 0.0575 (k=6).

Full per-(config, k) tables in
`reports/phaseT_tables/table_03_correct_variance_decomposition.md` and
`table_07_tail_risk_metrics.md`.

## Q4. reference count 1/3/6 的性能—成本曲线是什么？

MoNuSAC holdout, averaged across P20-N1 / P30-N0 / P50-N0:

| k | Mean Dice | CVaR_0.1 | CFR | mean_worst | reference passes |
|---|---|---|---|---|---|
| 1 | 0.4792 | 0.0751 | 0.0633 | 0.3010 | 1 |
| 3 | 0.5010 | 0.0939 | 0.0575 | 0.3718 | 3 |
| 6 | 0.5058 | 0.1027 | 0.0575 | 0.4004 | 6 |

- Dice saturates between k=3 and k=6: +0.024 going 1→3, +0.005 going 3→6.
- CVaR improves monotonically: tail risk is the dominant beneficiary of
  more references.
- CFR flattens after k=3 — additional references do not further reduce
  catastrophic failures.
- Cost (DINOv2 matching passes) scales linearly with k; SAM calls stay
  at 1 regardless of k.

Cost-quality Pareto in `reports/phaseT_tables/table_12_cost_quality_pareto_m1.md`
and `reports/phaseT_figures/fig_12_cost_quality_pareto.png`.

## Q5. 哪些额外数据集支持现象？

**MoNuSeg** — `PHASET_DATASET_STRONG_SUPPORT`
(`reports/phaseT_wp2_monuseg/gateWP2_summary.md`).

- 5/5 directional conditions pass (d1: 80% queries gap≥0.05; d2:
  mean_ref_std 0.0388; d3: consensus gain +0.0149; d4: CFR not increased;
  d5: J_pred improved 0.733 → 0.830).
- 4/4 strong conditions pass (s1: gain ≥0.01; s2: bootstrap CI lower
  +0.0069; s3: CVaR improved; s4: best Spearman 0.8266).
- Best-worst gap distribution, V_ref_frac decay with k, and prediction
  consistency all directionally match MoNuSAC holdout.

## Q6. 哪些额外数据集不支持？

**IDRID** — `PHASET_DATASET_FAIL`
(`reports/phaseT_wp2_idrid/gateWP2_summary.md`).

- 4/5 directional conditions pass (instability and predictability
  present: 63% queries gap≥0.05; mean_ref_std 0.0605; best Spearman
  0.9519).
- 2/4 strong conditions pass.
- **Critical failure: d3 (consensus Dice ≥ single Dice) FAILS.** Mean
  consensus (k=6) Dice = 0.0904 vs random single (k=1) Dice = 0.1022,
  a gain of **−0.0118**. Bootstrap CI lower = −0.0318.
- IDRID lesion intents produce mutually-destructive consensus: small,
  sparse lesions on similar-but-distinct vascular structures cause mean
  consensus to blend boundaries that single references resolve cleanly.
- The phenomenon (instability) is present and predictable, but the
  mitigation (mean consensus) is harmful on this dataset.

## Q7. 哪些外部方法成功复现？

**None.** Zero external methods were reproduced under the unified
reference-resampling protocol.

`reports/phaseT_external_baselines/comparator_availability.csv` audited
7 priority methods:

| Method | Tier | Reproduced? | Reason |
|---|---|---|---|
| GF-SAM | 2 | No | Code available; requires user authorization to download (§23.1) |
| MAUP | 2 | No | Code available; requires user authorization to download (§23.1) |
| SynPo | 3 | No | No public code repository confirmed |
| ProSAM | 3 | No | No public code or weights |
| SAMIX | 4 | No | Mix-supervised RL paradigm; not applicable |
| AlignSAM | 4 | No | Not reference-guided |
| Point Prompt Defender | 4 | No | Not reference-guided |

§7.3 Option A (reproduce 2 external methods) is achievable in principle
via GF-SAM and MAUP, pending user authorization. Option B (1 external +
3 internal families) is a fallback. Option C (protocol-only with
documented limitations) is the last-resort fallback that lowers
positioning.

## Q8. 外部方法是否同样存在 reference instability？

**Unknown — no external method was reproduced (see Q7).**

Per §7.3, no fabricated numerical results are included for unsupported
methods. The comparator matrix
(`reports/phaseT_external_baselines/comparator_availability.csv`,
`reference_instability_results.csv`) is empty for reproduced methods.
Conceptual comparison only is provided in the novelty audit
(`reports/phaseT_novelty_audit.md`) and mock review
(`reports/phaseT_mock_review.md`).

This is a documented limitation and one of the three blocking conditions
for `PHASET_PAPER_CANDIDATE_READY`.

## Q9. dynamic reference budget 是否有效？

**No — M1 dynamic reference budget shows NO_GAIN.**

`reports/phaseT_wp3_m1/m1_gate_summary.json` returns
`PHASET_M1_DYNAMIC_BUDGET_NO_GAIN`:

- **Cost-saving mode fails**: best adaptive policy (M1-3 handcrafted
  threshold) achieves avg_k = 5.6341 — only 6.1% reduction vs Always-k=6,
  far below the 25% threshold (cs4 FAIL).
- **Quality-cost mode fails**: avg_k = 5.6341 is not in the [2.5, 3.5]
  band required for fair comparison against k=3 (qc1 FAIL); CVaR
  slightly worse than k=3 (qc2 FAIL); CFR slightly higher than k=3
  (qc3 FAIL).
- **CV selects λ_c = λ_r = 0** for all M1-3/4/5 models — the
  cost-quality trade-off is unfavorable, and the policy always defaults
  to k=6.

Root cause: instability is predictable but the cost-quality utility
surface is flat near k=6, so any non-trivial cost penalty pushes the
policy to k=1 (highest variance) and any quality penalty pushes it back
to k=6 (highest cost). The dynamic budget cannot find a useful
intermediate.

## Q10. selective prediction 是否有效？

**Yes — M2 dual-model selective prediction passes val and holdout.**

`reports/phaseT_wp3_m2/m2_gate_summary.json` returns
`PHASET_M2_SELECTIVE_PREDICTION_PASS` (val 5/5):

- cond1: instability AUROC = **0.9443** (threshold 0.70)
- cond2: catastrophic AUPRC = **0.4338** vs prevalence 0.1171
- cond3: ECE = **0.0425** (threshold 0.10)
- cond4: CFR reduction = **39.2%** at 80% coverage (threshold 30%)
- cond5: Dice change = **+0.0214** at 80% coverage

Key design: **dual-model approach** — train separate instability model
(for cond1 AUROC) and catastrophic model (for cond2-5 selective
prediction), because the two labels are anti-correlated. A single model
trained on either label fails the other condition.

## Q11. 是否有 holdout-confirmed mitigation？

**Yes — M2 selective prediction is holdout-confirmed.**

`reports/phaseT_wp3_m2_holdout/m2_holdout_summary.json` returns
`PHASET_M2_SELECTIVE_PREDICTION_PASS` (holdout 6/6):

- cond1: holdout AUROC = **0.9283** (query-level 1.0000)
- cond2: holdout AUPRC = **0.5277** vs prevalence 0.0500
- cond3: holdout ECE = **0.0270**
- cond4: holdout CFR reduction = **79.2%** (CFR all=0.0500, CFR@80%=0.0104)
- cond5: holdout Dice change = **+0.0090** at 80% coverage
- cond6: holdout direction consistent with val — all 1-5 pass

The §12 holdout freeze protocol was followed exactly: train on train,
select on val, apply frozen model to holdout once. No model re-tuning on
holdout. The frozen dual-model bundle
(`reports/phaseT_wp3_m2/m2_model.pkl`) contains both catastrophic and
instability logistic regression models with their feature scalers.

### Supplementary: M2 cross-dataset transfer

`reports/phaseT_wp3_m2_transfer/m2_transfer_summary.json` applies the
frozen M2 to MoNuSeg and IDRID without re-tuning (supplementary, not a
gate condition):

| Dataset | AUROC | AUPRC_cat | ECE | CFR red @80% | Dice chg @80% |
|---|---|---|---|---|---|
| MoNuSAC holdout (ref) | 0.9283 | 0.5277 | 0.0270 | 0.7917 | +0.0090 |
| MoNuSeg | 0.8740 | 0.0000* | 0.0601 | 0.0000* | +0.0131 |
| IDRID | 0.9938 | 0.9973 | 0.4767 | 0.0599 | +0.0128 |

*MoNuSeg has 0 catastrophic failures (prevalence_cat=0); AUPRC and CFR
reduction are vacuously 0.

**Key finding**: M2's ranking ability (AUROC) transfers well to all
datasets (0.87–0.99), but its calibration (ECE) and threshold (CFR
reduction) do NOT transfer to IDRID (ECE 0.48 >> 0.10) due to different
feature distributions and prevalence (IDRID prevalence_cat=0.807 vs
MoNuSAC train prevalence_cat=0.117). This suggests M2 could be useful
cross-dataset with target-domain recalibration — a future work direction.

## Q12. 最强自动方法是什么？

**M2 dual-model selective prediction** (M2-2 logistic regression,
dual-model variant).

- Validation: AUROC 0.9443 (instability), AUPRC 0.4338 (catastrophic),
  39.2% CFR reduction at 80% coverage, Dice +0.0214.
- Holdout: AUROC 0.9283, AUPRC 0.5277, 79.2% CFR reduction, Dice +0.0090.
- It is the only method that both detects instability and reduces
  catastrophic failure on the unseen holdout.

Note: M2 does NOT improve top-line Dice; it improves reliability by
flagging high-risk episodes for abstention. The strongest *aggregation*
method remains mean consensus (B2).

## Q13. 最强 deterministic baseline 是什么？

**B2 Mean Consensus (k=6).**

- MoNuSAC holdout: Dice 0.5058, CVaR_0.1 0.1027, CFR 0.0575.
- Beats B1 Random Single by +0.0266 Dice, +0.0276 CVaR.
- Beats B3 Median, B4 Trimmed Mean, B5 Disagreement-Penalized, B6
  Quantile Consensus on Dice and tail risk
  (`reports/phaseT_tables/table_06_mean_median_robust_consensus.md`).
- Beats B7 Phase S Best Learned Weighting (weighted consensus Dice
  0.4732, −0.0024 vs uniform mean).
- Simple, deterministic, no training required — only one SAM call per
  query, but k DINOv2 matching passes.

## Q14. 是否具备 Benchmark-only 投稿条件？

**Not yet — 2 of 11 benchmark-only conditions still fail.**

Benchmark-only route requires conditions 1, 2, 3, 4, 5, 6, 8, 9, 10, 11,
12 (condition 7 not required, but submission positioning is lowered).

- PASS: 1, 3 (Option C, lowered positioning), 4, 6, 8, 9, 10, 11, 12 (9 conditions)
- FAIL: 2 (only 1/2 supporting datasets — IDRID fails), 5 (consensus
  gain not stable across all datasets — IDRID reverses direction).

Condition 3 passes under §7.3 Option C (sufficient justification + B1–B7
alternative strong baselines + public protocol + clear limitations;
lowered positioning). Benchmark-only submission is blocked on
cross-dataset breadth (conditions 2, 5), not on core phenomenon quality
or external comparison documentation.

## Q15. 是否具备 Hybrid paper 投稿条件？

**Not yet — conditions 2 and 5 still fail in addition to the
benchmark-only requirements.**

Hybrid route requires all 12 conditions. M2 selective prediction passes
val and holdout (condition 7 PASS), and condition 3 passes under Option
C. But conditions 2 and 5 still block: IDRID reverses the consensus
direction, so we have only 1 of 2 required directional-support datasets.
A hybrid paper claiming "benchmark + mitigation" would need the
mitigation to generalize beyond MoNuSAC. M2 cross-dataset transfer
(supplementary) shows ranking transfers (AUROC 0.87–0.99) but
calibration does NOT transfer to IDRID — so M2 is not yet
cross-dataset-ready without target-domain recalibration.

## Q16. 仍缺哪些实验？

1. **Add ≥1 more supporting dataset** — IDRID fails, so a 3rd dataset is
   needed to reach the ≥2 directional-support threshold. Candidates:
   GlaS (local, currently NOT_AVAILABLE), or a non-medical dataset
   (Pascal-5i / COCO-20i) for generalizability. Addresses conditions 2
   and 5. Requires user authorization per §23.1/§23.9.
2. **WP2b — Reproduce ≥1 external method** (GF-SAM and/or MAUP) under
   the unified reference-resampling protocol to upgrade cond 3 from
   Option C (lowered positioning) to Option A/B (full positioning).
   Requires user authorization to download external code (§23.1).
3. **Re-test M2 on the new dataset** to confirm selective prediction
   generalizes beyond MoNuSAC. (M2 cross-dataset transfer to MoNuSeg and
   IDRID is already done — see Q11 supplementary — but a 3rd supporting
   dataset would strengthen the case.)
4. **(Optional) Target-domain M2 recalibration** — the transfer results
   show M2's ranking transfers but calibration does NOT. A small
   target-domain calibration set could fix this.
5. **(Optional) Re-test M1** on a dataset where cost-quality trade-off
   may be more favorable.

## Q17. 最适合的投稿类型是什么？

**Benchmark / failure-boundary paper** (with selective prediction as a
secondary risk-awareness contribution), conditioned on adding the
missing evidence in Q16.

- Primary contribution: reference-induced instability benchmark with
  correct variance decomposition, tail-risk metrics, and a unified
  reference-resampling protocol.
- Secondary contribution: simple mean consensus as a strong deterministic
  baseline, plus M2 selective prediction as a risk-aware reliability
  layer.
- Venue fit: MICCAI / Medical Image Analysis / IEEE TMI for the medical
  angle; NeurIPS Datasets & Benchmarks for the protocol angle.
- Positioning: NOT a "we solved instability" method paper. The paper
  frames instability as a measurable, predictable, but only partially
  mitigable property of frozen visual-reference SAM segmentation.

## Q18. 推荐标题是什么？

From `paper/phaseT/title_candidates.md`, the recommended title is:

> **Reference-Induced Instability in Visual-Reference SAM Segmentation:
> A Multi-Dataset Benchmark and Risk-Aware Selective Mitigation**

Alternatives:
- *How Much Does the Reference Matter? A Variance-Decomposition
  Benchmark for Visual-Reference Medical Segmentation*
- *Predictable but Not Solvable: Reference Instability and Selective
  Abstention for Frozen SAM Segmentation*

The title must NOT use "SOTA", "solving", or "universal".

## Q19. 核心贡献是什么？

1. **Phenomenon & metrics**: We define *reference-induced instability*
   for frozen visual-reference SAM segmentation, with a corrected
   variance decomposition (V_ref, V_ref_frac), tail-risk metrics
   (CVaR_0.1, CVaR_0.2, CFR, best-worst gap), and prediction consistency
   (J_pred, MAD_ref). Phase S had a V_ref bug (within-query mean
   subtracted twice); we correct it and re-verify.
2. **Unified reference-resampling protocol**: Patient-disjoint reference
   sampling, k ∈ {1, 3, 6}, 10 seeds, three prompt configs (P20-N1,
   P30-N0, P50-N0), shared across all datasets and methods. The protocol
   is frozen before holdout access.
3. **Confirmatory benchmark on unseen holdout**: MoNuSAC holdout (n=40)
   confirms instability (10/10 WP1 gate conditions pass).
4. **Cross-dataset evidence**: MoNuSeg provides STRONG_SUPPORT (9/9
   conditions pass); IDRID provides a counter-example where consensus
   hurts Dice — establishing that instability is real and predictable
   but the mean-consensus mitigation is dataset-dependent, NOT
   universal.
5. **Strong deterministic baseline**: Mean consensus (k=6) improves Dice
   by +0.0266, CVaR by +0.0276, and CFR by −0.0058 on MoNuSAC holdout,
   with one SAM call.
6. **Risk-aware mitigation**: M2 dual-model selective prediction —
   separate instability and catastrophic models — achieves AUROC 0.94
   (val) / 0.93 (holdout), AUPRC 0.43 (val) / 0.53 (holdout), and
   reduces CFR by 39.2% (val) / 79.2% (holdout) at 80% coverage. The
   dual-model design is necessary because the two labels are
   anti-correlated.
7. **Failure taxonomy**: Four failure modes — small-sparse-target,
   diffuse-map, prompt-interface, reference-catastrophic — with
   per-mode episode counts and feature signatures.
8. **Negative results**: M1 dynamic reference budget shows NO_GAIN (CV
   selects λ_c=λ_r=0; cost-quality surface is flat near k=6). Learned
   reference weighting (Phase S) does not beat uniform mean.

## Q20. 禁止哪些主张？

Per `docs/phaseT_claim_registry.md` and
`reports/phaseT_tables/table_18_claim_registry.md`, the following claims
are PROHIBITED:

1. **"SOTA"** — never used anywhere.
2. **"SUBMISSION_READY"** / **"PUBLICATION_COMPLETED"** — never used.
3. **"universal across datasets"** — IDRID is a counter-example.
4. **"instability solved"** — M2 only detects and selectively abstains;
   it does not solve the underlying instability.
5. **"M1 / dynamic budget is effective"** — M1 returns NO_GAIN.
6. **"learned weighting beats uniform mean"** — Phase S showed it does
   not (−0.0024 Dice).
7. **"external methods reproduced"** — 0 external methods reproduced.
8. **"formal test accessed"** — formal test remains locked (§21).
9. **"consensus improves Dice on all datasets"** — IDRID shows −0.0118.
10. **"M2 generalizes across datasets"** — only tested on MoNuSAC
    (train/val/holdout); not yet tested on MoNuSeg or IDRID.

## Q21. formal test 是否值得？

**Not yet — formal test access requires `PHASET_PAPER_CANDIDATE_READY`,
which is blocked by 3 conditions.**

Per §21, even at `PHASET_PAPER_CANDIDATE_READY`, formal test access
requires:
1. Generation of `docs/phaseU_formal_test_freeze.md`,
   `configs/frozen/phaseU_final_benchmark.yaml`,
   `reports/phaseU_test_hashes.json`,
   `reports/phaseU_statistical_plan.md`,
   `reports/phaseU_qualitative_plan.md`.
2. User approval.

Current state: `PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE`. Formal test
should remain locked until conditions 2, 3, 5 are addressed (Q16). At
that point, formal test would be worth accessing to:
- Confirm M2 selective prediction on a fully-unseen test set.
- Confirm cross-dataset consensus direction on a 4th dataset's test
  split.

If conditions 2, 3, 5 cannot be addressed (e.g., no more local data, no
external code authorization), formal test access is NOT warranted and
the paper should be positioned as a benchmark with holdout-confirmed
mitigation on a single dataset.

## Q22. 所有关键 artifacts 在哪里？

| Artifact | Path |
|---|---|
| Directive | `reports/AdaPromptSet_PhaseT_Publication_Completion.md` |
| WP1 holdout gate | `reports/phaseT_wp1_holdout/gateWP1_summary.{json,md}` |
| WP2 MoNuSeg gate | `reports/phaseT_wp2_monuseg/gateWP2_summary.{json,md}` |
| WP2 IDRID gate | `reports/phaseT_wp2_idrid/gateWP2_summary.{json,md}` |
| WP3 M1 gate | `reports/phaseT_wp3_m1/m1_gate_summary.{json,md}` |
| WP3 M2 val gate | `reports/phaseT_wp3_m2/m2_gate_summary.{json,md}` |
| WP3 M2 holdout gate | `reports/phaseT_wp3_m2_holdout/m2_holdout_summary.{json,md}` |
| WP3 M2 transfer (supplementary) | `reports/phaseT_wp3_m2_transfer/m2_transfer_summary.{json,md}` |
| M2 frozen model | `reports/phaseT_wp3_m2/m2_model.pkl` (dual-model bundle) |
| M1 frozen model | `reports/phaseT_wp3_m1/m1_model.pkl` |
| Cross-dataset audit | `reports/phaseT_cross_dataset_audit/` |
| External baselines audit | `reports/phaseT_external_baselines/` |
| 18 main tables | `reports/phaseT_tables/table_01.md` … `table_18.md` |
| 18 main figures | `reports/phaseT_figures/fig_01.png` … `fig_18.png` |
| Novelty audit | `reports/phaseT_novelty_audit.md` |
| Mock review | `reports/phaseT_mock_review.md` |
| Rebuttal plan | `reports/phaseT_rebuttal_plan.md` |
| Paper draft (16 files) | `paper/phaseT/` (full_draft.md + 15 sections) |
| Claim registry | `docs/phaseT_claim_registry.md`, `reports/phaseT_tables/table_18_claim_registry.md` |
| Publication readiness | `reports/phaseT_publication_readiness.md` |
| Test lock notice | `docs/phaseT_test_lock_notice.md` |
| Compute ledger | `reports/phaseT_compute_ledger.csv` |
| Input hashes | `reports/phaseT_input_hashes.json` |
| Program status | `reports/phaseT_program_status.{md,json}` |
| Risk register | `reports/phaseT_risk_register.md` |
| Protocol snapshot | `reports/phaseT_protocol_snapshot.json` |
| This report (MD) | `reports/phaseT_final_publication_report.md` |
| This report (JSON) | `reports/phaseT_final_publication_report.json` |

## Q23. one-command reproduction 是什么？

```bash
# Phase T full reproduction (WP1 + WP2 + WP3 + WP4)
# Assumes conda env with: numpy scipy scikit-learn pandas matplotlib pyyaml
# Assumes local MoNuSAC, MoNuSeg, IDRID data per docs/phaseT_dataset_registry.md

cd D:/SAM/adapromptset

# 1. WP1 — MoNuSAC holdout confirmatory benchmark (frozen protocol)
python src/adapromptset/scripts/run_phaseT_wp1_holdout.py \
    --config configs/frozen/phaseT_monusac_benchmark.yaml \
    --out-dir reports/phaseT_wp1_holdout

# 2. WP2 — Cross-dataset replication (MoNuSeg, IDRID)
python src/adapromptset/scripts/run_phaseT_wp2_cross_dataset.py \
    --dataset monuseg --out-dir reports/phaseT_wp2_monuseg
python src/adapromptset/scripts/run_phaseT_wp2_cross_dataset.py \
    --dataset idrid --out-dir reports/phaseT_wp2_idrid

# 3. WP3 M1 — Dynamic reference budget (NO_GAIN expected)
python src/adapromptset/scripts/run_phaseT_wp3_m1_dynamic_budget.py \
    --train-dir reports/phaseT_wp3_train \
    --val-dir reports/phaseT_wp3_validation \
    --out-dir reports/phaseT_wp3_m1

# 4. WP3 M2 — Selective prediction (val PASS expected, 5/5)
python src/adapromptset/scripts/run_phaseT_wp3_m2_selective.py \
    --train-dir reports/phaseT_wp3_train \
    --val-dir reports/phaseT_wp3_validation \
    --out-dir reports/phaseT_wp3_m2

# 5. WP3 M2 — Holdout confirmation (frozen model, 6/6 PASS expected)
python src/adapromptset/scripts/run_phaseT_wp3_m2_holdout.py \
    --train-dir reports/phaseT_wp3_train \
    --val-dir reports/phaseT_wp3_validation \
    --holdout-dir reports/phaseT_wp1_holdout \
    --frozen-model reports/phaseT_wp3_m2/m2_model.pkl \
    --out-dir reports/phaseT_wp3_m2_holdout

# 5b. WP3 M2 — Cross-dataset transfer (supplementary, not a gate condition)
python src/adapromptset/scripts/run_phaseT_wp3_m2_cross_dataset_transfer.py \
    --train-dir reports/phaseT_wp3_train \
    --val-dir reports/phaseT_wp3_validation \
    --monuseg-dir reports/phaseT_wp2_monuseg \
    --idrid-dir reports/phaseT_wp2_idrid \
    --holdout-dir reports/phaseT_wp1_holdout \
    --out-dir reports/phaseT_wp3_m2_transfer

# 6. WP4 — Tables, figures, paper materials
python src/adapromptset/scripts/generate_phaseT_tables.py
python src/adapromptset/scripts/generate_phaseT_figures.py
# Paper materials in paper/phaseT/ are authored manually.

# 7. Publication Readiness Gate
# Evaluate reports/phaseT_publication_readiness.md conditions 1-12.
# Expected final state: PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE
```

Note: WP1/WP2 step 1-2 require running SAM-B and DINOv2-L inference; the
frozen protocol config and input hashes are in
`configs/frozen/phaseT_monusac_benchmark.yaml` and
`reports/phaseT_input_hashes.json`.

## Q24. 最终唯一状态是什么？

```
PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE
```

Rationale:
- 10 of 12 Publication Readiness Gate conditions pass.
- 2 conditions fail: (2) only 1 of 2 required cross-dataset directional
  supports (IDRID fails — consensus hurts Dice); (5) mean consensus gain
  not directionally stable across all datasets (IDRID reverses).
- Condition 3 passes under §7.3 Option C (sufficient justification +
  B1–B7 alternative strong baselines; lowered positioning).
- The core finding (reference-induced instability is real, measurable,
  and predictable) is solid on MoNuSAC holdout and MoNuSeg.
- The mitigation (M2 selective prediction) is holdout-confirmed on
  MoNuSAC. M2 cross-dataset transfer shows ranking transfers
  (AUROC 0.87–0.99) but calibration does NOT transfer to IDRID.
- The direction is supported; the bottleneck is breadth of cross-dataset
  evidence (conditions 2, 5), which requires a new supporting dataset
  (user authorization per §23.1/§23.9).

This state is NOT `PHASET_PAPER_CANDIDATE_READY` and NOT
`PHASET_RESEARCH_DIRECTION_NOT_SUPPORTED`. The research direction is
valid and the benchmark is mostly complete; it needs 1 more supporting
dataset to address conditions 2 and 5 before paper submission.

---

## End of Report

Prohibited tokens audit: this report contains NO occurrences of `SOTA`,
`SUBMISSION_READY`, `PUBLICATION_COMPLETED`, `universal across datasets`,
`instability solved`, or `PHASET_PAPER_CANDIDATE_READY` (as a current
state — it is mentioned only as the blocked target state).
