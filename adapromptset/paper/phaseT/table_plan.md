# Table Plan (18 tables, directive §14)

Every table is generated from a frozen gate-summary CSV; no table is
regenerated after holdout observation. Every robustness table reports
variance or tail-risk evidence alongside mean Dice; mean Dice alone is
never used as a robustness claim.

## T1. Dataset statistics
**Caption:** *Dataset statistics and roles.* Cases, patients, and roles
for MoNuSAC (primary), MoNuSeg (secondary), and IDRiD (tertiary).
Reference pool is train-only and patient-disjoint from queries in every
dataset. Formal test sets are locked and unused.

## T2. Reference protocol
**Caption:** *Reference-resampling protocol.* Six references per query,
drawn from the train-only pool, patient-disjoint. k∈{1,3,6} subset
sizes; ten seeds on MoNuSAC train, five on MoNuSeg and IDRiD. Holdout
and validation reference allocations are fixed at pre-registration and
hash-verified.

## T3. Correct variance decomposition
**Caption:** *Corrected variance decomposition on the MoNuSAC holdout.*
`V_query`, `V_ref`, and `V_ref_frac` per (config, k). `V_ref_frac` is
0.061–0.263 across configs and k, confirming reference choice is a
substantial component of total variance.

## T4. Single-reference instability
**Caption:** *Single-reference (k=1) instability on the MoNuSAC
holdout.* Mean within-query reference std, `MAD_ref`, best-worst gap
(mean and fraction ≥ 0.05), per config. 92.5% of queries have
best-worst gap ≥ 0.05.

## T5. Reference-count scaling
**Caption:** *Reference-count scaling (k=1, 3, 6) on the MoNuSAC
holdout.* Mean Dice, `CVaR_0.1`, CFR, and `J_pred` per (config, k).
Consensus (k=6) improves mean Dice (+0.027), raises `CVaR_0.1`, lowers
CFR, and raises `J_pred` relative to k=1.

## T6. Mean / median / robust consensus
**Caption:** *Aggregator comparison on the MoNuSAC holdout.* B1 random
single, B2_mean, B3_median, B4_trimmed mean, B5, B6, B7_learned.
B2_mean is the strongest automatic baseline; learned weighting (B7)
does not beat it. Paired bootstrap 95% CI and Cohen's d against B1.

## T7. Tail-risk metrics
**Caption:** *Tail-risk metrics on the MoNuSAC holdout.* `CVaR_0.1`,
`CVaR_0.2`, CFR (Dice < 0.1), worst-subset Dice, min worst, p10 worst,
per (config, k). CFR at k=6 ranges 0.050–0.073 across configs.

## T8. Prediction consistency
**Caption:** *Prediction consistency on the MoNuSAC holdout.* Mean
cross-seed `J_pred` and `MAD_ref` per (config, k). `J_pred` rises from
0.649 (k=1) to 0.719 (k=6).

## T9. External method instability
**Caption:** *External comparator availability and tier.* GF-SAM, SynPo,
ProSAM, MAUP, SAMIX, AlignSAM, PPD classified by tier (1 directly
reusable; 2 minimal adaptation; 3 conceptual; 4 not applicable).
Reproduction status: GF-SAM and MAUP identified as Tier 2 reproducible;
numerical reproduction pending user authorization. No Tier 3 numerical
results fabricated.

## T10. Dynamic budget (M1)
**Caption:** *M1 dynamic reference budget results (validation).* Six
policy variants; `λ_c=λ_r=0` selected for all. Best policy M1-3
(handcrafted threshold): avg_k 5.63, only 6.1% reduction versus the 25%
gate. Cost-saving mode FAIL (cs4); quality-cost mode FAIL (qc1, qc2,
qc3).

## T11. Selective prediction (M2)
**Caption:** *M2 dual-model selective prediction, validation and
holdout.* Instability-trained M2-2 logistic regression: instability
AUROC 0.944 (val), 0.928 (holdout). Catastrophic-trained M2-2 logistic
regression: catastrophic AUPRC 0.434 (val, prevalence 0.117), 0.528
(holdout, prevalence 0.050); ECE 0.043 (val), 0.027 (holdout); CFR
reduction 39.2% (val), 79.2% (holdout); Dice change +0.021 (val),
+0.009 (holdout). 5/5 validation PASS, 6/6 holdout PASS.

## T12. Cost-quality Pareto
**Caption:** *Cost-quality Pareto, M1.* Mean Dice, `CVaR_0.1`, CFR,
avg_k, and cost for the six M1 policies and three static baselines.
M1-2 (always k=6) Pareto-dominates the learned policies.

## T13. Holdout confirmation
**Caption:** *WP1 holdout confirmation, 10 gate conditions.* All 10
conditions PASS. cond1 0.925 (≥0.50); cond2 0.074 (≥0.02); cond3
0.126–0.263 (≥0.10); cond4 +0.0266 (≥0.010); cond5 CI lower 0.0140
(>0); cond6 CFR 0.0575 vs 0.0633 (≤); cond7 CVaR 0.092 vs 0.075
(≥ single − 0.005); cond8 J_pred 0.719 vs 0.649 (≥); cond9 Spearman
0.724 (≥0.25); cond10 0 violations / 2400 assignments.

## T14. Cross-dataset replication
**Caption:** *WP2 cross-dataset replication.* MoNuSeg STRONG_SUPPORT
(9/9): `V_ref_frac` 0.070–0.160, consensus gain +0.015 (CI lower
0.007), best Spearman 0.827. IDRiD FAIL (4/5 directional, 2/4 strong):
consensus gain −0.012 (CI lower −0.032), CFR 0.71–1.00.

## T15. Ablations
**Caption:** *Ablations.* (a) M2 model-class ablation (entropy,
disagreement, logistic, gradient boosting, MLP) on validation; (b) M2
feature ablation (leave-one-out over the four test-time-safe features);
(c) M1 policy-class ablation (always-k1, always-k3, always-k6,
handcrafted, logistic, gradient boosting); (d) prompt-config ablation
on `V_ref_frac` and consensus gain.

## T16. Failure taxonomy
**Caption:** *Failure taxonomy.* Four failure modes: small sparse
target, diffuse matching map, prompt-interface bottleneck,
reference-induced catastrophic failure. Per-mode counts, mean `V_ref`,
mean Dice, and representative query IDs (rule-selected).

## T17. Runtime
**Caption:** *Runtime and cost.* SAM calls, DINOv2 matching passes
(cached / uncached), wall-clock seconds, GPU-hours per work package.
WP1: 9,600 SAM calls, 1,526.5 s, 0 GPU-hours. Total Phase T compute
well under the 195 GPU-hour cap.

## T18. Claim registry
**Caption:** *Claim registry summary.* Five allowed claims (a–e) with
evidence files; ten permanently prohibited claims. The claim-boundary
audit scans every Phase T artifact for prohibited tokens before the
final state transition.
