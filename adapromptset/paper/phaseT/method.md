# Method

The method has four parts: (1) the reference-resampling protocol that
generates the prediction distribution we study; (2) the corrected variance
decomposition and tail-risk metric suite that *measure* it; (3) the M2
dual-model selective predictor that *mitigates* the worst outcomes; and
(4) the M1 dynamic-budget policy that we study and report as a negative
result. The frozen DINOv2-L matcher and SAM-B segmenter are inherited
unchanged from prior phases; the only learned components in Phase T are a
logistic regression (M2) and small regressors (M1), all on test-time-safe
features.

## 1. Reference-resampling protocol

**Reference pool.** The reference pool is *train-only*: no validation,
holdout, or test case may serve as a reference, and the pool is
*patient-disjoint* from the query pool. For MoNuSAC the pool is 80 cases /
18 patients, reused from prior phase manifests; for MoNuSeg it is 37 train
cases; for IDRiD it is 383 train cases under one frozen lesion intent.

**References per query.** Each query receives exactly six references drawn
from the train-only pool, patient-disjoint from the query. The six
references are the upper bound on what M1 may select from; the actual
subset size k is drawn from k∈{1,3,6} per the resampling rule below. All
methods (internal B1–B7 families, external comparators, M1, M2) evaluated
on the same query share the *exact same six-reference pool* for that
query; methods differ only in how they aggregate or select from the pool,
not in the pool itself. The k=6 subset *is* the full pool; no method sees
more than six references.

**Resampling seeds.** Ten resampling seeds are used on train (reused from
the prior 10-seed allocation); each seed produces a different k-subset for
each query, enabling within-query reference-variance estimation. Validation
and holdout reference allocations are *fixed* at pre-registration and
hash-verified before inference; holdout seeds are generated once and never
regenerated. Cross-dataset (MoNuSeg, IDRiD) uses five resampling seeds on
each dataset's development split, frozen before WP2 inference.

**Reference-query pair constraints.** For every (query, reference) pair:
`reference_case_id != query_case_id`, `reference_patient_id !=
query_patient_id`, `reference_role == "train"`, `query_role !=
"test"`. The constraint is enforced by an audit script on every
regenerated manifest; a failed audit blocks inference. On the WP1 holdout
this audit produced 0 violations across 2,400 reference assignments over
10 seeds.

**Prompt configurations.** Three frozen prompt configs are evaluated:
`P20-N1` (K+=20, K-=1, N_BOTTOMK), `P30-N0` (K+=30, K-=0, N_NONE), and
`P50-N0` (K+=50, K-=0, N_NONE). Matching normalization is z-score
foreground/background; candidate selection is P_TOPK positive /
N_BOTTOMK or N_NONE negative. SAM-B is invoked in a single call per
consensus aggregation (the B2_mean aggregator fuses reference-conditioned
prompts into one prompt set before the single SAM call), with the
multimask output rule frozen.

## 2. Corrected variance decomposition and tail-risk metrics

**Statistical unit.** The statistical unit is the *query/case*, not the
reference episode. References are repeated measurements and must not be
treated as independent samples. All aggregate statistics are computed by
first averaging within-query across references (or across resampling
seeds), then computing the cross-query statistic. For MoNuSAC, where
patients contribute multiple queries, the patient is the independent
unit and a patient-cluster bootstrap is used.

**Variance decomposition.** We decompose total variance into a
query-attributable component and a reference-attributable component:

```
V_ref      = (1/N) Σ_q Var_S [ D(q, S) ]
V_query    = (1/N) Σ_q Var_S [ D(q, ·) averaged over S ]
V_ref_frac = V_ref / (V_query + V_ref)
```

where S indexes reference subsets and D(q, S) is the Dice of query q under
reference subset S. `V_ref_frac` is the fraction of total variance
attributable to reference choice. Prior phases established
`V_ref_frac ∈ [0.1376, 0.3312]` on validation across the three frozen
prompt configs; WP1 must reconfirm `V_ref_frac ≥ 0.10` on holdout.

**Pairwise metrics.** `MAD_ref` is the mean absolute pairwise reference
deviation. `J_pred` is the prediction-consistency Jaccard: the expected
IoU between predictions produced by two different reference subsets for
the same query, `J_pred = E_{S_i, S_j} IoU(M̂(q, S_i), M̂(q, S_j))`. Higher
`J_pred` means more reference-stable predictions.

**Tail-risk metrics.** `CVaR_0.1` is the mean Dice of the worst 10% of
(query, reference-subset) episodes; `CVaR_0.2` is the worst 20%. `CFR`
(catastrophic-failure rate) is the fraction of episodes with Dice < 0.1,
where 0.1 is the *frozen* catastrophic threshold (not retunable after
holdout). The *worst-subset Dice* is the minimum per-query Dice across
all reference subsets for that query, averaged over queries. The
*best-worst gap* is the per-query difference between the best and worst
reference-subset Dice; we report its mean and the fraction of queries
with gap ≥ 0.05 (the WP1 gate threshold).

**Disagreement features.** Four *test-time-safe* features (computable
without query ground truth) are computed per episode:
`matching_map_disagreement`, `candidate_jaccard_mean_k1`,
`prompt_overlap_mean_k1`, and `prediction_jaccard_k1_within`. The last is
the within-query version of `J_pred` at k=1 and is the strongest
instability predictor on the WP1 holdout (Spearman |ρ|=0.724 against the
best-worst gap).

**Paired bootstrap.** For paired method comparisons (e.g., B2 mean vs B1
random single), we compute the per-query paired difference `d_q`, resample
queries (patient-cluster for MoNuSAC; query-level for MoNuSeg / IDRiD)
with replacement B=10,000 times, and take the 2.5th and 97.5th percentiles
as the 95% CI. A method is declared significantly better if the CI lower
bound > 0. When more than two methods are compared simultaneously, the
Holm step-down correction is applied at familywise α=0.05.

## 3. M2 dual-model selective prediction

M2 is the positive-mitigation contribution of this paper. The goal is a
deployable selective predictor that flags reference-conditioned episodes
likely to end in catastrophic failure, so the system can abstain at
deployment.

**Why dual-model — the anti-correlation finding.** A naive design would
train a single predictor on a single risk label. We find this is
impossible on our feature space. Instability (a high best-worst gap) and
catastrophic failure (a low Dice) are *anti-correlated*: episodes where
all references fail similarly have a low gap but a low Dice, and episodes
where references disagree have a high gap but a higher mean Dice. A
predictor trained on the instability label achieves AUROC 0.944 on
validation but a catastrophic AUPRC of only 0.071 (below prevalence
0.117); a predictor trained on the catastrophic-failure label achieves
AUPRC 0.434 (above prevalence) but an instability AUROC of 0.060
(near-random). The two axes are not the same, and a single model cannot
satisfy both. This finding *forces* the dual-model design: two logistic
regressions on the same four test-time-safe features, trained with
different labels. The dual-model design is a structural contribution, not
an engineering preference.

**Labels.** The instability label is the binary indicator that the
episode's best-worst gap exceeds the train-set 90th percentile. The
catastrophic label is the binary indicator that the episode's Dice is
below 0.1. Both labels are constructed on *train only*; no validation or
holdout label is used for model selection beyond the pre-registered gate.

**Model ladder.** Five model variants are evaluated: M2-0 entropy
threshold, M2-1 disagreement threshold, M2-2 logistic regression, M2-3
gradient boosting, M2-4 small MLP. All operate on the four test-time-safe
features. The frozen dual predictor selects M2-2 logistic regression on
both axes: it is deliberately simple for calibration, and its ECE meets
the gate on both validation and holdout.

**Gate conditions (5 on validation, 6 on holdout).** cond1: instability
AUROC ≥ 0.70. cond2: catastrophic AUPRC > prevalence. cond3: ECE ≤ 0.10
(or ≤ 0.08 after calibration). cond4: CFR reduction ≥ 30% at 80%
coverage. cond5: mean Dice not decrease at 80% coverage. cond6
(holdout only): cond1–5 all pass on holdout, directionally consistent with
validation.

**Output.** At deployment the catastrophic-trained model produces a
calibrated probability of catastrophic failure per episode; the system
abstains when the probability exceeds a coverage-derived threshold. The
instability-trained model is reported for the AUROC gate and as a
diagnostic; it does not drive abstention.

## 4. M1 dynamic reference budget (negative result)

M1 is the documented-negative route. The goal is a cost-quality-optimized
policy that chooses, per query, how many references to use, trading the
cost of reference conditioning against segmentation quality.

**Policy ladder.** Six variants are evaluated: M1-0 always k=1, M1-1
always k=3, M1-2 always k=6, M1-3 handcrafted threshold, M1-4 logistic
regression, M1-5 gradient boosting. Each policy outputs a per-query
reference count k∈{1,3,6}.

**Cost-quality utility.** The policy optimizes
`U(q,k) = (1-λ_c-λ_r)·Dice(q,k) - λ_c·Cost(q,k) - λ_r·Risk(q,k)`, where
λ_c and λ_r are cost and risk weights selected by cross-validation on
train. The CV selects `λ_c=λ_r=0` for *all* five policy variants — the
policy collapses to "maximize Dice", which means "always k=6".

**Gate conditions.** Cost-saving mode (vs k=6): cs1 Dice not below k=6
by more than 0.003; cs2 CVaR not decrease by more than 0.005; cs3 CFR not
increase; cs4 avg_k < 4.5 (≥25% reduction). Quality-cost mode (vs k=3):
qc1 Dice ≥ k=3 + 0.0075 *and* avg_k in [2.5, 3.5]; qc2 CVaR improve vs
k=3; qc3 CFR not increase vs k=3. Pass if either mode passes.

**Outcome.** The best policy is M1-3 (handcrafted threshold): val Dice
0.4571, avg_k 5.63, CVaR 0.0537, CFR 0.1203. cs1–cs3 pass (Dice diff
−0.0015; CVaR diff +0.0047; CFR diff −0.0049), but cs4 fails (avg_k 5.63,
only 6.1% reduction versus the 25% gate). Quality-cost mode fails on all
three conditions: qc1 fails because avg_k 5.63 is outside [2.5, 3.5] even
though Dice beats k=3; qc2 and qc3 fail. The M1 decision is
`PHASET_M1_DYNAMIC_BUDGET_NO_GAIN`. The scientific finding is that
dynamic budgeting cannot meaningfully reduce references without hurting
Dice under this pipeline, and we report this as a failure-boundary
contribution rather than discarding it.

## 5. Pre-registration, freeze list, and holdout policy

All of the above is frozen at pre-registration. The freeze list includes
the reference pool, k∈{1,3,6}, the resampling seeds, the three prompt
configs, matching normalization, candidate selection, the SAM config, all
metric definitions, the catastrophic threshold (0.1), the bootstrap
procedure (10,000 iterations, patient-cluster for MoNuSAC), and the
qualitative-sampling rules (worst-5, best-5, median-5 by V_ref — never
hand-picked). The MoNuSAC holdout may be observed exactly twice: once for
the WP1 confirmatory benchmark, once for the WP3 final-method one-shot
confirmation. Each holdout run is preceded by hash verification of the
frozen manifests and configs; no parameter may be retuned between the two
runs based on holdout observations. The claim-boundary audit scans every
holdout report for prohibited tokens before any state transition.
