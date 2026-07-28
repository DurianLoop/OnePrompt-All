# Benchmark Protocol

This section describes the benchmark protocol that produces the
reference-induced instability measurements. The protocol is identical
across WP1 (MoNuSAC holdout), WP2 (MoNuSeg / IDRiD cross-dataset), and
WP3 (M1 / M2 risk-aware mitigation); only the dataset and split roles
differ. The protocol is frozen at pre-registration and hash-verified
before any holdout inference; no parameter on the freeze list may be
changed based on holdout observations.

## Datasets and roles

**MoNuSAC (primary).** Multi-organ nuclear segmentation. The dataset is
split into 80 train cases (model training, threshold selection, M1/M2
policy fitting, reference-budget utility labels), 41 validation cases
(method selection and gate evaluation), 40 holdout cases (WP1
confirmatory benchmark; final-method one-shot confirmation), and 101
formal-test cases (LOCKED and unused for the entire Phase T program).
Reference pool: 80 cases / 18 patients, train-only, patient-disjoint
from queries. The 40 holdout queries span 8 patients, patient-disjoint
from train and validation.

**MoNuSeg (secondary pathology morphology).** 37 train cases serve as
the reference pool; 30 strict-validation queries are used for WP2
cross-dataset replication. Patient IDs in MoNuSeg are unreliable, so
case_id is used as the patient_id proxy, and the reference-query pair
constraint `reference_case_id != query_case_id` is still enforced. The
test split is locked and unused.

**IDRiD (tertiary cross-domain retinal).** 383 train cases serve as the
reference pool for one frozen lesion intent; 30 validation queries are
used for WP2 cross-dataset replication. The test split is locked and
unused.

**Exposure control.** No patient/case leakage is permitted between
train / validation / holdout / test within any dataset. The reference
pool is train-only and patient-disjoint from queries within each
episode. Validation and holdout reference allocations are fixed; no
re-randomization after gate evaluation. The Phase X historical 10-query
subset is excluded from Phase T primary statistics.

## Reference construction

Each query receives six references drawn from the train-only reference
pool, patient-disjoint from the query. The six references are the upper
bound on what M1 may select from; the actual subset size k is drawn from
k∈{1,3,6}. The k=1 subsets are the six individual references (subset_ids
0..5); the k=3 subset is the mean consensus over the first three; the k=6
subset is the mean consensus over all six. All methods evaluated on the
same query share the exact same six-reference pool; methods differ only
in aggregation or selection. Ten resampling seeds are used on MoNuSAC
train (reused from the prior 10-seed allocation); five seeds on MoNuSeg
and IDRiD development splits. Holdout reference allocation is frozen in
a hash-verified manifest and never regenerated.

## Frozen pipeline

**Matcher.** DINOv2-L features with z-score foreground/background
normalization, reused from the per-case feature cache. Matching produces
a per-reference foreground matching map `s_fg`, background map `s_bg`,
and contrast `C_r`.

**Candidate selection.** P_TOPK positive selector (top-K foreground
candidates) and N_BOTTOMK or N_NONE negative selector, frozen across
runs. Three prompt configs are evaluated: P20-N1 (K+=20, K-=1,
N_BOTTOMK), P30-N0 (K+=30, K-=0, N_NONE), P50-N0 (K+=50, K-=0, N_NONE).

**Segmenter.** SAM-B, single call per consensus aggregation, multimask
output rule frozen. Mean consensus (B2_mean) fuses reference-conditioned
prompts into one prompt set before the single SAM call; this makes the
consensus baseline a *single-SAM-call* baseline, controlling cost when
comparing against a single-reference baseline.

## Metrics

The full metric suite is defined in the Phase T metric protocol. Mean
performance: Dice, IoU, AJI, component recall, precision/recall.
Reference-instability: `V_query`, `V_ref`, `V_ref_frac`, `MAD_ref`,
`J_pred`. Tail risk: `CVaR_0.1`, `CVaR_0.2`, `CFR` (Dice < 0.1),
worst-subset Dice, best-worst gap. Cost: reference count, DINOv2 matching
passes (cached and uncached), SAM calls, runtime seconds, GPU-hours.

The statistical unit is the query/case. References are repeated
measurements and are not treated as independent samples; all aggregate
statistics first average within-query across references (or seeds), then
compute the cross-query statistic. For MoNuSAC, where patients contribute
multiple queries, the patient is the independent unit and a
patient-cluster bootstrap is used. Paired bootstrap CIs use B=10,000
iterations and the 2.5th / 97.5th percentiles; multiple comparisons use
Holm's step-down correction at familywise α=0.05. Every results table
reports mean, paired difference, paired bootstrap 95% CI, win/tie/loss
counts, effect size (Cohen's d), within-query reference std, `V_ref`,
`V_query`, `V_ref_frac`, `MAD_ref`, `J_pred`, `CVaR_0.1`, `CVaR_0.2`,
CFR, and seed variability. Reporting mean Dice without variance or
tail-risk evidence is a prohibited over-claim.

## k-scaling design

| k | subsets per (query, seed) | source | subset_id |
|---|---|---|---|
| 1 | 6 (individual references) | per-reference s_fg pool | 0..5 |
| 3 | 1 (mean consensus, first 3) | B2_mean contrast | first3 |
| 6 | 1 (mean consensus, all 6) | B2_mean contrast | all6 |

This design isolates the effect of reference count: k=1 measures
single-reference behaviour, k=3 measures intermediate consensus, and k=6
measures full-pool consensus. The same query under the same seed sees
the same six references at every k; only the aggregation differs.

## Holdout design and run policy

The MoNuSAC holdout may be observed exactly twice: (1) WP1 confirmatory
benchmark (B1–B7, k∈{1,3,6}, three prompt configs) under the frozen
protocol; (2) WP3 final-method one-shot confirmation of a single frozen
M1 or M2 model on the same 40 queries, after the validation gate passes.
Each holdout run is independent and preceded by its own hash
verification of the frozen manifests and configs. The WP1 run produces
9,600 SAM calls (40 queries × 10 seeds × 3 configs × 8 subsets per
(query, seed, config) on average). No parameter on the freeze list may
be retuned between the two runs based on holdout observations; the
catastrophic threshold (0.1), bootstrap iteration count (10,000), and
qualitative-sampling rules are all frozen.

## Cost accounting

Each reference induces one DINOv2-L matching pass (cached per case_id
and reused across seeds), one candidate-pool construction (cached per
case_id), and one SAM-B mask decode *per consensus aggregation* (not per
reference; consensus is single-SAM-call by construction for B2_mean).
The compute ledger records SAM calls, matching passes (cached +
uncached), runtime seconds, and GPU-hours. Phase T used 0 GPU-hours on
WP1 (CPU inference) against a 195 GPU-hour program cap; the remaining
budget is reserved for WP2 / WP3 / WP4.

## Prohibited operations

(1) No reference selection based on query ground truth at inference
time. (2) No reference re-ordering to favor a method. (3) No
reference-pool expansion after the holdout hash is verified. (4) No
cross-dataset reference sharing (a MoNuSAC query never receives a
MoNuSeg reference, and vice versa). (5) No test-set case may enter the
reference pool under any circumstance. (6) No retuning of any frozen
parameter after holdout observation; no selective reporting of holdout
subsets; no threshold retuning based on observed holdout Dice.
