# Related Work

We organize prior work into five clusters and state, for each cluster, what
the cluster does, what we do differently, and the Phase T evidence that
supports the differentiation. The boundary statements follow the Phase T
novelty audit. We do not rank external methods as "fully outperformed"
(that is a prohibited claim); we report head-to-head metrics under a
matched protocol where reproduction is possible, and document mechanisms
where it is not.

## Prompt-robustness methods (GF-SAM, SynPo, AlignSAM, PPD)

**GF-SAM** builds a graph over candidate references or prompts and uses
graph grouping to select a coherent reference set, with the goal of
improving prompt quality. The object of GF-SAM's optimization is
*selection*: "which references are best?". Our object is *distribution*:
we do not select references — every query receives the full per-query pool
of six references, resampled across ten seeds — and we measure the
distribution of predictions induced by resampling, not the quality of a
selected subset. The distinction matters empirically: on the WP1 holdout
the reference fraction of total variance is 0.061–0.263, and M1 shows
that an optimized dynamic budget over references collapses to "always
k=6", i.e. there is no selection gain to be had beyond using the full pool.
The phenomenon is distributional, not selectional.

**SynPo** synthesizes positive and negative point prompts (including
synthetic negatives) to augment the prompt space and improve SAM's
robustness to point placement. We use only *real* references and a frozen
candidate pool (DINOv2-L patch-grid centres); no prompts are synthesized
and no synthetic negatives are constructed. Our negative results concern
natural reference-induced instability, not adversarial or synthetic prompt
perturbations. The candidate pool is frozen across all Phase T runs; the
only varying factor is the reference subset drawn per seed. The WP2 IDRiD
failure (consensus gain −0.012) is attributable to the dataset and
reference pool, not to prompt construction, since prompt construction is
identical to MoNuSeg where consensus helps (+0.015).

**AlignSAM** uses reinforcement learning to iteratively refine prompts
during SAM inference, treating prompt improvement as a sequential decision
problem. We do no RL, no iterative refinement, and no SAM fine-tuning.
The pipeline is single-call SAM-B with a frozen matcher. Our mitigations
operate on the reference axis, not the prompt axis: M2 abstains on
high-risk reference-conditioned episodes, and M1 allocates reference
budget. Neither refines prompts. M2's holdout confirmation (6/6, CFR
reduction 79.2%) is achieved with a static logistic regression, with no
policy gradient, reward, or environment interaction, demonstrating that
the predictable component of failure risk can be captured without
iterative prompting.

**Point Prompt Defender (PPD)** filters or defends against *adversarial*
point prompts, removing points that are hostile perturbations. We study
*natural* reference-induced instability: every point prompt is a
legitimate DINOv2-matched candidate, none are adversarial. Our M1
"filtering" is over *reference subsets*, not over points, and it fails
(NO_GAIN): the best policy selects `lambda_c=lambda_r=0`, i.e. no
filtering of references is beneficial beyond using all six. This is a
negative result that bounds the filtering paradigm for *natural*
instability under this pipeline.

## Reference-aware methods (ProSAM, MAUP, SAMIX)

**ProSAM** models prompt robustness probabilistically, learning prompt
distributions that are stable under perturbation, with the goal of
producing prompts that SAM answers consistently. We do not learn prompts
and do not model a prompt distribution to be *made* robust. We treat the
prompt distribution induced by reference resampling as the *object of
measurement* — we report its variance (`V_ref`), tail (`CVaR_0.1`), and
failure mass (CFR) — and we predict failure risk from test-time-safe
features. Our probability model is over *failure*, not over *prompts*.
M2's catastrophic-trained model produces a calibrated probability of
catastrophic failure per episode (AUPRC 0.434 on validation against
prevalence 0.117; 0.528 on holdout against prevalence 0.050; ECE 0.043
on validation, 0.027 on holdout), not a probability over prompts.

**MAUP** uses multi-scale anchors and uncertainty to decide *how many*
prompts to use, framing prompt quantity as an uncertainty-driven choice.
We share the question "how many references?" (M1 dynamic budget) but
differ on three points: (i) we use single-scale DINOv2-L features, not
multi-scale anchors; (ii) our budget decision is cost-quality-optimized,
not uncertainty-quantile-driven; and (iii) we report a *negative result*:
the cost-quality optimum is `lambda=0`, i.e. always k=6. All five M1
policy variants (always-k1, always-k3, always-k6, handcrafted threshold,
logistic regression, gradient boosting) select `lambda_c=lambda_r=0`;
`M1-2_always_k6` Pareto-dominates the learned policies on validation.
The cost-quality trade-off is unfavorable under our pipeline, which is
the opposite of MAUP's positive prompt-quantity result; we report this
as a failure boundary, not as a competing positive method.

**SAMIX** selects or mixes references to produce a stronger composite
prompt for SAM. We do not select or mix references to build a stronger
prompt. We use mean consensus (B2_mean) over the full reference set as a
*baseline*, and our method contributions (M1, M2) operate on the *risk*
of the reference set, not on improving it. Crucially, we show that on
IDRiD, mixing/consensus *hurts* Dice (−0.012), contradicting the SAMIX
premise that reference aggregation is beneficial. The WP2 IDRiD result
(d3 FAIL, s1 FAIL, s2 FAIL; consensus Dice 0.090 versus single 0.102) is
direct evidence that reference aggregation is not universally beneficial,
a finding that SAMIX-style methods do not report.

## Few-shot segmentation

Few-shot segmentation (PANet, PPNet, and related) trains a segmenter with
a support set of labeled images and generalizes to a query; the support
set is a *training* signal. We do not train a segmenter. SAM-B is frozen;
the references are used only to condition point prompts via DINOv2
matching. Our "support set" is a prompt-conditioning set, not a learning
set, and we benchmark the *instability* of this paradigm, which few-shot
works typically do not measure. All Phase T runs use frozen SAM-B with
no gradient updates; WP1 (10/10) and WP2 MoNuSeg (9/9) replicate the
instability phenomenon with zero segmenter training, isolating
reference-induced variance from any learning-induced variance.

## Selective prediction and abstention

Selective prediction and abstention methods learn a confidence function
and abstain on low-confidence inputs to improve a risk-coverage curve.
M2 overlaps with this cluster, and the overlap is the largest among the
ten novelty-audit items. The differentiation is fourfold. First, the
*object of abstention* is a reference-conditioned episode, not an input
in general; the risk we model is reference-induced catastrophic failure,
not generic uncertainty. Second, the *dual-model design is forced by an
empirically measured anti-correlation* between instability (high
best-worst gap) and catastrophic failure (low Dice): a single predictor
cannot both flag instability and flag catastrophic risk, so we train two
logistic regressions on the same four features with different labels.
The catastrophic-trained model's instability AUROC is 0.060 (near-random)
while the instability-trained model's AUROC is 0.944 — an empirical
demonstration that the two axes are not the same. Third, the predictor
uses *test-time-safe features only* (the four disagreement features are
computable without query ground truth), enabling deployment. Fourth, the
result is *pre-registered holdout confirmation* (6/6, CFR reduction
79.2%), not a validation-split claim.

## Set aggregation and consensus

Set aggregation (DeepSets, Set Transformer, mean/median/trimmed ensembles)
combines a variable-size set into a fixed representation. Prior Phase S
work in this pipeline already tested six deterministic aggregators
(B1–B6) plus learned weighting and set-conditioned gates, all failing to
beat B2_mean. Phase T's contribution here is *not* a new aggregator; it
is (i) holdout confirmation that B2_mean remains the strongest automatic
baseline (WP1 cond4/cond5: consensus gain +0.027, CI lower 0.014), and
(ii) the M1 result that *dynamic* set sizing (a form of adaptive
aggregation) also fails. The aggregation question is closed negatively
under this pipeline: no adaptive or learned aggregator beats static mean
consensus, and the failure is documented as a failure-boundary
contribution.

## Summary of differentiation

Across the ten items above, Phase T does not select references (vs.
GF-SAM, SAMIX), does not synthesize prompts (vs. SynPo), does not refine
prompts iteratively (vs. AlignSAM), does not defend against adversarial
points (vs. PPD), does not learn a prompt distribution (vs. ProSAM), does
not train a segmenter (vs. few-shot segmentation), and does not propose
a new set aggregator. What it does is *measure* the prediction
distribution, tail risk, and failure probability under reference
resampling, *bound* the phenomenon cross-dataset, and *mitigate* its worst
outcomes with a dual-model selective predictor that is holdout-confirmed
and a dynamic-budget method that is a documented negative. External
comparator reproduction (GF-SAM, MAUP identified as Tier 2 reproducible)
is best-effort and pending user authorization for code download; we do
not fabricate Tier 3 numerical results, and we do not claim any external
method is "fully outperformed".
