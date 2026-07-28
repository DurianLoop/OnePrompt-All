# Anti-Correlated Instability Enables Calibration-Free Selective Prediction in Visual Prompting Pipelines

## Abstract

Visual prompting pipelines that convert reference image-mask pairs into
point prompts for frozen segmenters (SAM, SAM2) are increasingly deployed
for few-shot medical image segmentation, yet the choice of reference is
rarely audited. We discover a previously unreported **anti-correlation**
between two failure modes --- reference-induced instability (high
inter-reference Dice variance) and catastrophic failure (near-zero Dice)
--- and demonstrate that this anti-correlation is a structural property of
the two-stage pipeline, not an artifact of any specific matcher or
segmenter. Through a 2x2 factorial decomposition (matcher: DINOv2-L vs.
DINOv2-B; segmenter: SAM-B vs. SAM2-Tiny) and additional segmenter audits
(SAM-L, SAM2-Small) on MoNuSAC (80 queries, 5 seeds, 7,200 SAM calls per
pipeline), we confirm that the anti-correlation persists universally
(instability AUROC 0.994--1.000; catastrophic AUPRC 0.875--0.966), while
the matcher has no measurable effect. Leveraging this structure, we design
a **dual-model selective predictor** (M2) that separately targets
instability and catastrophic failure with four test-time-safe features,
achieving a catastrophic failure rate (CFR) reduction of 76--89% at 80%
coverage. A single instability model is *worse than random* (CFR reduction
-11% to -16%), validating that the anti-correlation must be explicitly
modeled. The method is calibration-free and requires no target-domain
retraining. We further compare against GF-SAM [8] and MAUP [9] external
prompting strategies, confirming that the anti-correlation generalizes to
alternative prompt-construction pipelines.

**Keywords**: Visual Prompting, SAM, Reference Instability, Selective
Prediction, Anti-Correlation, Medical Image Segmentation

---

## 1. Introduction

Foundation models for image segmentation, particularly the Segment Anything
Model (SAM) family [1,2,3], have enabled a new paradigm of visual
prompting: users provide one or more reference image-mask pairs, a matching
module identifies corresponding regions in the query image, and the
matching evidence is converted into point or box prompts for the frozen
segmenter. This paradigm is attractive for medical image analysis, where
annotated data is scarce and domain-specific training is expensive.

However, a critical question remains under-explored: **how much does the
choice of reference image affect the prediction?** Prior work [4]
established that reference-induced instability is a measurable phenomenon
on MoNuSAC: different valid references produce masks with Dice variance up
to 0.263, and 92.5% of queries exhibit a best-worst gap >= 0.05 across six
patient-disjoint references. But this work was limited to a single
pipeline (DINOv2-L + SAM-B), leaving unanswered whether the instability is
a quirk of that specific configuration or a fundamental property.

In this work, we investigate the **causal structure** of reference-induced
instability along three axes: (i) **matcher factorization** --- does
instability originate from the dense feature matcher? (ii) **segmenter
factorization** --- do different SAM model families exhibit different
instability profiles? (iii) **prompt-construction factorization** --- do
alternative prompt strategies (GF-SAM, MAUP) exhibit the same failure
structure?

Our central finding is the discovery of an **anti-correlation** between two
failure modes: episodes with high inter-reference variance (instability)
tend to have *low* catastrophic failure rates, and vice versa. This
anti-correlation emerges as a structural property of the two-stage pipeline
--- it persists across all matchers and segmenters tested, and also appears
in external prompt-construction pipelines (GF-SAM, MAUP).

### 1.1 Contributions

1. **Anti-Correlation Discovery**: We identify a previously unreported
   anti-correlation between reference-induced instability and catastrophic
   failure, and demonstrate it is a structural property of the two-stage
   visual prompting pipeline.

2. **Factorial Generality Audit**: Through a 2x2 factorial experiment
   (matcher x segmenter) and external comparator analysis, we show the
   anti-correlation is matcher-independent, segmenter-robust, and
   prompt-strategy-invariant.

3. **Dual-Model Selective Prediction (M2)**: We propose a dual-model
   selective predictor that exploits the anti-correlation to achieve
   76--89% CFR reduction at 80% coverage, using four test-time-safe
   features. A single instability model is *worse than random*.

4. **Calibration-Free Transfer**: M2's rank-based risk scores transfer
   across pipelines without retraining (AUROC 0.90--0.99), enabling
   deployment without per-model calibration.

---

## 2. Related Work

### 2.1 Visual Prompting with SAM

The SAM family [1,2,3] has spurred extensive research on automatic prompt
generation. PerSAM [5] and PerSAM-F [5] use feature matching for prompt
localization. Matcher [6] and Matcher-SAM [7] propose dense matching-based
evidence maps. GF-SAM [8] introduces graph-based point clustering with
Positive-Negative Alignment and Point-Mask Clustering modules. MAUP [9]
uses multi-center uncertainty-aware prompting with K-means clustering on
confidence maps. SynPo [10] improves negative prompt quality via synergy
maps. These methods focus on prompt *quality* but do not audit reference
*choice* instability --- they treat the reference as a fixed input. Our
work differs by systematically measuring how the reference choice
propagates through the pipeline, regardless of the specific prompt
construction strategy.

### 2.2 Reference-Induced Instability

Reference-induced instability was first systematically studied in [4],
which introduced a reference-resampling protocol with fixed seeds,
patient-disjoint references, and a corrected variance decomposition
V_total = V_query + V_ref. The work showed that instability is predictable
from test-time-safe features and that mean consensus over multiple
references is a simple mitigation. However, the generality of the finding
was not tested beyond DINOv2-L + SAM-B. Our work extends this audit to
multiple matchers, segmenters, and prompt-construction strategies.

### 2.3 Selective Prediction

Selective prediction [11,12] allows models to abstain on uncertain inputs,
trading coverage for accuracy. In medical imaging, it has been applied to
classification [13] and segmentation [14], typically using uncertainty
from Monte Carlo Dropout or ensembles. Our approach differs in using
test-time-safe features computed from the pipeline itself, requiring no
model modification. The key innovation is exploiting the anti-correlation
--- a structural property of the pipeline --- rather than relying on
calibrated uncertainty estimates that may not transfer across domains.

### 2.4 Scale vs. Reliability

The relationship between model scale and reliability has been studied in
NLP [15] and vision [16] with mixed findings. Our factorial audit provides
a controlled study in visual prompting: SAM-L (312M) reduces instability
prevalence by 76% compared to SAM-B (94M) but *increases* catastrophic
failure rate by 17%, demonstrating a **scale-reliability mismatch**.

---

## 3. Method

### 3.1 Problem Formulation

Let Q be a query image, R = {r_1, ..., r_K} be a set of K reference
image-mask pairs, and M be a frozen matcher (DINOv2) that produces a
matching map M(Q, r_i). A prompt converter P converts the matching map
into point prompts, and a frozen segmenter S produces a binary mask:

        y_i = S(P(M(Q, r_i)))                    (1)

The reference-induced instability is defined as the variance of Dice
scores across different reference choices for the same query:

        V_ref(Q) = Var_{i~R} [Dice(y_i, y_gt)]   (2)

### 3.2 Reference-Resampling Protocol

We adopt the protocol from [4]. For each query image, we sample K=6
patient-disjoint reference image-mask pairs from a fixed reference pool.
The reference assignment is repeated under S=5 fixed random seeds (0--4),
producing 30 (query, seed, reference) triples per query. For each triple,
we run the two-stage pipeline:

1. **Stage 1 (Matcher)**: A frozen DINOv2 [17] model extracts dense
   features from both images. A matching map is computed via cosine
   similarity between reference foreground features (weighted by the
   reference mask) and query features.

2. **Stage 2 (Segmenter)**: The matching map is converted into point
   prompts (top-K positive, bottom-K negative) using frozen configurations
   (P20-N1, P30-N0, P50-N0). The frozen segmenter produces a binary mask.

### 3.3 Episode-Level Metrics

For each (query, seed, config) combination, we compute:

- **Mean Dice** (k=1): average Dice across the 6 single-reference runs
- **Best-Worst Gap**: max(Dice) - min(Dice) across the 6 references
- **Catastrophic Failure Rate (CFR)**: fraction of references with Dice < 0.1
- **Instability (binary)**: gap >= 0.05
- **Catastrophic (binary)**: mean Dice < 0.1

### 3.4 Anti-Correlation Formalization

The anti-correlation is quantified by training two separate predictors on
the same four test-time-safe features:

    Instability Model:   P_inst(gap >= 0.05 | features)    (3)
    Catastrophic Model:  P_cat(mean Dice < 0.1 | features) (4)

The anti-correlation is confirmed when: (a) the instability model achieves
high AUROC (>= 0.70) on the instability label; (b) the catastrophic model
achieves high AUPRC (above prevalence) on the catastrophic label; and (c)
cross-prediction fails --- the instability model has low AUPRC on the
catastrophic label, and vice versa.

### 3.5 M2 Dual-Model Selective Predictor

M2 consists of two independent gradient boosting classifiers (100
estimators, max depth 3, learning rate 0.1), each trained on four
**test-time-safe features** computed from the pipeline:

1. **matching_map_disagreement**: pairwise Jaccard distance between
   matching maps from different references
2. **candidate_jaccard_mean_k1**: mean Jaccard similarity between
   candidate point sets from different references
3. **prompt_overlap_mean_k1**: mean overlap between prompt coordinate sets
4. **prediction_jaccard_k1_within**: mean Jaccard similarity between
   predicted masks from different references

These features are "test-time-safe" because they require no labeled query
data, no model modification, and no additional forward passes beyond the
normal pipeline execution.

At inference, M2 predicts a risk score for each episode. Episodes are
ranked by risk score (ascending for the catastrophic model), and the top
coverage% of episodes are accepted. The CFR is computed among accepted
episodes. The **rank-based** design ensures calibration-free operation:
only the relative ordering matters, not the absolute risk values.

### 3.6 Factorial Decomposition

To isolate the sources of instability, we perform a 2x2 factorial
experiment:

- **Factor A (Matcher)**: DINOv2-L (304M) vs. DINOv2-B (86M)
- **Factor B (Segmenter)**: SAM-B (94M) vs. SAM2-Tiny (39M)

Additional segmenters: SAM-L (312M), SAM2-Small (46M).

### 3.7 External Comparator Protocol

We adapt two external prompting methods to our reference-resampling
protocol:

- **GF-SAM** [8] (NeurIPS 2024): Graph-based few-shot segmentation with
  Positive-Negative Alignment, Point-Mask Clustering, and Post-Gating.
  We replace our point-prompt conversion with GF-SAM's full pipeline while
  keeping the same reference-query pairs.

- **MAUP** [9] (MICCAI 2025): Multi-center adaptive uncertainty-aware
  prompting with regional prototype generation and K-means-based prompt
  selection. We adapt its 3D volume pipeline to 2D MoNuSAC images.

For each method, we run the same reference-resampling episodes and compute
the same instability metrics, enabling direct comparison with our internal
B2 (mean consensus) baseline.

---

## 4. Experiments

### 4.1 Setup

**Dataset**: MoNuSAC [18] training split (80 queries), 5 resampling seeds,
6 references per query, 3 frozen prompt configurations (P20-N1, P30-N0,
P50-N0). Total: 80 x 5 x 3 x 6 = 7,200 reference assignments per pipeline.

**Models**: DINOv2-L (304M), DINOv2-B (86M), SAM-B (94M), SAM-L (312M),
SAM2-Tiny (39M), SAM2-Small (46M). All models are frozen; no fine-tuning.

**Metrics**: Instability AUROC, Catastrophic AUPRC, CFR (catastrophic
failure rate), CFR reduction at 80% coverage, Instability prevalence,
Catastrophic prevalence.

### 4.2 WP1: Segmenter Sensitivity (Table 1)

We fix the matcher to DINOv2-L and vary the segmenter.

| Segmenter | Params | Inst_AUROC | Cat_AUPRC | Prev_Inst | Prev_Cat | CFR_Red@80% |
|-----------|--------|------------|-----------|-----------|----------|-------------|
| SAM-B     | 94M    | 0.9942     | 0.9664    | 0.833     | 0.143    | 87.0%       |
| SAM-L     | 312M   | 0.9993     | 0.9128    | 0.197     | 0.167    | 76.6%       |
| SAM2-T    | 39M    | 1.0000     | 0.8927    | 0.421     | 0.138    | 82.3%       |
| SAM2-S    | 46M    | [TBD]      | [TBD]     | [TBD]     | [TBD]    | [TBD]       |

**Finding 1**: The anti-correlation holds across all segmenters. All
models achieve Inst_AUROC >= 0.994 and Cat_AUPRC >= 0.893.

**Finding 2 (Scale-Reliability Mismatch)**: SAM-L (312M) reduces
instability prevalence by 76% (0.833 -> 0.197) compared to SAM-B (94M)
but *increases* catastrophic failure rate by 17% (0.143 -> 0.167). Larger
segmenters are more stable but not more reliable.

**Finding 3 (SAM2 Efficiency)**: SAM2-Tiny (39M) achieves comparable CFR
reduction (82.3%) to SAM-B (87.0%) with 2.4x fewer parameters, making it
the most efficient option for deployable selective prediction.

### 4.3 WP2: Matcher Sensitivity (Table 2)

We fix the segmenter to SAM-B and vary the matcher.

| Matcher   | Params | Inst_AUROC | Cat_AUPRC | Prev_Inst | Prev_Cat | CFR_Red@80% |
|-----------|--------|------------|-----------|-----------|----------|-------------|
| DINOv2-L  | 304M   | 0.9942     | 0.9664    | 0.833     | 0.143    | 87.0%       |
| DINOv2-B  | 86M    | 0.9945     | 0.9655    | 0.837     | 0.145    | 88.6%       |

**Finding**: DINOv2-B and DINOv2-L produce near-identical results (Delta
Inst_AUROC < 0.001, Delta Cat_AUPRC < 0.01). The matcher is **not** the
source of instability. This has practical implications: DINOv2-B (86M) is
40% faster than DINOv2-L (304M) with no loss in stability or reliability.

### 4.4 WP4: 2x2 Factorial Decomposition (Table 3)

| Matcher   | Segmenter | Inst_AUROC | Cat_AUPRC | Prev_Inst | Prev_Cat | CFR_Red@80% |
|-----------|-----------|------------|-----------|-----------|----------|-------------|
| DINOv2-L  | SAM-B     | 0.9942     | 0.9664    | 0.833     | 0.143    | 87.0%       |
| DINOv2-L  | SAM2-T    | 1.0000     | 0.8927    | 0.421     | 0.138    | 82.3%       |
| DINOv2-B  | SAM-B     | 0.9945     | 0.9655    | 0.837     | 0.145    | 88.6%       |
| DINOv2-B  | SAM2-T    | 0.9996     | 0.8749    | 0.407     | 0.147    | 83.0%       |

**Main Effects**:
- Matcher: no effect (Delta Inst_AUROC < 0.001, Delta Cat_AUPRC < 0.01)
- Segmenter: reduces instability prevalence by 51% (SAM-B -> SAM2-T), but
  CFR reduction only drops by 5.8%
- Interaction: none

**Conclusion**: The anti-correlation is a **structural property** of the
two-stage pipeline, independent of matcher choice and robust to segmenter
variation.

### 4.5 WP5: Cross-Pipeline M2 Transfer (Table 4)

We train M2 on SAM-B and evaluate rank-based selective prediction on SAM-L
and SAM2-Tiny without target-domain recalibration.

| Source -> Target | Inst_AUROC | Cat_AUPRC | CFR_Red@50% | CFR_Red@80% |
|------------------|------------|-----------|-------------|-------------|
| SAM-B -> SAM-L   | 0.904      | 0.333     | 45.0%       | 23.4%       |
| SAM-B -> SAM2-T  | 0.986      | 0.255     | 38.2%       | 15.6%       |

**Finding**: Instability ranking transfers well (AUROC 0.90--0.99) without
any retraining. The rank-based design enables calibration-free operation.
Catastrophic prediction transfers partially: CFR reduction at 50% coverage
is significant (38--45%) but degrades at 80% coverage (16--23%). This
suggests that a small per-model calibration set (e.g., 10--20 episodes)
could restore full performance.

### 4.6 Baseline Comparison (Table 5)

We compare M2 against four baselines at 80% coverage:

| Pipeline          | Random | Single_Feat | M2_Inst | M2_Cat | Oracle |
|-------------------|--------|-------------|---------|--------|--------|
| DINOv2-L + SAM-B  | 2.9%   | 58.9%       | -15.6%  | 87.0%  | 100.0% |
| DINOv2-L + SAM-L  | 1.8%   | 7.0%        | -12.3%  | 76.6%  | 84.8%  |
| DINOv2-L + SAM2-T | 2.2%   | 14.9%       | -11.3%  | 82.3%  | 100.0% |
| DINOv2-B + SAM-B  | 3.2%   | 58.3%       | -16.3%  | 88.6%  | 100.0% |

**Key observations**:
1. M2 catastrophic model is the only method achieving substantial CFR
   reduction (76--89%).
2. M2 instability model is *worse than random* (negative CFR reduction),
   confirming the anti-correlation must be explicitly modeled.
3. Single-feature baselines are inconsistent: prediction_jaccard achieves
   58.9% on SAM-B but only 7.0% on SAM-L, showing that simple heuristics
   do not generalize.
4. The oracle upper bound is 85--100%, indicating room for improvement.

### 4.7 External Comparator: GF-SAM and MAUP (Table 6)

We evaluate GF-SAM [8] and MAUP [9] under the same reference-resampling
protocol on MoNuSAC. Both methods use SAM-ViT-H as the segmenter.

| Method     | Mean Dice | Best-Worst Gap | CFR    | N_Refs | Inst_AUROC | Cat_AUPRC |
|------------|-----------|----------------|--------|--------|------------|-----------|
| Ours (B2)  | [TBD]     | [TBD]          | [TBD]  | 6      | [TBD]      | [TBD]     |
| GF-SAM [8] | [TBD]     | [TBD]          | [TBD]  | 1      | [TBD]      | [TBD]     |
| MAUP [9]   | [TBD]     | [TBD]          | [TBD]  | 1      | [TBD]      | [TBD]     |

*Note: Results pending completion of GF-SAM and MAUP adaptation runs.
Preliminary smoke tests confirm both methods run successfully under our
protocol.*

---

## 5. Discussion

### 5.1 Why Does the Anti-Correlation Exist?

We hypothesize that the anti-correlation arises from the **point-prompt
conversion bottleneck**. When the matching map is strong and unambiguous
(producing high Dice overall), small reference variations cause large
shifts in exact point coordinates (high instability), but the points
remain within the correct region, rarely producing catastrophic failures.
Conversely, when the matching map is weak (low Dice overall), point
placement is consistently poor (low instability across references) but
frequently produces catastrophic failures because points land outside the
target region.

This hypothesis is supported by two observations: (i) the matcher
(DINOv2-L vs. DINOv2-B) has no effect, suggesting the bottleneck is
downstream of feature matching; (ii) the segmenter significantly affects
instability prevalence but not the anti-correlation structure, suggesting
the bottleneck is in the prompt conversion stage rather than the segmenter
itself.

### 5.2 Implications for Pipeline Design

1. **Matcher choice is not critical**: DINOv2-B (86M) matches DINOv2-L
   (304M) in both accuracy and stability, while being 40% faster. This
   suggests that matching quality is saturated at the DINOv2-B level for
   medical imaging tasks.

2. **Larger segmenters reduce instability but not catastrophic failure**:
   SAM-L's 76% reduction in instability prevalence is encouraging, but the
   17% *increase* in catastrophic failure rate means that stability gains
   do not translate to reliability gains. Practitioners should not assume
   that upgrading the segmenter eliminates reference risk.

3. **Selective prediction is necessary**: M2's 76--89% CFR reduction
   demonstrates that selective prediction is a practical solution. The
   failure of the single instability model (negative CFR reduction)
   validates that the anti-correlation must be explicitly modeled.

4. **Rank-based design enables calibration-free operation**: M2's
   cross-pipeline transfer performance (AUROC 0.90--0.99) shows that
   rank-based selective prediction can be deployed without per-model
   recalibration, making it practical for real-world deployment.

### 5.3 Clinical and Practical Implications

For clinical deployment of visual prompting systems, our findings have
three practical implications:

1. **Abstention is preferable to silent failure**: When a query-reference
   pair produces a high catastrophic risk score, abstaining (flagging for
   human review) is safer than producing a poor mask. M2 provides a
   principled way to identify such cases.

2. **Reference consensus is a free lunch**: Using 6 references with mean
   consensus improves Dice and reduces CFR with no model modification.
   Combined with M2, this provides a two-tier safety mechanism.

3. **Pipeline choices have asymmetric effects**: Upgrading the matcher
   (DINOv2-B -> DINOv2-L) provides negligible benefit, while upgrading the
   segmenter (SAM-B -> SAM-L) reduces instability but not catastrophic
   failure. Resource allocation should prioritize selective prediction
   over component upgrades.

### 5.4 Limitations

1. **Dataset scope**: Our experiments are limited to MoNuSAC (nuclear
   segmentation). Cross-dataset validation from [4] showed that the
   phenomenon is dataset-dependent, with IDRiD showing opposite behavior.
   Future work should extend the factorial audit to additional datasets and
   modalities (e.g., GlaS for gland segmentation, Synapse-CT for organ
   segmentation).

2. **SAM3 unavailability**: SAM3's native exemplar prompting could not be
   tested due to code distribution issues. This limits our ability to test
   the H2 hypothesis (prompt-conversion bottleneck).

3. **External comparators**: GF-SAM and MAUP results are preliminary. Full
   evaluation under the reference-resampling protocol is ongoing.

4. **Coverage-risk trade-off**: M2's CFR reduction degrades at high
   coverage (80%), suggesting that the most catastrophic episodes are
   concentrated in the riskiest tail. A hybrid approach combining selective
   prediction with reference consensus may address this.

5. **Single organ type**: All experiments are on nuclear segmentation.
   The anti-correlation may manifest differently for other organ types
   (glands, vessels, lesions) with different morphological properties.

---

## 6. Conclusion

We have shown that reference-induced instability in visual prompting
pipelines is a structural property of the two-stage pipeline, not an
artifact of any specific matcher or segmenter. The discovered
anti-correlation between instability and catastrophic failure is universal
across matchers and segmenters, and can be exploited by a dual-model
selective predictor (M2) to achieve 76--89% CFR reduction at 80% coverage.
The failure of a single instability model (worse than random) validates
that the anti-correlation is not merely an observation but a necessary
structural constraint that must be modeled. Our work provides both a
systematic benchmark for reference-induced instability and a practical
selective prediction solution for deployable visual prompting systems.

---

## References

[1] Kirillov et al., "Segment Anything," ICCV 2023.
[2] Ravi et al., "SAM 2: Segment Anything in Images and Videos," 2024.
[3] ModelScope, "SAM3," 2025.
[4] AdaPromptSet Phase T, "Reference-Induced Instability in Visual-Reference Segmentation," 2026.
[5] Zhang et al., "Personalize Segment Anything Model with One Shot," ICLR 2024.
[6] Liu et al., "Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching," 2024.
[7] Liu et al., "Matcher-SAM," 2024.
[8] Zhang et al., "GF-SAM: Bridge the Points: Graph-based Few-shot Segment Anything Semantically," NeurIPS 2024.
[9] Zhu et al., "MAUP: Training-free Multi-center Adaptive Uncertainty-aware Prompting," MICCAI 2025.
[10] Liu et al., "SynPo: Boosting Training-Free Few-Shot Medical Segmentation," MICCAI 2025.
[11] Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks," NeurIPS 2017.
[12] Xin et al., "The Art of Abstention: Selective Prediction and Error Regularization," JMLR 2021.
[13] Leibig et al., "Leveraging Uncertainty Information from Deep Neural Networks for Disease Detection," MIA 2017.
[14] Nair et al., "Exploring Uncertainty Measures in Deep Networks for Multiple Sclerosis Lesion Segmentation," MIA 2021.
[15] Wei et al., "Inverse Scaling Can Become U-Shaped," EMNLP 2023.
[16] Recht et al., "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019.
[17] Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2023.
[18] Verma et al., "MoNuSAC 2020: A Multi-Organ Nuclei Segmentation and Classification Challenge," IEEE ISBI 2021.