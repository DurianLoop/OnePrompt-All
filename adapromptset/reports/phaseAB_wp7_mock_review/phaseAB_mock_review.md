# Phase AB Mock Review

**Date**: 2026-07-28
**Paper**: "When References Disagree: Prediction Consistency for Selective Visual Prompting"
**Paper Type**: Exploratory/Benchmark

---

## Reviewer 1 (R1): "The Method Is Too Simple"

### Overall Rating: Weak Accept

### Summary
This paper addresses an important practical problem—reliability estimation in multi-reference 
visual prompting—and proposes a remarkably simple solution: use the pairwise IoU among 
segmentations from different references as a risk signal. The method is well-motivated, 
clearly described, and the experiments are thorough.

### Major Concerns

1. **The method is too simple to be a significant contribution.** Prediction consistency 
   is essentially just computing the variance of segmentations. What is the intellectual 
   novelty beyond "if references disagree, the result is unreliable"?

2. **The AUROC of 0.036 is alarmingly low.** Even though the authors argue that AURC is 
   the primary metric, an AUROC near random suggests the method cannot distinguish 
   catastrophic from non-catastrophic queries.

3. **No comparison to standard uncertainty estimation methods.** Why not compare against 
   Monte Carlo dropout, test-time augmentation, or ensemble-based uncertainty? These are 
   standard baselines in uncertainty estimation.

### Minor Concerns
- The paper would benefit from more qualitative examples showing high-consistency vs. 
  low-consistency queries.
- The cross-dataset analysis is limited to only 3 datasets.

---

## Reviewer 2 (R2): "No Real New Model"

### Overall Rating: Reject

### Summary
The paper proposes using prediction consistency—the mean pairwise IoU of segmentations 
from different references—as a risk signal. While the problem is relevant, the contribution 
is insufficient for a top-tier venue.

### Major Concerns

1. **No new model or learning algorithm.** The method is a heuristic. There is no training, 
   no optimization, no theoretical analysis. A heuristic alone does not constitute a 
   research contribution.

2. **The "method" is trivial.** Computing pairwise IoU among multiple segmentations is 
   an obvious baseline that any practitioner would try. The paper does not provide 
   non-obvious insights.

3. **The evaluation is on a single dataset (80 queries).** The cross-dataset results 
   show that the method fails on IDRID and is unnecessary on MoNuSeg. This is a very 
   narrow evaluation.

4. **The paper acknowledges it is exploratory/benchmark.** If the authors themselves 
   do not believe the results are confirmatory, why should reviewers?

---

## Reviewer 3 (R3): "Multi-Reference Inference Cost"

### Overall Rating: Weak Reject

### Summary
The paper addresses an interesting reliability problem in multi-reference visual prompting. 
The proposed solution is elegant in its simplicity, but I have concerns about practical 
value.

### Major Concerns

1. **The cost of multi-reference inference.** The method requires 6 forward passes through 
   SAM. At 6x the inference cost, is the reliability benefit worth it? The paper does not 
   compare against a single-reference baseline with a simpler uncertainty estimate.

2. **The primary metric (AURC) is not standard.** Most readers are more familiar with 
   AUROC and expected calibration error. The paper should justify why AURC is the 
   appropriate metric.

3. **The CFR reduction of 88.6% sounds impressive but the baseline CFR is only 15.4%.** 
   In absolute terms, this means going from ~2 catastrophic failures per 13 queries to 
   ~0.2 per 13 queries. Is this practically meaningful?

---

## Reviewer 4 (R4): "Cross-Dataset Instability"

### Overall Rating: Weak Accept

### Summary
The paper makes a clear contribution to the emerging area of reliability in visual prompting. 
The method is simple, well-motivated, and the evaluation is honest about limitations.

### Major Concerns

1. **Cross-dataset results are inconsistent.** MoNuSeg shows no benefit (CFR=0), IDRID 
   shows only marginal improvement (14.5% CFR reduction). The paper acknowledges this 
   but does not provide a satisfying explanation for why the method works on one dataset 
   and not others.

2. **The boundary analysis is post-hoc.** The paper categorizes MoNuSeg as "boundary" 
   and IDRID as "boundary" after seeing the results. A pre-registered boundary analysis 
   would be more convincing.

3. **The paper claims "directional support" in 5/5 pipelines but the actual AUROC values 
   are all below 0.5.** This suggests the method is not actually discriminating between 
   good and bad queries.

---

## Reviewer 5 (R5): "Selective Prediction Is Just Abstention"

### Overall Rating: Borderline

### Summary
The paper proposes a simple risk signal for selective prediction in multi-reference 
visual prompting. The writing is clear and the experiments are systematic.

### Major Concerns

1. **Selective prediction is not a new idea.** The paper applies existing selective 
   prediction concepts to visual prompting. What is the novel contribution beyond the 
   application domain?

2. **The paper does not compare against learned risk estimators.** There is no comparison 
   to a simple learned baseline (e.g., a small MLP trained on consistency features). 
   The RRN in the appendix (AUROC=0.573) actually outperforms prediction consistency 
   (AUROC=0.036) on the AUROC metric.

3. **The coverage-risk trade-off is not explored in depth.** The paper only reports 
   CFR@80. What about CFR@50, CFR@90, CFR@95? The full risk-coverage curve would be 
   more informative.

---

## Reviewer 6 (R6): "Limited Confirmatory Evidence"

### Overall Rating: Weak Accept

### Summary
This is a well-written paper on an important practical problem. The method is simple 
and the evaluation is honest about limitations.

### Major Concerns

1. **No confirmatory cohort.** The paper explicitly states this is an exploratory study. 
   For a venue that expects confirmatory evidence, this is a significant weakness.

2. **The development cohort of 80 queries is small.** Power analysis suggests 75 queries 
   as the minimum, meaning the current evaluation is barely above the threshold.

3. **The paper would benefit from natural image experiments.** Medical images are a 
   specific domain. Would the method work on COCO, Pascal, or other natural image 
   datasets?

---

## Reviewer 7 (R7): "Uncertainty Estimation Comparison"

### Overall Rating: Weak Accept

### Summary
The paper addresses an interesting problem with a simple solution. The evaluation is 
thorough within its scope.

### Major Concerns

1. **No comparison to standard uncertainty methods.** The paper should compare against 
   at least one standard uncertainty estimation method (e.g., MC dropout, ensemble, 
   or test-time augmentation).

2. **The relationship between prediction consistency and model uncertainty is not 
   discussed.** Are these complementary or redundant signals?

3. **The paper does not discuss calibration.** Even if the risk ranking is good, are 
   the risk scores calibrated? Can we interpret R_PC=0.3 as a 30% risk of failure?
