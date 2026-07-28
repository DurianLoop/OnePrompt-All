# Appendix

## A. RRN Results

The Reference Reliability Network (RRN), a learned risk estimator trained on the 
development cohort, achieves an AUROC of 0.573 for catastrophic failure prediction. 
This is substantially lower than the 0.70 threshold required for practical utility 
and is included here for completeness.

**Recommendation**: RRN is not recommended as a primary method. Its performance 
does not justify the additional complexity of training a separate model.

## B. Anti-Correlation as Mechanical Artefact

Earlier phases of this work investigated the anti-correlation between prediction 
instability and catastrophic failure rate. After correction of a null model 
implementation error, the corrected $\Delta r$ values are +0.071 (L+SAM-B) and 
+0.034 (B+SAM-B), indicating that the anti-correlation is a mechanical artefact 
rather than a structural property of the segmentation pipeline. This finding is 
included for transparency and to prevent future misinterpretation.

## C. Full RAMP Component Ablation

The Full RAMP (Reference-Aware Mask Prediction) framework—which included learned 
reference weighting, adaptive prompt fusion, and consensus-based mask aggregation—was 
evaluated in earlier phases. Component ablation showed that no RAMP component provided 
meaningful improvement over the simple consensus baseline ($\Delta$Dice = 0.0). 
Full RAMP has been removed from the main method.

## D. Power Analysis

A bootstrap-based power analysis using the development cohort's effect distribution 
recommends a minimum sample size of N=75 queries for confirmatory evaluation, with 
at least 10 expected catastrophic events. The current development cohort of 80 queries 
meets this threshold, but a fresh, independent cohort is required for confirmatory claims.

## E. Reproducibility Checklist

- [x] Code available: D:/SAM/adapromptset/src/adapromptset/scripts/
- [x] Frozen config: configs/frozen/phaseAB_prediction_consistency.yaml
- [x] Primary pipeline: DINOv2-L + SAM-B
- [x] Reference count: 6
- [x] Reference seed: 42
- [x] Query-level statistics with bootstrap CIs
- [x] No query GT used in risk construction
- [x] All baseline methods documented
- [ ] Fresh confirmatory cohort (not available)
