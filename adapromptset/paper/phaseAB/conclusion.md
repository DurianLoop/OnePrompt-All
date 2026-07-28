# Conclusion

We have presented a simple yet effective approach to reliability estimation in multi-reference 
visual prompting. Our key contribution is the observation that prediction consistency—the 
agreement among segmentations produced by different references—serves as a practical, 
ground-truth-free risk signal for query-level quality estimation.

Through systematic evaluation across five pipelines and three datasets, we demonstrate that 
prediction consistency enables effective selective prediction, reducing catastrophic failure 
rates by 88.6% at 80% coverage on the primary pipeline. We provide an honest analysis of 
boundary conditions, including datasets where the risk signal transfers but segmentation 
quality does not improve, and pipelines where the effect is weaker.

This work establishes a foundation for reliability-aware multi-reference visual prompting. 
Future directions include: (1) confirmatory evaluation on a fresh independent cohort, 
(2) extension to natural image and video domains, (3) integration with model-level 
uncertainty estimation, and (4) development of adaptive reference selection strategies 
that optimize the trade-off between inference cost and reliability.
