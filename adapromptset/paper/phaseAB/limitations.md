# Limitations

1. **No Confirmatory Cohort**: The primary results are based on a development cohort of 
   80 queries. No fresh, independent confirmatory cohort was available at the time of 
   writing. As such, this paper should be considered an exploratory/benchmark study. 
   Power analysis suggests that a minimum of 75 queries with at least 10 catastrophic 
   events is needed for confirmatory evaluation.

2. **Pipeline Specificity**: The method is strongest for SAM-B pipelines and shows 
   weaker effects for SAM2-based pipelines. Claims of broad effectiveness across all 
   foundation models are not supported.

3. **Single Task Domain**: The primary evaluation is on nuclei segmentation. Cross-dataset 
   evaluation on IDRID (retinal lesions) shows boundary behavior. Generalization to other 
   medical imaging tasks or natural images requires further study.

4. **Reference Quality Dependence**: The method requires a support set of reference 
   image-mask pairs. If the reference pool is of poor quality or insufficiently diverse, 
   the risk signal may be unreliable.

5. **Computational Overhead of Multi-Reference Inference**: While prediction consistency 
   itself is cheap, the multi-reference inference requires $m$ forward passes through 
   the segmenter. For $m=6$, this is 6x the cost of single-reference inference. Whether 
   this cost is justified depends on the application's tolerance for failures.

6. **No Absolute Quality Guarantee**: Prediction consistency estimates relative risk 
   within a batch of queries. It does not provide an absolute quality estimate. A query 
   with high consistency may still have low Dice if all references produce similarly 
   poor segmentations.
