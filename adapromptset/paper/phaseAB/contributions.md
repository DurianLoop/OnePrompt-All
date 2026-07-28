# Contributions

1. **RRE Problem Formulation**: We formalize Reference Reliability Estimation (RRE) as the 
   task of predicting query-level segmentation quality without query ground truth, and 
   introduce a reference-resampling evaluation protocol that isolates the effect of reference 
   choice from other sources of variation.

2. **Simple Risk Signal**: We propose Prediction Consistency—the mean pairwise IoU among 
   segmentations from different references—as a ground-truth-free, no-additional-training 
   risk signal for query-level reliability estimation. The risk score R_PC(q) = 1 - C_pred(q) 
   requires only the segmentations already produced during multi-reference inference.

3. **Systematic Evaluation**: We conduct a comprehensive evaluation across five pipelines 
   (DINOv2-L/B + SAM-B/L, DINOv2-L + SAM2-Tiny/Small), three datasets (MoNuSAC, MoNuSeg, 
   IDRID), and six baseline risk signals, providing an honest analysis of cross-pipeline 
   behavior, cross-dataset boundaries, and failure modes.
