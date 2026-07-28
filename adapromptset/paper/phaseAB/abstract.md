# Abstract

Multi-reference visual prompting, where a segmentation model conditions on a query image and 
multiple reference image-mask pairs, has emerged as a flexible paradigm for few-shot segmentation. 
However, different references can produce substantially different predictions for the same query, 
creating a reliability problem: when should we trust the model's output? We formalize this as 
*Reference Reliability Estimation* (RRE)—the task of predicting query-level segmentation quality 
without access to query ground truth—and introduce a reference-resampling evaluation protocol 
that isolates the effect of reference choice.

We propose *Prediction Consistency*, a simple, ground-truth-free risk score defined as the mean 
pairwise IoU among segmentations produced by different references. Queries with low prediction 
consistency (high disagreement) are flagged as high-risk and can be selectively abstained, 
enabling coverage-based reliability control. Across five pipelines spanning DINOv2 and SAM/SAM2 
backbones on 80 nuclei segmentation queries, prediction consistency achieves an AURC of 0.0158 
with 88.6% relative reduction in catastrophic failure rate at 80% coverage, substantially 
outperforming entropy-based (AURC 0.1281), candidate Jaccard, and prompt overlap baselines. 

Cross-dataset evaluation on MoNuSeg and IDRID reveals boundary conditions: the risk signal 
transfers to datasets where segmentation quality is moderate (rho=0.83 on MoNuSeg), but 
segmentation quality itself may not improve. We conduct systematic ablation of reference count, 
consistency metrics, and pipeline components, and provide an honest analysis of failure modes 
and limitations. This work establishes prediction consistency as a practical, training-free 
approach to reliability estimation in multi-reference visual prompting, and provides a benchmark 
for future research in this direction.

*Note: This is an exploratory/benchmark paper. No confirmatory holdout cohort was available at 
the time of writing.*
