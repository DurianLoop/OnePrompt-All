# Failure Analysis

## When Prediction Consistency Fails

Prediction consistency as a risk signal is not universally effective. We identify the 
following failure modes:

### 1. Low Base Segmentation Quality

When the base segmenter produces consistently poor segmentations (e.g., IDRID with 
Dice=0.098), the pairwise IoU among references may be consistently low, reducing the 
dynamic range of the risk signal. In this regime, prediction consistency can still rank 
risk (rho=0.952 on IDRID) but cannot identify which queries are "good enough" because 
none are.

### 2. Absence of Catastrophic Failures

When the base segmenter produces no catastrophic failures (e.g., MoNuSeg with CFR=0.0), 
selective prediction provides no benefit regardless of risk ranking quality. The risk 
signal is valid but unnecessary.

### 3. Architecturally-Limited Pipelines

SAM2-Small produces low-quality segmentations on nuclei (Dice=0.298), and the prediction 
consistency signal is weaker (CFR reduction -1.6%). This suggests that the method requires 
a minimum level of base segmentation quality to be effective.

### 4. Task Mismatch

IDRID (retinal lesions) represents a different task domain from MoNuSAC (nuclei). While 
the risk signal transfers (rho=0.952), the segmentation quality does not. This highlights 
that prediction consistency estimates *relative* risk within a dataset but does not 
guarantee *absolute* segmentation quality.

## When Prediction Consistency Works

The method is most effective when:
1. The base segmenter produces reasonable-quality segmentations for most queries
2. There exists a meaningful fraction of high-risk queries to identify
3. The references provide sufficient variation to reveal inconsistency
4. The task domain is within the segmenter's capability
