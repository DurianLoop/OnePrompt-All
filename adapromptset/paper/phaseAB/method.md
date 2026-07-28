# Method: Prediction Consistency for Selective Prediction

## Overview

Our method is remarkably simple and requires no additional training, no model modification, 
and no extra forward passes. The core idea is:

1. **Generate**: For each query, produce $m$ segmentations using $m$ different references
2. **Compare**: Compute the pairwise IoU among all $m$ segmentations
3. **Score**: The risk score $R_{\text{PC}}(q) = 1 - \text{mean}(\text{IoU}_{i,j})$
4. **Abstain**: Flag the highest-risk queries for abstention based on a target coverage

## Implementation Details

The prediction consistency risk score is computed as follows:

- **Reference count**: $m = 6$ references per query, randomly sampled from the support set
- **Patient/query disjointness**: References are drawn from different patients/cases than the query
- **Pairwise metric**: Standard IoU (Intersection over Union) at original resolution
- **Empty mask handling**: Empty vs. empty = 0.0, empty vs. non-empty = 0.0
- **Binary threshold**: 0.5 for binarizing soft masks
- **No postprocessing**: Raw segmentations used directly

## Primary Pipeline

The primary pipeline uses DINOv2-L as the visual matcher and SAM-B (ViT-B) as the segmenter. 
This combination was chosen because it represents the most common configuration and showed 
the strongest prediction consistency signal in preliminary experiments.

## Baselines

We compare against five baselines:

1. **Random Ranking** (B0): Random risk scores—expected to perform at chance level
2. **Matching Entropy** (B1): Entropy of the DINOv2 matching similarity distribution across 
   references—higher entropy suggests ambiguous matching
3. **Candidate Jaccard** (B2): Mean pairwise Jaccard index of candidate regions (before 
   segmentation)—measures pre-segmentation agreement
4. **Prompt Overlap** (B3): Overlap of prompt point locations across references—measures 
   spatial agreement of prompts
5. **M2-Instability** (B5): Coefficient of variation of per-reference Dice scores (requires 
   query GT, used as supervised upper bound)

## Computational Cost

Prediction consistency adds negligible computational overhead. The pairwise IoU computation 
requires only the $m$ already-generated segmentation masks. For $m=6$, this is $\binom{6}{2} = 15$ 
IoU computations per query, each taking milliseconds. No additional model calls, feature 
extractions, or training steps are required.
