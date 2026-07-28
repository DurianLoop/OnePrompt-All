# Introduction

Visual prompting has emerged as a powerful paradigm for adapting foundation segmentation 
models to specific tasks without fine-tuning. A typical multi-reference visual prompting 
pipeline uses a visual matcher (e.g., DINOv2) to retrieve similar reference image-mask pairs 
from a support set, then prompts a segmenter (e.g., SAM) with the reference mask locations 
to segment the query image. This approach has been applied to medical image segmentation, 
natural image segmentation, and beyond.

However, a critical reliability issue arises in multi-reference prompting: different reference 
images can produce substantially different segmentations for the same query. Some references 
lead to accurate masks while others produce catastrophic failures. Currently, there is no 
principled way to determine, at inference time, whether a given query's segmentation is 
trustworthy—without access to query ground truth.

## The Reference Reliability Problem

We formalize this as *Reference Reliability Estimation* (RRE): given a query image and a set 
of reference image-mask pairs, predict the segmentation quality (e.g., Dice score) for the 
query *without* access to the query's ground truth mask. This is a selective prediction 
problem: we want to identify high-risk queries and either flag them for human review or 
abstain from providing a prediction.

## Key Insight

Our key insight is simple: when multiple references produce consistent segmentations for a 
query, the result is likely reliable; when they disagree substantially, the result is likely 
unreliable. This intuition leads to *Prediction Consistency*—a risk score computed as the 
mean pairwise IoU among the segmentations produced by different references. Because these 
segmentations are already generated during standard multi-reference inference, prediction 
consistency incurs no additional model calls and requires no query ground truth.

## Findings

On our primary pipeline (DINOv2-L + SAM-B) with 80 MoNuSAC nuclei segmentation queries, 
prediction consistency achieves an AURC of 0.0158 and reduces the catastrophic failure 
rate (Dice < 0.1) by 88.6% at 80% coverage (CFR@80 = 0.016 vs. baseline 0.154). This 
substantially outperforms entropy-based ranking (AURC 0.128, CFR reduction -3.8%) and 
other baselines.

Cross-pipeline evaluation across five pipelines (DINOv2-L/B + SAM-B/L, SAM2-Tiny/Small) 
shows directional support in all five pipelines, with the strongest effects in SAM-B-based 
pipelines (CFR reduction 88.6% for both DINOv2-L and DINOv2-B). Cross-dataset evaluation 
on MoNuSeg reveals that the risk signal transfers (rho=0.83 between prediction consistency 
and per-reference Dice variance), but the segmentation quality itself is dataset-dependent.

## Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work in 
visual prompting, selective prediction, and uncertainty estimation. Section 3 formalizes 
the RRE problem and evaluation protocol. Section 4 presents our prediction consistency 
method. Section 5 describes the experimental setup and main results. Section 6 provides 
failure analysis. Section 7 discusses limitations, and Section 8 concludes.
