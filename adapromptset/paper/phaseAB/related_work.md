# Related Work

## Visual Prompting for Segmentation

Visual prompting leverages foundation models like SAM (Segment Anything Model) for 
downstream segmentation tasks by providing visual cues such as points, boxes, or masks. 
In multi-reference settings, a visual matcher (e.g., DINOv2) retrieves similar reference 
image-mask pairs, and the segmenter is prompted with the reference mask locations. This 
paradigm has been adopted in medical imaging (e.g., PerSAM, MedSAM), natural image 
segmentation, and video object segmentation (SAM2).

However, existing work primarily focuses on improving segmentation accuracy through better 
matching, prompt engineering, or model adaptation. The reliability of the resulting 
segmentations—particularly the variability introduced by reference choice—has received 
limited attention.

## Selective Prediction and Risk Estimation

Selective prediction (also known as classification with rejection) allows models to abstain 
from predictions when confidence is low. In segmentation, uncertainty estimation methods 
include Monte Carlo dropout, ensemble disagreement, and test-time augmentation. Most 
existing approaches require either model modification (e.g., dropout layers), multiple 
forward passes, or access to training data for calibration.

Our work differs in two key aspects: (1) we focus on the *reference-induced* uncertainty 
specific to multi-reference prompting, rather than model epistemic uncertainty; and (2) 
our risk signal requires no model modification, no additional training, and no extra 
forward passes beyond those already performed.

## Uncertainty in Medical Image Segmentation

Uncertainty estimation in medical image segmentation has been studied extensively, with 
methods ranging from Bayesian neural networks to test-time augmentation and ensemble 
approaches. Most of these methods operate on single-input segmentation and estimate 
model-level uncertainty. Our work addresses a different source of uncertainty: the 
variability introduced by different conditioning references in a multi-reference 
prompting setup.

## Multi-Reference and Few-Shot Segmentation

Few-shot segmentation methods typically use a single or small number of support images. 
Recent work has explored using multiple supports through attention mechanisms, prototype 
learning, or feature fusion. Our reference-resampling protocol provides a systematic way 
to evaluate how reference choice affects segmentation quality, and our prediction 
consistency score leverages the multi-reference setup for reliability estimation.
