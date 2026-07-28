# WP2b — External Comparator Implementation Notes

> Audit date: 2026-07-23
> Phase T directive §7 (WP2b — Related Method Comparison)
> Scope: research/documentation only. No code was downloaded or run.

This document records, for each of the seven priority external methods listed in
the Phase T directive §7.1: what the method does, the key differences from our
reference-resampling approach, what would be needed to run it under our protocol,
and an estimated adaptation effort.

Our protocol (for comparison): frozen DINOv2-L matcher produces a matching map
between a query image and one or more reference images (with reference masks);
the matching map is converted into point prompts for a frozen SAM-B segmenter;
aggregation strategies (random single / mean consensus / etc.) combine outputs
across reference subsets of size k ∈ {1, 3, 6}. The benchmark measures
reference-induced instability (V_ref, best-worst gap, CVaR, CFR) rather than
single-reference accuracy.

---

## 1. GF-SAM — Bridge the Points: Graph-based Few-shot Segment Anything Semantically

- **Venue / year:** NeurIPS 2024 (Spotlight)
- **arXiv:** 2410.06964
- **Code:** https://github.com/ANDYZAQ/GF-SAM
- **Authors:** Anqi Zhang, Guangyu Gao, Jianbo Jiao, Chi Harold Liu, Yunchao Wei
  (Beijing Institute of Technology / University of Birmingham / Beijing Jiaotong University)
- **Tier:** 2 (official code exists; interface incompatible with our protocol)

### What it does
GF-SAM extends SAM to few-shot semantic segmentation (FSS) by automatically
generating point prompts from support (reference) images. It has three modules:

1. **Positive-Negative Alignment (PNA):** computes correlation between target
   features and reference features to dynamically select point prompts,
   including background-context negative points.
2. **Point-Mask Clustering (PMC):** builds a directed graph linking points to
   the masks SAM produces for them (by mask coverage), then decomposes weakly
   connected components to form point clusters.
3. **Post-Gating:** positive gating aggregates high-confidence masks;
   overshooting gating filters false-positive masks.

Reports mIoU 58.7% on COCO-20i and 35.2% on LVIS-92i.

### Key differences from our approach
- GF-SAM's prompt construction uses its own feature correlation (SAM/DINOv2
  features) and a graph-clustering step, whereas we use a single DINOv2-L
  matching map reduced to a fixed point config (P20-N1 / P30-N0 / P50-N0).
- GF-SAM is designed for natural-image FSS (Pascal-5i, COCO-20i, LVIS-92i,
  FSS-1000) and one-shot part segmentation, not medical MoNuSAC/GlaS.
- GF-SAM aggregates multiple SAM masks via graph clustering, which is a
  different aggregation strategy than our mean/median consensus over reference
  subsets.
- GF-SAM does not study reference-induced instability or reference resampling.

### What would be needed to run it
- Clone the official repository and install its environment.
- Adapt the data loader to MoNuSAC/GlaS reference-query pairs and our
  reference-resampling subsets (k ∈ {1, 3, 6}, fixed seeds).
- Ensure the SAM checkpoint and feature extractor versions match those used
  by GF-SAM (it uses SAM-ViT-H features and DINOv2).
- Wire GF-SAM's prompt-construction output into our evaluation harness so
  that, for each reference subset, we obtain a mask and can compute V_ref,
  best-worst gap, etc.
- User authorization to download external code (Phase T directive §23.1).

### Estimated effort
Medium-high. The code is public and the method is training-free (no learned
weights beyond frozen SAM/DINOv2), but adapting the FSS data pipeline to our
medical reference-resampling protocol and re-running GF-SAM's graph-clustering
per reference subset requires non-trivial integration work. Rough estimate:
2–4 engineering days plus compute.

---

## 2. SynPo — Boosting Training-Free Few-Shot Medical Segmentation via High-Quality Negative Prompts

- **Venue / year:** MICCAI 2025 (Early Accept)
- **arXiv:** 2506.15153
- **Project page:** https://liu-yufei.github.io/synpo-project-page/
- **Authors:** Yufei Liu, Haoke Xiao, Jiaxing Chai, Yongcun Zhang, Rong Wang,
  Zijie Meng, Zhiming Luo (Xiamen University / Peking University / Kuaishou)
- **Tier:** 3 (no public code repository confirmed at audit time)

### What it does
SynPo is a training-free few-shot medical segmentation method. Its core insight
is improving the quality of *negative* prompts. It has three modules:

1. **Confidence Map Synergy Module (CMSM):** combines DINOv2 (strong semantics,
   weak absolute localization) and SAM-ViT (absolute position encoding, better
   localization) features into a synergy map and a negative-confidence
   distribution.
2. **Point Selection Module (PSM):** selects top-k pixels as positive points
   and negative points via a Gaussian distribution over the confidence map,
   followed by independent K-means clustering for both sets.
3. **Noise-aware Refine Module (NRM):** refines the coarse SAM output.

Reports results on Synapse-CT and CHAOS-MRI; claims performance comparable to
training-based few-shot methods.

### Key differences from our approach
- SynPo uses the same backbone stack we do (DINOv2 + SAM), making it the most
  architecturally aligned comparator. However, it focuses on *prompt quality*
  (especially negative prompts) rather than on *reference choice* instability.
- SynPo's prompt construction is more elaborate (synergy map, Gaussian negative
  selection, K-means) than our fixed point-config reduction of the DINOv2
  matching map.
- SynPo does not study reference resampling or reference-induced instability;
  it treats the support (reference) as a fixed input.

### What would be needed to run it
- A public code repository. At audit time the project page links only to the
  arXiv paper; no GitHub URL is published. We cannot confirm code availability.
- If code is released later, adaptation would be comparatively low-effort
  because the stack matches ours: we would swap our prompt-construction step
  for SynPo's CMSM+PSM and run it per reference subset.
- Without code, a faithful reimplementation from the paper is possible but
  risks subtle deviations (e.g., the exact synergy-map formula, K-means
  parameters) and would not be a true reproduction.

### Estimated effort
- With official code: Low-medium (1–2 days; stack is compatible).
- Without code (reimplementation): High and not recommended for a
  "reproduction" claim; would be a concept comparison only.

---

## 3. ProSAM — Enhancing the Robustness of SAM-based Visual Reference Segmentation with Probabilistic Prompts

- **Venue / year:** arXiv preprint (June 2025), Bosch Research North America
- **arXiv:** 2506.21835
- **Code:** not available (no public code or weights found)
- **Authors:** Xiaoqi Wang, Clint Sebastian, Wenbin He, Liu Ren
  (Bosch Research / Bosch Center for AI)
- **Tier:** 3 (no public code or weights)

### What it does
ProSAM is the most conceptually relevant comparator: it explicitly studies
*instability* in SAM-based visual reference segmentation. It observes that
existing training-based methods generate prompts near object boundaries (an
unstable region of SAM's prompt space), causing robustness problems. It
proposes a **variational prompt encoder** that, via the reparameterization
trick, predicts a multivariate prompt distribution; sampling prompts from this
distribution avoids unstable regions. At inference, the predicted mean is used.

Reports state-of-the-art on Pascal-5i and COCO-20i for visual reference
segmentation.

### Key differences from our approach
- ProSAM attacks *prompt-space instability* (where to place the prompt);
  we attack *reference-choice instability* (which reference image to use).
  These are complementary sources of instability.
- ProSAM is training-based (learns the variational prompt encoder), whereas
  our protocol freezes the matcher and segmenter.
- ProSAM operates on natural-image visual reference segmentation, not medical
  data.
- ProSAM's notion of "robustness" is about avoiding boundary prompts, not
  about measuring variance across reference resamples.

### What would be needed to run it
- Public code and pretrained weights for the variational prompt encoder.
  Neither is available. Bosch has not released code as of audit time.
- Without code/weights, reproduction is impossible: the variational encoder
  must be trained on Pascal-5i/COCO-20i with a specific training procedure
  that is not fully specified in the paper.
- A reimplementation would require training a new model, which violates our
  frozen-matcher/frozen-segmenter protocol and would not be a faithful
  comparison.

### Estimated effort
Not reproducible without code. Concept comparison only (Tier 3). If code were
released, effort would be high (training-based, natural-image domain, needs
adaptation to medical data and to our reference-resampling subsets).

---

## 4. MAUP — Training-free Multi-center Adaptive Uncertainty-aware Prompting for Cross-domain Few-shot Medical Image Segmentation

- **Venue / year:** MICCAI 2025
- **Code:** https://github.com/YazhouZhu19/MAUP
- **Authors:** Yazhou Zhu, Haofeng Zhang
  (Nanjing University of Science and Technology)
- **Tier:** 2 (official code exists; interface incompatible with our protocol)

### What it does
MAUP is a training-free framework adapting SAM for cross-domain few-shot
medical image segmentation. It has three components:

1. **Multi-center prompt generation:** uses K-means clustering on the
   confidence map to generate multiple candidate prompt "centers".
2. **Uncertainty-aware prompt selection:** estimates uncertainty and selects
   prompts that balance confidence and spatial diversity.
3. **Dynamic optimization:** iteratively refines the selected prompts.

Reports Dice improvements of ~12.3% over baselines on abdominal CT and other
medical datasets, with ~30× computational efficiency gain.

### Key differences from our approach
- MAUP is cross-domain FSS (source domain → target domain transfer), whereas
  our reference-resampling is within-dataset (query and references from the
  same dataset, non-overlapping cases).
- MAUP's "multi-center" prompt generation is a per-query prompt-construction
  strategy; our reference-resampling varies the *reference image* while
  keeping the prompt-construction rule fixed.
- MAUP is training-free and uses frozen SAM, which aligns with our frozen-
  segmenter constraint.
- MAUP does not study reference-induced instability.

### What would be needed to run it
- Clone the official repository and install its environment.
- Adapt the data loader: MAUP expects source/target domain splits; we need to
  map our reference-resampling subsets (k ∈ {1, 3, 6}) into MAUP's multi-center
  prompt generation. Concretely, for each reference subset, feed the reference
  image(s) and mask(s) into MAUP's confidence-map + K-means pipeline and obtain
  a mask for the query.
- Wire MAUP's output into our evaluation harness to compute V_ref, best-worst
  gap, CVaR, CFR per reference subset.
- User authorization to download external code (Phase T directive §23.1).

### Estimated effort
Medium. The code is public and the method is training-free (medical domain,
frozen SAM), so the stack is compatible. The main work is mapping the
cross-domain FSS interface to our within-dataset reference-resampling protocol.
Rough estimate: 2–3 engineering days plus compute.

---

## 5. SAMIX — Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy for Mix-Supervised Segmentation

- **Venue / year:** CVPR 2026
- **Code:** https://github.com/Huster-Hq/SAMIX (referenced in paper)
- **Authors:** Qiang Hu, Jiajie Wei, Zhenyu Yi, Zhifen Yan, Yingjie Guo,
  Hongkuan Shi, Ge-Peng Ji, Qiang Li, Zhiwei Wang
  (Huazhong University of Science and Technology / Shanghai Jiao Tong
  University / Wuhan United Imaging / Australian National University)
- **Tier:** 4 (method not applicable to current task)

### What it does
SAMIX adapts SAM2 into a semantic-aware pseudo-label generator (SA-SAM2) via a
lightweight semantic adapter, and trains a **Selecting Policy Network (SPNet)**
via reinforcement learning to auto-regressively retrieve relevant and
complementary reference samples for each query image. The reference samples
serve as dense contextual prompts. It is designed for mix-supervised
segmentation (combining fully-labeled, weakly-labeled, and unlabeled data).

Datasets: PASCAL VOC 2012, Cityscapes, camouflaged object detection, polyp
segmentation.

### Key differences from our approach
- SAMIX's reference selection is a *learned, training-time* policy (RL-trained
  SPNet), whereas our protocol freezes the matcher and studies *test-time*
  reference aggregation. The philosophies are opposite: SAMIX learns to pick
  references; we measure how much the choice matters.
- SAMIX uses SAM2 (not SAM-B) and adapts it with a semantic adapter, violating
  our frozen-segmenter constraint.
- SAMIX's "reference" is a dense contextual prompt (image-mask pair), not a
  point-prompt source. It does not construct point prompts for SAM.
- SAMIX targets pseudo-label generation for mix-supervised training, not a
  reference-instability benchmark.

### What would be needed to run it
- Training SPNet via RL with customized rewards, which requires a training
  pipeline and labeled data — fundamentally incompatible with our frozen-model
  test-time protocol.
- SAM2 + semantic adapter weights.
- Even if run, the comparison would be apples-to-oranges: SAMIX optimizes
  reference selection (a learned policy) while we benchmark reference
  instability under fixed aggregation rules.

### Estimated effort
Not applicable (Tier 4). Running SAMIX would require abandoning the frozen-
matcher/frozen-segmenter protocol and training an RL policy, which is out of
scope for the reference-instability benchmark. Concept comparison only,
documented in the novelty audit (§17) to distinguish our work from learned
reference selection.

---

## 6. AlignSAM — Aligning Segment Anything Model to Open Context via Reinforcement Learning

- **Venue / year:** CVPR 2024
- **arXiv:** 2406.00480
- **Code:** https://github.com/Duojun-Huang/AlignSAM-CVPR2024
- **Authors:** Duojun Huang, Xinyu Xiong, Jie Ma, Jichang Li, Zequn Jie,
  Lin Ma, Guanbin Li (Sun Yat-sen University / Meituan)
- **Tier:** 4 (method not applicable to current task)

### What it does
AlignSAM formulates automatic prompting as a sequential decision-making
process and trains an RL agent (actor + critic) to iteratively recommend
prompting positions to a frozen SAM, imitating human annotators. A semantic
recalibration module provides fine-grained prompt labels. It targets diverse
open-context segmentation tasks (saliency, shadow, glass-like objects).

### Key differences from our approach
- AlignSAM is **not reference-guided**. It does not use reference images or
  reference masks; it learns an RL policy to place prompts from the query
  image alone.
- It requires a trained RL actor/critic policy, which violates our frozen-
  matcher constraint (we use a frozen DINOv2 matcher, not a learned prompting
  agent).
- It does not study reference-induced instability.

### What would be needed to run it
- Trained RL policy weights (actor + critic).
- The method has no notion of "reference image", so it cannot be plugged into
  a reference-resampling protocol without fundamental redesign.
- Even if we used its RL prompter as a prompt-construction baseline, it would
  not answer the reference-instability question (it ignores references).

### Estimated effort
Not applicable (Tier 4). Concept comparison only, relevant to the novelty
audit's distinction between "RL iterative prompting" and our reference-
resampling benchmark.

---

## 7. Point Prompt Defender — Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model

- **Venue / year:** CVPR 2026
- **arXiv:** 2509.18891
- **Code:** https://github.com/L-AILab/PPD
- **Authors:** Xueyu Liu, Xiaoyi Zhang, Guangze Shi, Meilin Liu, Yexin Lai,
  Yongfei Wu, Mingqiang Wei (Taiyuan University of Technology / Nanjing
  University of Aeronautics and Astronautics)
- **Tier:** 4 (method not applicable to current task)

### What it does
Point Prompt Defender is an adversarial RL framework for point-prompt
optimization. It builds a task-agnostic point-prompt environment (dual-space
graph with physical and semantic distances) and trains two Deep Q-Network
agents:

1. **Attacker:** selects a subset of prompts that maximally degrade SAM
   segmentation.
2. **Defender:** suppresses these disruptive prompts and restores accuracy.

At inference, only the defender is deployed to refine arbitrary coarse prompt
sets. It is plug-and-play and task-agnostic.

### Key differences from our approach
- Point Prompt Defender is **not reference-guided**. It optimizes point-prompt
  *sets* via trained adversarial agents, without using reference images.
- It requires trained attacker/defender DQN networks, violating our frozen-
  model constraint.
- Its notion of "robustness" is against adversarial prompt perturbations, not
  against reference-image choice. These are different instability axes.
- It does not study reference resampling.

### What would be needed to run it
- Trained attacker/defender DQN weights.
- A prompt set to refine. Since our protocol produces point prompts from
  DINOv2 matching maps, we could in principle feed those prompts to the
  defender. But the defender is a learned model trained on a different
  objective (adversarial robustness), and using it would change our
  segmenter pipeline from "frozen SAM" to "frozen SAM + trained defender",
  breaking protocol comparability.
- The method cannot answer the reference-instability question because it has
  no notion of reference images.

### Estimated effort
Not applicable (Tier 4). Concept comparison only, relevant to the novelty
audit's distinction between "point prompt filtering/defense" and our
reference-resampling instability benchmark.

---

## Summary table

| Method | Tier | Reference-guided | Code available | Reproduction feasible under our protocol? |
|---|---|---|---|---|
| GF-SAM | 2 | Yes | Yes | Yes, with adaptation (natural-image FSS → medical reference-resampling) |
| SynPo | 3 | Yes | No (project page only) | No, until code is released |
| ProSAM | 3 | Yes | No | No, no code/weights |
| MAUP | 2 | Yes | Yes | Yes, with adaptation (cross-domain FSS → within-dataset reference-resampling) |
| SAMIX | 4 | Yes (reference selection) | Yes (referenced) | No, requires RL training + SAM2 adapter; wrong paradigm |
| AlignSAM | 4 | No | Yes | No, not reference-guided |
| Point Prompt Defender | 4 | No | Yes | No, not reference-guided; requires trained DQN agents |

### Implication for §7.3 minimum external comparison requirement

- **Option A (reproduce 2 external methods):** Feasible with GF-SAM (Tier 2)
  and MAUP (Tier 2). Both have public code and are reference-guided; both
  need adaptation to our reference-resampling protocol. This is the
  recommended path.
- **Option B (1 external + 3 strong internal families):** Also feasible as a
  fallback using MAUP or GF-SAM plus our internal B1–B7 families.
- **Option C (no reproduction, protocol only):** Not needed if A is pursued,
  but remains the fallback if adaptation proves blocked.
