# WP2b — Unsupported / Non-Reproducible External Comparators

> Audit date: 2026-07-23
> Phase T directive §7 (WP2b — Related Method Comparison)
> Scope: research/documentation only. No code was downloaded or run.

This document lists the external methods that **cannot be reproduced** under
our reference-resampling protocol and explains why. Per the directive §7.2,
Tier 3 methods (no code/weights) are concept-comparison only and must not
have fabricated numbers; Tier 4 methods (not applicable) are documented for
the novelty audit (§17) but are not run.

---

## Tier 3 — No available code or weights (concept comparison only)

### SynPo

- **Paper:** SynPo: Boosting Training-Free Few-Shot Medical Segmentation via
  High-Quality Negative Prompts (MICCAI 2025, Early Accept)
- **arXiv:** 2506.15153
- **Project page:** https://liu-yufei.github.io/synpo-project-page/
- **Why unsupported:** The project page links only to the arXiv paper. No
  public GitHub repository could be confirmed at audit time (2026-07-23).
  Web searches for "SynPo github code liu-yufei" returned the project page
  and paper summaries but no code URL.
- **Reproduction feasibility:** If the authors release code, SynPo would be
  the most natural comparator (same DINOv2 + SAM stack, same few-shot medical
  segmentation task). Until then, a reimplementation from the paper risks
  subtle deviations (exact synergy-map formula, K-means parameters,
  Noise-aware Refine Module details) and would not qualify as a faithful
  reproduction.
- **Disposition:** Concept comparison only. Documented in the novelty audit
  (§17) under "positive/negative point construction". No fabricated results.
  Re-audit if code is released.

### ProSAM

- **Paper:** ProSAM: Enhancing the Robustness of SAM-based Visual Reference
  Segmentation with Probabilistic Prompts (arXiv, June 2025)
- **arXiv:** 2506.21835
- **Authors:** Xiaoqi Wang, Clint Sebastian, Wenbin He, Liu Ren
  (Bosch Research North America / Bosch Center for AI)
- **Why unsupported:** No public code or pretrained weights. The paper does
  not publish a code URL. Web searches for "ProSAM Bosch visual reference
  segmentation code github" returned only the arXiv paper and secondary
  summaries; no GitHub repository exists.
- **Reproduction feasibility:** ProSAM is training-based (it learns a
  variational prompt encoder via reparameterization). Reproduction would
  require (a) the full training code, (b) the training procedure details
  (not fully specified in the paper), and (c) pretrained weights or the
  training data (Pascal-5i, COCO-20i) to retrain. None of these are
  available. A from-scratch reimplementation would be a research project in
  itself, not a reproduction, and would violate our frozen-matcher
  constraint.
- **Disposition:** Concept comparison only. ProSAM is the most conceptually
  relevant comparator (it explicitly studies instability in visual reference
  segmentation), so it is discussed at length in the novelty audit (§17) and
  the mock review (§18). No fabricated results.

---

## Tier 4 — Method not applicable to current task

### SAMIX

- **Paper:** SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference
  Selecting Policy for Mix-Supervised Segmentation (CVPR 2026)
- **Code:** https://github.com/Huster-Hq/SAMIX (referenced in paper)
- **Why not applicable:** SAMIX is a *mix-supervised pseudo-label generation*
  method. Its reference selection is a *learned, training-time* policy (an
  RL-trained Selecting Policy Network, SPNet) that retrieves reference samples
  to maximize pseudo-label quality. This is the opposite of our protocol,
  which freezes the matcher and segmenter and studies *test-time* reference
  aggregation under fixed rules. Running SAMIX would require:
  1. Training SPNet via reinforcement learning with custom rewards.
  2. Adapting SAM2 with a semantic adapter (SA-SAM2), violating the
     frozen-segmenter constraint and switching from SAM-B to SAM2.
  3. A mix-supervised data setup (fully-labeled + weakly-labeled + unlabeled),
     which is not our benchmark setting.
- **Conceptual relevance:** SAMIX is relevant to the novelty audit (§17) under
  "reference selection" — it is a learned reference-selection method, whereas
  we benchmark reference instability under fixed aggregation. The distinction
  is that SAMIX optimizes *which* references to use; we measure *how much the
  choice matters* and mitigate with simple consensus.
- **Disposition:** Not run. Documented in the novelty audit as a learned
  reference-selection contrast. No fabricated results.

### AlignSAM

- **Paper:** AlignSAM: Aligning Segment Anything Model to Open Context via
  Reinforcement Learning (CVPR 2024)
- **arXiv:** 2406.00480
- **Code:** https://github.com/Duojun-Huang/AlignSAM-CVPR2024
- **Why not applicable:** AlignSAM is **not reference-guided**. It trains an
  RL agent (actor + critic) to iteratively recommend prompt positions to a
  frozen SAM, imitating human annotators. It does not consume reference
  images or reference masks; it places prompts from the query image alone.
  Therefore it cannot be plugged into a reference-resampling protocol, which
  requires varying the reference image as the input. Additionally, it requires
  a trained RL policy, violating our frozen-matcher constraint.
- **Conceptual relevance:** Relevant to the novelty audit (§17) under "RL
  iterative prompting" — AlignSAM learns *where* to prompt; we study *which
  reference* to use and how much that choice destabilizes the output.
- **Disposition:** Not run. Documented in the novelty audit as an RL iterative
  prompting contrast. No fabricated results.

### Point Prompt Defender

- **Paper:** Attack for Defense: Adversarial Agents for Point Prompt
  Optimization Empowering Segment Anything Model (CVPR 2026)
- **arXiv:** 2509.18891
- **Code:** https://github.com/L-AILab/PPD
- **Why not applicable:** Point Prompt Defender is **not reference-guided**.
  It trains two Deep Q-Network agents (attacker and defender) that operate on
  a dual-space graph of image patches to refine point-prompt sets. It has no
  notion of reference images. Its robustness objective is against adversarial
  prompt perturbations, not against reference-image choice. Running it would
  require trained DQN weights and would change the segmenter pipeline from
  "frozen SAM" to "frozen SAM + trained defender", breaking protocol
  comparability.
- **Conceptual relevance:** Relevant to the novelty audit (§17) under "point
  filtering" — Point Prompt Defender defends against adversarial point
  perturbations; we measure reference-induced instability and mitigate with
  consensus. These are orthogonal robustness axes.
- **Disposition:** Not run. Documented in the novelty audit as a point-filtering
  contrast. No fabricated results.

---

## Summary

| Method | Tier | Reason not reproduced |
|---|---|---|
| SynPo | 3 | No public code repository confirmed; reimplementation would not be faithful |
| ProSAM | 3 | No public code or weights; training-based; cannot reproduce |
| SAMIX | 4 | Requires RL training + SAM2 adapter; mix-supervised paradigm; not applicable |
| AlignSAM | 4 | Not reference-guided; RL prompter without references; not applicable |
| Point Prompt Defender | 4 | Not reference-guided; adversarial point optimization; not applicable |

### Impact on the §7.3 minimum external comparison requirement

Of the seven priority methods, **two are reproducible** (GF-SAM, MAUP — both
Tier 2) and **five are not** (SynPo and ProSAM are Tier 3; SAMIX, AlignSAM,
and Point Prompt Defender are Tier 4).

This satisfies **Option A** of §7.3 (reproduce 2 external methods) via GF-SAM
and MAUP, provided their code can be downloaded and adapted (requires user
authorization per §23.1). If adaptation is blocked, **Option B** (1 external
method + 3 strong internal method families) is available as a fallback using
either GF-SAM or MAUP plus our internal B1–B7 families. **Option C** (no
reproduction, protocol + interface + limitations only) is the last-resort
fallback and would lower the submission positioning per §7.3.

All five unsupported methods are documented in the novelty audit (§17) and
mock review (§18) as concept comparisons. No fabricated numerical results are
included for any unsupported method.
