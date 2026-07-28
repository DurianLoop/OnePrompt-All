# WP2b — Protocol Alignment for Reproducible External Comparators

> Audit date: 2026-07-23
> Phase T directive §7 (WP2b — Related Method Comparison)
> Scope: research/documentation only. No code was downloaded or run.

This document specifies, for each reproducible (Tier 1 or Tier 2) external
method, how to align it with our reference-resampling protocol, what changes
are needed, and what metrics can be compared.

## Our reference-resampling protocol (recap)

- **Matcher:** frozen DINOv2-L; produces a 2-D matching map between query and
  each reference image (using the reference mask to define the target region).
- **Segmenter:** frozen SAM-B; invoked exactly once per (query, reference
  subset) episode.
- **Reference subsets:** k ∈ {1, 3, 6}, 10 fixed resampling seeds per k.
  Reference cases never overlap the query case.
- **Aggregation strategies (internal baselines):** B1 random single, B2 mean
  consensus, B3 median consensus, B4 trimmed mean, B5 disagreement-penalized,
  B6 quantile consensus, B7 Phase S best learned weighting (frozen).
- **Prompt configs:** P20-N1, P30-N0, P50-N0 (frozen before holdout).
- **Metrics:** Dice, IoU, AJI, component recall, precision/recall; reference
  instability (V_ref, V_ref_frac, MAD_ref, J_pred, best-worst gap); tail risk
  (worst-subset Dice, CVaR_0.1, CVaR_0.2, catastrophic failure rate); cost
  (reference count, DINOv2 passes, SAM calls, runtime).
- **Statistical unit:** the query/case (never the reference episode).

The key question for each external method is: **can it be run per reference
subset, with the reference image as the only varying input, so that we can
measure reference-induced instability on equal footing with our internal
baselines?**

---

## 1. GF-SAM (Tier 2) — alignment plan

### How to align
GF-SAM is a reference-guided FSS method: it takes a support (reference) set
and a query image and produces a segmentation mask. This maps naturally onto
our per-reference-subset execution model.

**Alignment steps:**

1. **Reference-subset loop.** For each query case, for each resampled reference
   subset S of size k, run GF-SAM with S as the support set and the query
   image as the target. This yields one mask per (query, S) episode — exactly
   the structure our V_ref / best-worst-gap computation expects.
2. **Frozen backbones.** GF-SAM already uses frozen SAM and frozen DINOv2-style
   features, so no training is required. Verify the SAM checkpoint version
   (GF-SAM uses SAM-ViT-H; our protocol uses SAM-B). Record this as a
   protocol deviation. If strict SAM-B alignment is required, re-run GF-SAM's
   prompt construction with SAM-B features and note the change.
3. **Prompt config.** GF-SAM has its own prompt-construction pipeline (PNA +
   PMC + Post-Gating) and does not accept our P20-N1 / P30-N0 / P50-N0
   configs. We treat GF-SAM's prompt construction as a fixed alternative
   prompt strategy and run it as-is. Do **not** force our prompt configs onto
   GF-SAM; that would defeat the purpose of comparing prompt-construction
   methods.
4. **Aggregation.** GF-SAM internally aggregates multiple masks via graph
   clustering. For the reference-resampling comparison, we also compute our
   own aggregations (mean/median consensus) over GF-SAM's per-subset outputs,
   so that GF-SAM can be compared both with its native aggregation and with
   our consensus rules.
5. **Single SAM call constraint.** GF-SAM may invoke SAM multiple times (one
   per candidate point cluster). Record the actual SAM-call count per episode
   in the cost ledger. If strict single-SAM-call alignment is required, cap
   GF-SAM to its top-1 cluster and flag the deviation.

### Changes needed
- Adapt GF-SAM's data loader to read MoNuSAC/GlaS reference-query pairs in our
  manifest format.
- Adapt the reference-subset iterator to feed GF-SAM the same k ∈ {1, 3, 6}
  subsets (same seeds) used by our internal baselines.
- Add a wrapper that, for each episode, returns the mask plus cost counters
  (SAM calls, DINOv2 passes, runtime).
- Decide and document the SAM checkpoint (SAM-B vs SAM-ViT-H) deviation.

### Metrics that can be compared
- **Directly comparable:** Dice, IoU, AJI, V_ref, V_ref_frac, MAD_ref,
  best-worst gap, CVaR, CFR, consensus gain (mean vs random single computed
  over GF-SAM's per-subset outputs).
- **Comparable with caveat:** cost metrics (GF-SAM uses more SAM calls than
  our single-call protocol; report both raw and per-call-normalized cost).
- **Not comparable:** prompt-config ablations (P20-N1 etc.) do not apply to
  GF-SAM since it has its own prompt construction.

---

## 2. MAUP (Tier 2) — alignment plan

### How to align
MAUP is a training-free, reference-guided few-shot medical segmentation
method that uses frozen SAM. Its cross-domain FSS framing (source domain →
target domain) needs to be mapped onto our within-dataset reference-resampling
framing.

**Alignment steps:**

1. **Reference-subset loop.** For each query case, for each resampled reference
   subset S of size k, treat S as the "source domain" support set and the
   query as the "target domain" image. Run MAUP's multi-center prompt
   generation + uncertainty-aware selection to produce a mask. This yields one
   mask per (query, S) episode.
2. **Frozen backbones.** MAUP is training-free and uses frozen SAM, matching
   our frozen-segmenter constraint. Verify the SAM checkpoint version and
   record any deviation.
3. **Within-dataset mapping.** MAUP expects a domain shift (source ≠ target).
   In our within-dataset setting, source and target are the same dataset
   (MoNuSAC), so the "cross-domain" machinery operates without a domain gap.
   This is a valid configuration (it tests MAUP's prompt construction, not its
   domain-adaptation story) and must be reported as a deviation from MAUP's
   intended use.
4. **Prompt config.** MAUP generates prompts via K-means clustering and
   uncertainty selection; it does not accept our P20-N1 / P30-N0 / P50-N0
   configs. Run MAUP as-is and treat its prompt construction as an alternative
   strategy, exactly as for GF-SAM.
5. **Aggregation.** Compute our mean/median consensus over MAUP's per-subset
   outputs in addition to MAUP's native dynamic-optimization output.
6. **Single SAM call constraint.** MAUP's dynamic optimization may invoke SAM
   multiple times. Record the actual SAM-call count. If strict single-call
   alignment is required, disable the dynamic-optimization loop and use only
   the initial prompt, flagging the deviation.

### Changes needed
- Adapt MAUP's data loader to our MoNuSAC/GlaS manifest and reference-subset
  iterator.
- Remove the cross-domain split logic and replace it with our within-dataset
  reference resampling (same k ∈ {1, 3, 6}, same seeds).
- Add a cost-counting wrapper.
- Document the within-dataset (no domain gap) deviation explicitly.

### Metrics that can be compared
- **Directly comparable:** Dice, IoU, AJI, V_ref, V_ref_frac, MAD_ref,
  best-worst gap, CVaR, CFR, consensus gain.
- **Comparable with caveat:** cost metrics (MAUP's dynamic optimization adds
  SAM calls; report raw and normalized).
- **Not applicable:** cross-domain transfer metrics (MAUP's intended domain-gap
  evaluation does not exist in our within-dataset setting).

---

## 3. SynPo (Tier 3, conditional) — alignment plan if code is released

SynPo is not currently reproducible (no public code). If the authors release
code, the alignment would be the most natural of all seven methods because
SynPo uses the same DINOv2 + SAM stack and the same few-shot medical
segmentation task.

### How to align (hypothetical)
1. **Reference-subset loop.** For each (query, reference subset S), run
   SynPo's CMSM + PSM with S as the support set. SynPo's support-query pair
   interface maps directly onto our reference-subset model.
2. **Frozen backbones.** SynPo uses frozen SAM-ViT and frozen DINOv2 —
   verify checkpoint versions and align to SAM-B if required.
3. **Prompt config.** SynPo has its own positive/negative point selection; run
   as-is.
4. **Metrics.** Same directly-comparable set as GF-SAM/MAUP.

### Status
Blocked on code release. Do not fabricate results. If code remains unavailable,
SynPo stays in `unsupported_comparators.md` as a Tier 3 concept comparison.

---

## 4. Methods with no alignment plan

The following methods are Tier 3 (no code) or Tier 4 (not applicable) and have
no protocol-alignment plan. See `unsupported_comparators.md` for rationale.

- **ProSAM (Tier 3):** no code/weights; cannot align without a release.
- **SAMIX (Tier 4):** requires RL training of SPNet and SAM2 adaptation; wrong
  paradigm (learned reference selection vs. test-time reference aggregation).
- **AlignSAM (Tier 4):** not reference-guided; RL prompter without references.
- **Point Prompt Defender (Tier 4):** not reference-guided; adversarial point
  optimization without references.

---

## Cross-method comparability notes

When GF-SAM and MAUP are reproduced, the following comparisons are valid:

1. **Reference-instability comparison (primary):** Does GF-SAM / MAUP exhibit
   the same reference-induced instability (V_ref_frac, best-worst gap) as our
   internal B1–B7? This is the core WP2b question. Run each external method
   under identical reference subsets and compute V_ref_frac etc. on its
   per-subset outputs.
2. **Mean-consensus gain comparison:** Does mean consensus over reference
   subsets improve Dice for GF-SAM / MAUP as it does for our internal
   baselines? Compute (B2 − B1) for each external method.
3. **Prompt-construction comparison:** GF-SAM (graph clustering), MAUP
   (multi-center uncertainty), and our DINOv2-matching point config represent
   three different prompt-construction strategies. Compare their absolute Dice
   and their instability profiles.
4. **Cost comparison:** Report SAM calls, DINOv2 passes, and runtime per
   method, normalized per query and per reference subset. Flag any method that
   exceeds the single-SAM-call budget.

Comparisons that are **not** valid:
- Prompt-config ablations (P20-N1 etc.) across methods with native prompt
  construction.
- Cross-domain transfer metrics for MAUP in our within-dataset setting.
- Any head-to-head "method X is better than method Y" claim that ignores cost
  differences or SAM-call-count differences.
