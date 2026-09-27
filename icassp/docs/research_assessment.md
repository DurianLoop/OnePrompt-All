# Research assessment and decision record

## Decision

**Proceed with a focused ICASSP 2027 paper, conditional on completing the two pre-submission checks below.** The contribution is not a universal prompt policy. It is a response-gated, two-candidate decision rule for a frozen reference-guided segmentation pipeline.

## Hypothesis

For a dense all-nuclei task, reference-background evidence improves where positive prompts are placed. Given those prompts, the segmenter's observable response to adding five background points (predicted-area and mask-quality changes) can identify a subset of queries for which that extra constraint improves Dice. An anchor fallback should prevent most harmful switches.

## Method frozen for this manuscript

1. A DINOv2-L/14 matcher embeds a reference image-mask pair and query image.
2. Foreground and background prototypes define a contrast score `z(cos(q,p_fg)) - 0.75 z(cos(q,p_bg))`; the top 40 patch centers are the shared positive prompts.
3. Candidate actions are an anchor and a challenger differing only in requested negative-point budget (`N0` or `N5`).
4. A train-only ExtraTrees response regressor receives deployment-available changes in SAM score, predicted area, and actual negative count, and accepts the challenger only if its predicted gain exceeds a validation-selected threshold. Otherwise it returns the anchor.

Both foundation components are frozen. Ground truth is used only to generate training rewards and to evaluate held-out queries.

## Confirmatory protocol

**MoNuSeg.** The primary evidence is a 5-fold nested CV on the official source-train set: 37 outer-test query cases, 5 inner-train references per query, and 185 reference episodes. Metrics are first averaged over the five references and then paired at the 37 independent query cases. The official 14-case test set is excluded from nested model selection.

**MoNuSAC.** The external protocol is patient-disjoint: 154/46/101 train/validation/test patches from 35/9/25 patients; support and query always come from different patients. Both patch and patient aggregation are retained; the manuscript does not use the post-hoc best fixed setting as a selected baseline.

**PanNuke.** A pre-registered field-level zero-shot diagnostic evaluates every nonempty public fold-1 tile (2,540) against five SHA-256-selected nonempty fold-2 supports. N0/N5 outcomes are averaged over supports before query-level paired bootstrap. The public release has no patient identifier; no patient-level claim is permitted. Raw N0/N5 inference finished before any gate artifact was loaded. Because saved folds 0--1 and 2--4 originate from sklearn 1.7.2 and 1.9.0 respectively, each was deserialized only in its matching environment before deterministic five-fold majority voting.

**Statistics.** Paired 95% confidence intervals use 10,000 bootstrap resamples with seed 20260715. Sign tests and win/tie/loss counts are reported. An existing 128-pair CPU/GPU audit found exactly equal coordinates and maximum Dice difference 0.000145, below its declared 0.0002 tolerance.

**Latency.** A separate GPU benchmark uses 5 warmups plus 30 CUDA-synchronized PanNuke support--query episodes at 256$\times$256. It opens no query annotation and includes the frozen DINOv2-L matcher, SAM ViT-B image embedding, and both N0/N5 decodes. It records both full per-episode work and a support-feature-cached path; these are inference latency measurements, not clinical-system throughput.

## Evidence-backed findings

| Claim | Estimate | Status |
|---|---:|---|
| FG-BG matching vs FG-only, MoNuSeg | +0.064725 Dice, 95% CI [0.046994, 0.084141], 36/0/1 | supported |
| response gate vs fold anchor, MoNuSeg | +0.010097, [0.003870, 0.018777], 15/21/1 | supported |
| response gate vs fixed N5, MoNuSeg | +0.009183, [0.000437, 0.019333], 14/21/2 | supported |
| oracle-gap reduction, MoNuSeg | 66.1% | diagnostic, not a deployment result |
| frozen MoNuSeg gate vs N0, MoNuSAC | +0.007904 patch Dice, [0.001635, 0.015518] | supportive transfer |
| frozen MoNuSeg gate vs N0, MoNuSAC patient aggregate | +0.007617, [0.001808, 0.014621], 14/5/6; sign-test p=.115 | bootstrap-positive directional corroboration, not standalone rank-test confirmation |
| fixed N5 vs N0, PanNuke | -0.012708, [-0.014252, -0.011150], 872/1/1667 | fixed-policy negative-transfer boundary |
| frozen five-fold gate vs N0, PanNuke | +0.004700, [0.004020, 0.005415], 1354/469/717 | supported field-level zero-shot diagnostic |
| nested object-recall diagnostic, MoNuSeg | -0.002156, [-0.005645, 0.000028] | no positive instance-recall claim |
| nested merge-error diagnostic, MoNuSeg | +0.002708, [-0.000219, 0.005620] | no positive instance-separation claim |
| per-support PanNuke sensitivity | all 5 fixed supports preserve gate-positive and N5-negative CI directions | post-hoc descriptive; no support selection |
| PanNuke public tissue-code sensitivity | 15/19 positive, 4/19 indeterminate, 0/19 negative gate CIs; 17--816 tiles/stratum | post-hoc descriptive; no tissue/subgroup selection |
| full DINOv2-L + SAM-B N0/N5 latency, RTX 5090 D v2 | median/p90 240.7/389.4 ms (30 episodes) | reproducible declared-inference measurement |
| support-cached latency, same setting | median/p90 178.9/309.3 ms (30 episodes) | deployment-relevant component measurement |
| same gate on SAM-Med2D | -0.000330, [-0.000872, 0.000024] | negative transfer boundary |
| adapted FoB, frozen MoNuSeg fold 0 | 0.004770 outer Dice after 30-episode inner selection | diagnostic failure; not head-to-head |

## Failure boundaries and prohibited claims

- The MoNuSAC gates do **not** beat its post-hoc global N5 value; no domain-optimal or universal-policy claim is allowed.
- PanNuke contains no patient identifier in the public release. Its positive five-fold-gate effect is field-level, zero-shot evidence only, and cannot be used to imply patient-disjoint or clinical generalization.
- The gate's nested union-Dice gain does not imply better instance separation: object recall and merge-error diagnostics have CIs crossing zero. No instance-quality claim is allowed.
- The gate does **not** transfer positively to SAM-Med2D; no model-agnostic claim is allowed.
- FoB is the closest published background-prompt method. A compatibility-documented one-fold adaptation now trains it on the materialized public MoNuSeg train images (115 train episodes), selects from four pre-registered checkpoints on 30 inner-validation episodes, and tests once on 40 outer-test episodes. It obtains 0.004770 Dice. The run uses SAM ViT-H and only one fold, unlike the primary five-fold DINOv2-L + SAM ViT-B protocol; it is a negative transfer diagnostic, not a numerical superiority comparison.
- The intervention costs a second prompt decode, not a second image embedding. The supplied timing is segmenter-only and excludes DINO matching and disk I/O.

## Pre-submission gates

1. Run a fully protocol-matched, multi-fold comparator reproduction if a numerical FoB comparison is desired; do not reinterpret the one-fold adaptation as such.
2. Run the frozen gate on one further patient-disjoint pathology cohort, or label the result exploratory if data access prevents it.
3. Rebuild with the official ICASSP 2027 template and run PDF eXpress once its submission portal is available.
4. Add author names, affiliations, funding, and the required ethical-compliance statement only after review anonymity rules are confirmed.

## Why this is suitable for ICASSP

The work is signal-processing oriented: a foreground/background contrast representation, structured prompt selection under a constrained action space, conditional decision-making based on response signals, and paired uncertainty analysis. It fits the official Biomedical Imaging & Signal Processing and Machine Learning & Generative AI areas. ICASSP 2027 permits four technical pages plus an optional references-only fifth page; the manuscript is deliberately scoped to that limit.
