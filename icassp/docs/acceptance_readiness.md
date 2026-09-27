# ICASSP 2027 acceptance-readiness decision record

## Current decision

The package is a defensible anonymous conference draft, but no honest process can verify or promise an 80% acceptance probability. The current record supports submission after the official-template conversion; it does not support an 80%-certainty claim because two material reviewer risks remain externally unresolved.

## Evidence that materially strengthens the submission

| Dimension | Current evidence | Assessment |
|---|---|---|
| Primary validity | 37 independent MoNuSeg query cases, five-fold nested CV, five reference episodes per query, paired bootstrap | strong for the stated DINOv2-L + SAM ViT-B protocol |
| Mechanistic ablation | FG--BG N0 beats FG-only by +0.0647 Dice, 95% CI [0.0470, 0.0841] | strong |
| Adaptive action rule | four-budget N0/N5/N10/N20 selector beats the prior two-action selector by +0.00315 and best fixed N5 by +0.01234 on held-out MoNuSeg queries | strong protocol evidence; cross-domain policy transfer remains unsupported |
| Instance-quality boundary | nested object-recall and merge-error diagnostics do not have positive CIs | efficacy claim properly limited to union-mask Dice |
| Patient-level transfer | 101-patch MoNuSAC test from 25 held-out patients, support/query patient-disjoint | useful, bounded external evidence |
| Large additional stress test | full 2,540-query PanNuke field-level N0/N5/N10/N20 cohort; N0 is best and larger fixed budgets are harmful | valuable retained counterexample, not cross-domain policy corroboration |
| Robustness | all five fixed PanNuke support strata preserve the direction descriptively | reassuring but post-hoc descriptive only |
| Reproducibility | frozen artifacts, version-isolated gate deserialization, 16-check integrity audit, CUDA-synchronized latency record | strong package hygiene |

## Risks that remain material

1. **No protocol-matched FoB comparison.** A five-fold FoB diagnostic is now complete, but it uses SAM ViT-H and a different prompt representation. It cannot establish numerical superiority to FoB.
2. **No second locally feasible patient-disjoint pathology cohort.** The only local candidate, GlaS, has an official-test-preserving split with only seven train cases and one validation case after patient-overlap exclusion. Its current raw-data location contains no usable images/masks. PanNuke cannot fill this gap because its public release has no patient identifier.
3. **Transfer and endpoint scope.** The four-action incremental effect is positive on MoNuSeg, but its frozen MoNuSAC transfer is not significant; auxiliary object-recall/merge-error diagnostics do not establish better instance separation.
4. **Formatting and submission environment.** The current PDF is a visual/content draft built with `article`; it still requires the official ICASSP 2027 class, author-side final compliance checks, and PDF eXpress validation.

## Evidence-based submission wording

Safe central claim: *For a frozen DINOv2-L + SAM ViT-B reference-guided nuclei pipeline, response-gating a fixed N0/N5 background constraint improves nested MoNuSeg performance and rejects a harmful fixed-N5 policy in a pre-registered, field-level PanNuke transfer diagnostic.*

Prohibited upgrades: SOTA, universal policy, patient-level PanNuke generalization, model-agnostic transfer, or superiority to FoB.

## Conditions for a credible “high-confidence” submission

The closest actionable path to materially higher confidence is:

1. Acquire/authorize one additional pathology cohort with reliable patient identity and preserve an untouched patient-disjoint test split.
2. Reproduce the closest comparator under the same DINOv2-L + SAM ViT-B support/query protocol across all outer folds, or remove any expectation of a direct comparator claim.
3. Convert the source to the official ICASSP 2027 template and validate its final PDF.

Until (1) and (2) are met, the correct status is **submission-ready focused draft with material acceptance risk**, not a verified 80%-acceptance paper.
