# Phase 3 Scientific Pivot

Date: 2026-07-19
Branch: `phase3-glas-region-prompts`

## Historical Operational State

The Phase 2D operational state remains:

`STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

This state means SAM-B and SAM-L failed the registered P3 grouped-point viability protocol on all 30 GlaS strict-validation queries. It must not be deleted, overwritten, or reinterpreted as a success.

## Interpretation Correction

The evidence does not justify the broad statement that the SAM backbone completely fails on GlaS, because:

- coordinate audit passed;
- cost audit passed;
- P4 GT union-box Dice was positive for both SAM-B and SAM-L;
- P4 margins over C1/C2 controls were positive;
- P2 one-group multi-positive prompting was substantially stronger than P3;
- failure was worse for high-component-count morphology.

The corrected scientific interpretation is:

`STOP_GLAS_GROUPED_POINT_PROMPT_PATH_CONFIRMED`

The research direction is:

`PIVOT_GLAS_REGION_PROMPT_REQUIRED`

## New Research Question

Given reference-conditioned matching evidence, can an automatic region prompt provide a viable interface to a frozen segmentation foundation model for whole-image gland segmentation?

Sub-questions:

- RQ1: Can frozen reference-query matching evidence generate sufficiently high-quality automatic region proposals?
- RQ2: Do automatic box or mask prompts into SAM-B improve over matched controls and the frozen point baseline?
- RQ3: Do queries prefer different region prompt representations enough to justify a later prompt-type router?

## Temporary Method Names

- `AdaRegionPrompt`
- `AdaPromptType`

No final paper title is frozen before results.
