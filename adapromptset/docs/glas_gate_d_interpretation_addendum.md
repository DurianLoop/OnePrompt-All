# GlaS Gate D Interpretation Addendum

Date: 2026-07-19

## Preserved Historical Record

The original Phase 2D operational state remains:

`STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

This is the historical Gate-D state under the registered P3 grouped sparse-point viability protocol.

## Addendum Interpretation

The Phase 2D evidence supports the narrower conclusion:

`grouped sparse point prompting failed`

It does not support the broader conclusion:

`the frozen SAM backbone completely fails on GlaS`

The scientific interpretation state for Phase 3 is:

`PIVOT_GLAS_REGION_PROMPT_REQUIRED`

## Control Interpretation

The Phase 2D C2 control used another GlaS gland reference from the frozen support manifest. It is therefore relabeled in scientific discussion as:

`alternative-reference control`

The original Phase 2D CSV column names and historical reports are not modified.

## Evidence

From `reports/phase3_interpretation_audit/interpretation_summary.json`:

- SAM-B P3 matched-control margin: `-0.157937`
- SAM-L P3 matched-control margin: `-0.096909`
- SAM-B P4 mean Dice: `0.727637`
- SAM-L P4 mean Dice: `0.723478`
- high-component-count mean P3 margin: `-0.144578`
- low-component-count mean P3 margin: `0.026974`

## Consequence

Phase 3 must not reopen P3 or change the Gate D criterion. It may only study whether automatic region prompts derived from frozen matching evidence provide a viable prompt interface.
