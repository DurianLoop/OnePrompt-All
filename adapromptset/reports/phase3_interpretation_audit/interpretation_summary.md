# Phase 3 Interpretation Audit Summary

Status: `PHASE3_INTERPRETATION_AUDIT_COMPLETE`

Interpretation state: `PIVOT_GLAS_REGION_PROMPT_REQUIRED`

The historical operational Gate-D state remains `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED` under the registered P3 grouped-point viability protocol.

Scientific correction: grouped sparse point prompting failed; region prompting remains viable as indicated by the privileged box upper bound.

Key evidence:
- SAM-B P4 mean Dice `0.727637`, margin `0.044447` vs C1 and `0.064274` vs C2.
- SAM-L P4 mean Dice `0.723478`, margin `0.039224` vs C1 and `0.094778` vs C2.
- SAM-B P3 mean matched-control margin `-0.157937`.
- SAM-L P3 mean matched-control margin `-0.096909`.

Failure taxonomy counts: `{'image_wide_output': 19, 'low_dice_catastrophic': 13, 'non_catastrophic_control_advantage': 16, 'under_coverage_false_negative_dominant': 12}`

No new SAM inference was run for this audit. GlaS testA/testB were not accessed.
