# Phase 0 Scope and Structure Audit

Status: `PASS_SCOPE_STRUCTURE`
Policy training remains prohibited for the current MoNuSAC evidence.

## Structure

- Required paths: 26
- Missing paths: 0
- Toolchain: `PASS`

## Research Boundary

- Research question: Under frozen DINOv2-L and SAM-B, does budget-constrained structured prompt-set selection show enough query-specific oracle heterogeneity to justify learned selection?
- Does not claim first-to-learn prompt policy.
- RQs: 3 / 3
- Hypotheses: 5 / 5

## Similar Work Boundaries

- GF-SAM: `True`
- SynPo: `True`
- AlignSAM: `True`
- Point Prompt Defender: `True`

## Non-mainline Exclusions

- `sequential_rl`: present=`True`, mainline=`False`
- `continuous_coordinate_generation`: present=`True`, mainline=`False`
- `online_sam_response_refinement`: present=`True`, mainline=`False`

## Notes

- The executable mainline remains one-shot, offline, and set-level.
- This audit is not deployable policy evidence and does not reopen MoNuSAC policy training.
- Sequential RL, continuous coordinate generation, and online SAM-response refinement are non-mainline extensions.
