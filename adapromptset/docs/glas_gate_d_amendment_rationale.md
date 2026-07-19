# GlaS Gate D Amendment Rationale

Date: 2026-07-19
Protocol: `adapromptset.phase2d.glas_gate_d_confirmation.v1`

## What Remains Frozen

The previous Phase 2 facts remain unchanged:

- MoNuSAC remains frozen at `PIVOT_ORACLE_ANALYSIS`.
- GlaS official testA/testB remain locked and must not be read.
- GlaS Gate C passed only after the single predeclared `local_peaks` candidate redesign.
- The redesigned candidate generator is now frozen.
- The original 6-case Gate D reports remain permanent historical evidence.

The original 6-case Gate D decision is preserved as:

`EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`

It must not be deleted, overwritten, or disguised as if it never happened.

## Why Confirmation Is Allowed

The original Gate D smoke used a strict all-case rule: if any selected case did not clearly beat random/mismatched controls, the segmenter was marked as failed. That rule is too strong for an execution/viability smoke gate with only six cases because it cannot distinguish:

- systematic segmenter execution failure;
- an unstable morphology case;
- a strong random or mismatched control by chance;
- a prompt family that is unsuitable for one gland morphology;
- small-sample fragility.

The scientific purpose of Gate D is narrower: confirm whether SAM-B or SAM-L responds to deterministic privileged prompts, whether coordinate/label/resize/multimask/fusion are correct, whether privileged prompts are generally better than cost-matched controls at the query level, and whether catastrophic failures are rare enough to proceed.

## Frozen Confirmation Standard

The confirmation study uses all 30 strict-validation query images. It does not drop difficult cases and does not select cases after seeing results.

Primary protocol:

`P3_component_grouped_centroids`

Primary matched control:

`mean(C1_random_coordinate, C2_mismatched_reference)`

Gate D pass for a segmenter requires all of:

- coordinate and label audits pass;
- P3 query-level mean Dice margin over matched control is at least `0.05`;
- P3 query win rate is at least `0.70`;
- P3 catastrophic failure rate is at most `0.10`;
- query-level paired bootstrap point estimate is positive;
- cost matching passes for prompt count, group count, and SAM calls;
- no hidden oracle selection is used.

Bootstrap CI lower bound may be at or below zero, but then an uncertainty warning is recorded rather than silently hidden.

## Why This Is Not Threshold Relaxation

This amendment is frozen before running any new SAM inference. It uses only the 30 strict-validation queries and never reads testA/testB. It does not modify Gate C, the candidate generator, the action bank, the matcher, the reference manifest, or any policy model.

After this confirmation, no second Gate D criterion revision is allowed. The final Gate D state must be one of the registered states in the confirmation protocol.
