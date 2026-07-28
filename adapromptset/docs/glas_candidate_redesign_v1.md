# GlaS Candidate Redesign v1 Pre-registration

Date: 2026-07-19
Protocol version: `adapromptset.phase2.glas_candidate_redesign_v1`
Parent protocol: `adapromptset.phase2.glas_strict.v2`
Parent Gate C status: `FAIL`
Parent Gate C decision: `REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`

## Failure Being Addressed

The frozen candidate generator passed precision enrichment, negative blank fraction, and coordinate validation on the Phase 2 GlaS strict validation split, but failed component coverage:

- validation query images: `30`;
- reference episodes: `90`;
- mean point precision at max pool: `0.971111`;
- mean random precision at max pool: `0.548889`;
- precision delta: `0.422222`;
- mean component coverage at max pool: `0.418820`;
- required component coverage: `0.600000`.

This redesign tests whether the failure is caused by an overly narrow positive candidate pool and insufficient local-peak diversity, not by changing the action bank or lowering the gate.

## Allowed Change

Exactly one candidate generator variant is allowed:

- positive candidate layout: `local_peaks`;
- positive candidate diagnostic pool sizes: `5`, `10`, `20`;
- negative candidate count follows the same limit for diagnostic symmetry;
- non-maximum spacing is fixed at `2.0` feature-grid cells;
- foreground/background contrast score remains the same DINOv2-L score contrast used by the frozen candidate generator;
- matcher remains frozen DINOv2-L;
- segmenter is not used in this Gate C redesign.

The `local_peaks` layout selects local maxima from the frozen contrast score map, applies fixed non-maximum spacing, and fills from the global score ranking only when there are fewer local peaks than the requested pool size.

## Prohibited Changes

- Do not use GlaS testA/testB.
- Do not change Gate C thresholds.
- Do not add or modify the GlaS action bank.
- Do not run Gate D/E/F until this redesigned Gate C is audited.
- Do not select among multiple redesign variants after seeing validation results.
- Do not modify MoNuSAC `prompt_set_bank_v2`, MoNuSAC episodes, or MoNuSAC Gate F thresholds.

## Evaluation Plan

Run the redesigned candidate pilot on the same fixed Phase 2 validation manifest:

`manifests/phase2/glas/episodes_val.csv`

Outputs must be isolated from the frozen-generator Gate C run:

- `reports/phase2_glas/candidates_redesign_v1/*`;
- `paper_phase2/figures/candidate_diagnostics_redesign_v1/*`;
- `cache/glas_candidates_redesign_v1/*`.

The strict audit must still aggregate to query level:

- 30 independent validation query images;
- 90 reference episodes;
- reference episodes are repeated measurements, not independent query units.

## Pass/Fail Rule

Pass criteria are unchanged:

- precision enrichment delta at least `0.15` or ratio at least `1.5`;
- max-pool component coverage at least `0.60`;
- coverage curve nondecreasing with pool size;
- negative pool mean blank fraction at most `0.50`;
- coordinate validation has zero systematic issues;
- query count and reference episode count match the protocol.

If this redesign fails, the GlaS candidate branch must be marked:

`STOP_GLAS_CANDIDATE_FAILURE`

and the policy path must stop rather than hiding the failure with action-bank expansion.
