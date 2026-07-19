# AdaPromptSet Phase 2 GlaS Strict Diagnostic Protocol

Date: 2026-07-19
Protocol version: `adapromptset.phase2.glas_strict.v2`
Status: `PRE_EXPERIMENT_PROTOCOL`

## Scientific Question

In GlaS gland segmentation, do validation queries show enough best-prompt-set variation that cost-aware adaptive action selection has learnable headroom?

The target decision problem is:

`A* = argmax U(A | R, M_R, Q)` subject to `C(A) <= B`.

Each action must be a complete prompt-set structure: positive points, negative points, prompt groups, SAM calls, estimated cost, and measured cost.

## Frozen Components

- Matcher: DINOv2-L.
- Segmenter: SAM-B.
- No backbone change is allowed because of poor GlaS performance unless Gate D fails under its registered rule.
- No unrestricted coordinate generation, PPO, DQN, iterative SAM interaction, learned coordinate offsets, SAM-response-conditioned refinement, broad model matrix, or universal transfer claim is allowed.

## Data Roles

| Split | Role in Phase 2 |
| --- | --- |
| official train 85 | split into 55 action-utility training images and 30 strict validation query images |
| official testA 60 | locked evaluation split; not used before a test freeze |
| official testB 20 | second locked evaluation split; not used before testA report is complete |

Because patient IDs overlap between official train and official testA/testB, Phase 2 records image-level official split isolation and patient-overlap limitation. It does not claim patient-level isolation for official tests.

## Phase 2 Split Rule

- Seed: `20260719`.
- Split only official train cases.
- Target training pool size: `55`.
- Target validation query pool size: `30`.
- Stratify by GlaS benign/malignant grade when assigning the 55/30 split.
- Preserve full image and mask hashes.
- Preserve patient IDs and source grades.
- Each validation query gets 3 fixed references from the 55-image training pool.
- A reference must not be the same image as the query.
- If possible, references should be grade-matched; otherwise use deterministic fallback from the training pool and record fallback.
- Query-level aggregation is the unit for Gate F. Per-reference rows are retained only for reference sensitivity.

## Gate C: Candidate Quality

Use the frozen candidate generator first. Do not add GlaS-specific heuristics before Gate C.

Required outputs:

- `reports/phase2_glas/candidates/*`
- `paper_phase2/figures/candidate_diagnostics/*`
- `cache/glas_candidates/*`

Default pass requirements:

- top candidate precision beats random foreground prevalence by at least 15 percentage points or reaches 1.5x random prevalence;
- maximum candidate-pool gland/component coverage is at least 60%;
- coverage increases with candidate-pool size rather than saturating immediately;
- negative pool is not mostly easy empty background;
- coordinate transforms have no systematic error.

If Gate C fails, only one predeclared candidate redesign is allowed. If that redesign also fails, the status becomes `STOP_GLAS_CANDIDATE_FAILURE`.

## Gate D: Segmenter Smoke

Privileged sanity prompts may be used only as upper-bound/API checks. They must not be mixed with automatic methods as fair baselines.

Required checks include GT centroid, multi-centroid, GT box, GT positive plus controlled negative, random point, and mismatched reference point. The gate must cover single gland, multi-gland, small gland, large gland, adjacent glands, irregular malignant morphology, and benign morphology when available from the split.

If SAM-B systematically fails, exactly one replacement segmenter may be tested: SAM-L or SAM2, not both.

## GlaS Action Bank

Create a new GlaS bank:

- `configs/actions/glas_prompt_set_bank_v1.yaml`
- `configs/actions/glas_prompt_set_bank_v1.json`

Do not reuse the MoNuSAC bank hash as the GlaS bank hash.

Initial registered actions:

- A00 safe anchor.
- A01 Top-3 one-group.
- A02 Top-5 one-group.
- A03 Top-3 plus hard negatives.
- A04 local maxima one-group.
- A05 KMeans-3 groups.
- A06 KMeans groups plus local negatives.
- A07 spatial spread groups.
- A08 component-aware groups.
- A09 boundary-aware negative one-group.
- A10 graph grouping, only if reproducible.
- A11 graph grouping plus local negatives, only if A10 is valid.

No validation-oracle result may be used to add a new action. No DBSCAN/MeanShift parameter sweep may be enumerated after seeing validation utilities.

## Utility Generation

Utility data for training can only come from the 55-image training pool. Validation utility is used only for Gate F and model selection permitted by the protocol. Ground truth is never a policy input.

Budgets:

- `B_call=1`
- `B_call=3`
- `B_call=5`

Report Dice at each supported budget. If the action bank does not support a budget, mark it unavailable rather than duplicating an action.

## Gate F: GlaS Oracle Gap

Compute on 30 validation query images using query-level aggregation across fixed references.

Required outputs:

- best fixed action;
- per-query oracle action;
- oracle gap;
- action win distribution;
- action entropy;
- budget-specific oracle gap;
- query-level paired bootstrap 95% CI.

Gate F can end in exactly one state:

- `CONTINUE_GLAS_POLICY`
- `PIVOT_GLAS_DETERMINISTIC`
- `PIVOT_GLAS_ORACLE_ANALYSIS`
- `REDESIGN_GLAS_ACTION_BANK_ONCE`
- `STOP_GLAS`

Policy training is allowed only for `CONTINUE_GLAS_POLICY`.

## Formal Test Boundary

Formal testA/testB evaluation is allowed only after Gate H succeeds or a deterministic final method is frozen. Before formal testing, create:

- `docs/glas_test_freeze.md`
- `configs/frozen/glas_final_protocol.yaml`
- `reports/phase2_test_freeze_hashes.json`

No retuning after testA is allowed before testB.
