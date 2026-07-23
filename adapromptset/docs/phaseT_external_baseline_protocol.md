# Phase T External Baseline Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §7 (WP2b — Related Method Comparison)

## Purpose

Phase T must compare the internal B1–B7 reference-aggregation families
against external reference-guided / prompt-robustness methods under the
SAME reference-resampling protocol. This document defines the comparator
tiers, the candidate method list, and the minimum reproduction requirement.

## Comparator Tiers

Every candidate external method is classified into exactly one tier. The
tier determines how it is evaluated and what claims may be made about it.

### Tier 1 — Official Code and Weights Directly Reusable

The method's official code and pretrained weights run under the Phase T
reference-resampling protocol with no interface change beyond data loading.
The method is evaluated on the same 6-reference pool, same k∈{1,3,6}
subsets, same queries, and same metrics as the internal B1–B7 families.

### Tier 2 — Official Code Exists but Interface Incompatible

The method's official code exists but requires minimal necessary adaptation
to accept Phase T reference images and emit a segmentation mask. Every
adaptation is recorded in `reports/phaseT_external_baselines/implementation_notes.md`
with: file changed, line, rationale, and validation that the adaptation does
not unfairly favor or disadvantage the method.

### Tier 3 — No Available Code or Weights (Conceptual Only)

The method has no public code or weights. Only a conceptual comparison is
made; no numerical results are fabricated. The comparison documents the
method's stated mechanism and how it relates to reference-induced
instability, without claiming quantitative superiority or inferiority.

### Tier 4 — Not Applicable to Current Task

The method is not applicable to the Phase T task (e.g., it requires a
different input modality, a different prompt type, or a non-segmentation
output). The reason is documented; no evaluation is performed.

## Candidate Methods

The candidate external method list (directive §7.1):

| Method | Primary Mechanism | Reference-Instability Relevance |
|---|---|---|
| GF-SAM | Graph-based foreground grouping of point prompts | Grouping strategy; may affect reference sensitivity |
| SynPo | Positive / negative point construction from support | Point construction; directly reference-dependent |
| ProSAM | Probabilistic prompt robustness | Prompt uncertainty; related to M2 risk prediction |
| MAUP | Uncertainty-aware prompt quantity | Adaptive k; related to M1 dynamic budget |
| SAMIX | Reference selection / mixing | Reference selection; directly competes with B7 |
| AlignSAM | RL iterative prompting | Iterative refinement; different cost profile |
| Point Prompt Defender (PPD) | Point filtering / defense | Point robustness; may affect reference sensitivity |

## Minimum External Comparator Requirement (directive §7.3)

Before the Publication Readiness Gate (§20) may pass, AT LEAST ONE of the
following must be satisfied:

- **Option A**: Reproduce 2 external methods under Tier 1 or Tier 2.
- **Option B**: Reproduce 1 external method (Tier 1 or Tier 2) + 3 strong
  internal method families (B1–B7 cover at least 3 distinct families).
- **Option C**: Cannot reproduce any external method, but provide a public
  protocol, code interface, and clear limitations. In this case the
  submission positioning is LOWERED to a benchmark-without-external-comparison
  paper, and the claim registry is tightened accordingly.

Option C does NOT satisfy the Hybrid paper route (§13); it only satisfies a
reduced Benchmark-only route.

## Reproduction Protocol

For every Tier 1 / Tier 2 reproduced method:

1. Record official code repository URL / commit / license in
   `reports/phaseT_external_baselines/comparator_availability.csv`.
2. Run the method on the same MoNuSAC validation queries (41) and the same
   6-reference pool as the internal B1–B7 families.
3. Compute the full metric suite (V_ref, MAD_ref, J_pred, CVaR, CFR, Dice,
   cost) for the external method.
4. Report whether the external method exhibits reference instability under
   the Phase T protocol (this is the WP2b research question).
5. Apply the paired bootstrap 95% CI procedure when comparing the external
   method to B2 mean.
6. Do NOT claim the external method is "fully outperformed" (prohibited
   claim #5); report only the head-to-head metrics under the matched
   protocol.

## Comparator Matrix Output

`reports/phaseT_external_baselines/` must contain:

```
comparator_availability.csv      # tier, code URL, license, status
implementation_notes.md          # Tier 2 adaptation log
protocol_alignment.md            # how each method aligns with Phase T protocol
reproduced_results.csv           # per-query metrics for Tier 1/2 methods
reference_instability_results.csv # V_ref / CVaR / CFR per external method
unsupported_comparators.md       # Tier 3 / Tier 4 documentation
```

## Prohibited External-Baseline Operations

1. Fabricating numerical results for Tier 3 methods.
2. Running an external method on test data.
3. Tuning an external method's hyperparameters on the holdout.
4. Claiming an external method is "fully outperformed" without a matched
   head-to-head under the Phase T protocol.
5. Excluding an external method's failure cases from the report.
6. Downloading code / data / weights without user approval (directive §23).
