# AdaPromptSet Phase 2D Final Report

Date: 2026-07-19
Branch: `phase2-glas-strict`
Preregistration commit: `f5c2de3` (`phase2d: preregister glas gate d confirmation`)
Final status: `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

This report supersedes only the final GlaS segmenter-viability conclusion from the previous 6-case smoke. It does not delete or overwrite the original Gate D failure report.

## Frozen History

MoNuSAC remains frozen at `PIVOT_ORACLE_ANALYSIS`. No MoNuSAC policy training, post-hoc action redesign, or test-time action search is allowed.

GlaS Gate C passed only after the single preregistered `local_peaks` candidate redesign:

- precision delta: `0.433889`;
- precision ratio: `1.806818`;
- component coverage: `0.630221`;
- negative blank fraction: `0.135556`;
- coordinate validation: `PASS`.

The original 6-case Gate D result is preserved as:

`EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`

Original 6-case evidence:

| Segmenter | Cases | Mean privileged Dice | Mean control Dice | Status |
| --- | ---: | ---: | ---: | --- |
| SAM-B | 6 | `0.733908` | `0.638274` | `FAIL` |
| SAM-L | 6 | `0.762905` | `0.412671` | `FAIL` |

## Why Confirmation Was Allowed

The original Gate D all-case rule marked the segmenter failed if any selected case did not beat controls. That rule is too strong for a 6-case execution/viability smoke because it cannot separate systematic segmenter failure from morphology-specific instability, strong random controls, or small-sample fragility.

Phase 2D therefore allowed exactly one preregistered 30-query confirmation before any Gate E/F reopening. The confirmation used only GlaS official-train strict-validation queries. GlaS testA/testB remained locked and unused.

## Protocol

| Item | Value |
| --- | --- |
| Query pool | 30 strict-validation GlaS official-train images |
| Statistical unit | query image |
| Primary family | `P3_component_grouped_centroids` |
| Primary control | `mean(C1_random_coordinate, C2_mismatched_reference)` |
| C2 mapping | `first_frozen_episode_support_mask` |
| Segmenters | SAM-B and SAM-L |
| Pass margin | P3 mean paired Dice margin `>= 0.05` |
| Pass win rate | P3 query win rate `>= 0.70` |
| Catastrophic limit | P3 catastrophic failure rate `<= 0.10` |
| Bootstrap rule | paired query-level point estimate must be positive |
| Gate changes after run | prohibited |

Preregistration artifacts:

- `docs/glas_gate_d_confirmation_protocol_v1.md`
- `docs/glas_gate_d_amendment_rationale.md`
- `configs/segmenter/glas_gate_d_confirmation_v1.yaml`
- `reports/phase2_glas/gate_d_confirmation/protocol_snapshot.json`
- `reports/phase2_glas/gate_d_confirmation/input_hashes.json`
- `reports/phase2_glas/gate_d_confirmation/compute_plan.json`

## Confirmation Results

| Segmenter | P3 privileged Dice | Matched control Dice | P3 margin | Win rate | Bootstrap CI | Cat. rate | Status |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| SAM-B | `0.509561` | `0.667498` | `-0.157937` | `0.433333` | `[-0.250074, -0.076146]` | `0.666667` | `FAIL` |
| SAM-L | `0.470705` | `0.567614` | `-0.096909` | `0.466667` | `[-0.196157, -0.001773]` | `0.400000` | `FAIL` |

Coordinate audit: `PASS`

Cost matching audit: `PASS`

testA/testB access: `false`

Both segmenters fail the preregistered primary criterion. The point estimate is negative for both SAM-B and SAM-L, win rates are below `0.70`, and catastrophic failure rates exceed `0.10`.

## Prompt-Family Diagnostics

Secondary diagnostics do not change the primary Gate D decision.

| Segmenter | Family | Mean Dice | Margin | Mean calls |
| --- | --- | ---: | ---: | ---: |
| SAM-B | P1 | `0.348` | `-0.069` | `1.0` |
| SAM-B | P2 | `0.609` | `-0.064` | `1.0` |
| SAM-B | P3 | `0.510` | `-0.158` | `2.9` |
| SAM-B | P4 | `0.728` | `0.054` | `1.0` |
| SAM-B | P5 | `0.342` | `-0.150` | `1.0` |
| SAM-L | P1 | `0.378` | `0.012` | `1.0` |
| SAM-L | P2 | `0.660` | `0.007` | `1.0` |
| SAM-L | P3 | `0.471` | `-0.097` | `2.9` |
| SAM-L | P4 | `0.723` | `0.067` | `1.0` |
| SAM-L | P5 | `0.404` | `0.017` | `1.0` |

The box upper-bound diagnostic P4 is positive, but it cannot replace the registered primary point/grouped prompt protocol and cannot be reported as an automatic AdaPromptSet method.

## Morphology Diagnostics

| Group | Queries | Mean P3 margin | Cat. rate |
| --- | ---: | ---: | ---: |
| benign | 13 | `-0.220299` | `0.538462` |
| malignant | 17 | `-0.056400` | `0.529412` |
| high_component_count | 16 | `-0.224595` | `0.718750` |
| large_foreground | 15 | `-0.095095` | `0.600000` |

Morphology analysis explains failure modes but does not reopen the overall Gate D decision.

## Generated Artifacts

Reports:

- `reports/phase2_glas/gate_d_confirmation/gate_d_confirmation_summary.json`
- `reports/phase2_glas/gate_d_confirmation/gate_d_confirmation_summary.md`
- `reports/phase2_glas/gate_d_confirmation/query_level_metrics.csv`
- `reports/phase2_glas/gate_d_confirmation/prompt_family_metrics.csv`
- `reports/phase2_glas/gate_d_confirmation/matched_control_metrics.csv`
- `reports/phase2_glas/gate_d_confirmation/sam_b_metrics.csv`
- `reports/phase2_glas/gate_d_confirmation/sam_l_metrics.csv`
- `reports/phase2_glas/gate_d_confirmation/bootstrap_summary.json`
- `reports/phase2_glas/gate_d_confirmation/bootstrap_summary.md`
- `reports/phase2_glas/gate_d_confirmation/win_tie_loss.csv`
- `reports/phase2_glas/gate_d_confirmation/catastrophic_failures.csv`
- `reports/phase2_glas/gate_d_confirmation/coordinate_audit.json`
- `reports/phase2_glas/gate_d_confirmation/cost_audit.csv`
- `reports/phase2_glas/gate_d_confirmation/runtime_audit.csv`
- `reports/phase2_glas/gate_d_confirmation/morphology_analysis.csv`
- `reports/phase2_glas/gate_d_confirmation/morphology_analysis.md`
- `reports/phase2_glas/gate_d_confirmation/failure_case_registry.csv`
- `reports/phase2_glas/gate_d_confirmation/run_metadata.json`
- `reports/phase2_glas/gate_d_confirmation/compute_ledger.csv`

Figures:

- `paper_phase2/figures/gate_d_confirmation/paired_privileged_control_dice_scatter.png`
- `paper_phase2/figures/gate_d_confirmation/paired_dice_difference_distribution.png`
- `paper_phase2/figures/gate_d_confirmation/per_query_samb_saml_comparison.png`
- `paper_phase2/figures/gate_d_confirmation/prompt_family_comparison.png`
- `paper_phase2/figures/gate_d_confirmation/win_tie_loss.png`
- `paper_phase2/figures/gate_d_confirmation/catastrophic_failure_summary.png`
- `paper_phase2/figures/gate_d_confirmation/dice_vs_component_count.png`
- `paper_phase2/figures/gate_d_confirmation/dice_vs_foreground_area.png`
- `paper_phase2/figures/gate_d_confirmation/benign_malignant_stratification.png`
- `paper_phase2/figures/gate_d_confirmation/representative_failure_panels.png`

Paper assets:

- `paper_phase2/tables/table_gate_d_confirmation.tex`
- `paper_phase2/tables/table_prompt_family_viability.tex`
- `paper_phase2/tables/table_morphology_failures.tex`
- `paper_phase2/sections/gate_d_confirmation.tex`

## Gate Consequence

Final state:

`STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

Therefore:

- Gate E action-bank validation remains closed;
- Gate F oracle-gap analysis remains closed;
- Gate G utility reliability remains closed;
- Gate H policy learning remains prohibited;
- Gate I formal test remains prohibited;
- GlaS testA/testB remain locked and unused.

## Compute

| Item | GPU-hours |
| --- | ---: |
| Phase 2 before Phase 2D | `0.039522` |
| Phase 2D confirmation | `0.091388` |
| Total recorded | `0.130910` |
| Phase 2D budget | `16.000000` |

## Claim Boundary

Allowed claim:

`On all 30 GlaS strict-validation queries, both SAM-B and SAM-L fail the preregistered P3 privileged-vs-matched-control viability criterion; the GlaS AdaPromptSet policy path remains closed and testA/testB remain unused.`

Not allowed:

- no GlaS policy success claim;
- no automatic point-prompt method claim from privileged GT prompts;
- no Gate E/F reopening;
- no formal testA/testB result;
- no cross-segmenter transfer claim.

## Reproduction

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_glas_gate_d_confirmation.py --project-root D:\SAM\adapromptset --device cuda:0
```
