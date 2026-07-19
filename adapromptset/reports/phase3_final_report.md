# AdaPromptSet Phase 3 Final Report

Date: 2026-07-19
Branch: `phase3-glas-region-prompts`
Preregistration commit: `471f529`
Final state: `STOP_GLAS_REGION_PROPOSAL_FAILURE`

## Interpretation Correction

The historical Phase 2D operational state remains:

`STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

The Phase 3 scientific correction is narrower:

`grouped sparse point prompting failed; region prompting remains viable as indicated by the privileged box upper bound`

C2 is discussed as `alternative-reference control`, not as a true negative control.

## Gate States

| Gate | State |
| --- | --- |
| Interpretation audit | `PHASE3_INTERPRETATION_AUDIT_COMPLETE` |
| R1 proposal quality | `STOP_GLAS_REGION_PROPOSAL_FAILURE` |
| R2 automatic region prompt viability | `BLOCKED_BY_R1_FAILURE` |
| R3 region action oracle | `BLOCKED_BY_R1_FAILURE` |
| Router | `PROHIBITED_BY_R1_FAILURE` |
| Formal testA/testB | `PROHIBITED_BY_R1_FAILURE` |

## R1 Result

| Family | Route | Metric | Improvement | Win rate | Recall | Empty | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R00 direct matching mask | mask | `0.178754` | `-0.416321` | `0.066667` | `0.099628` | `0.000000` | `FAIL` |
| R03 automatic union box | box | `0.744785` | `0.025965` | `0.200000` | `0.790664` | `0.000000` | `FAIL` |
| R04 largest-component box | box | `0.019410` | `0.002199` | `0.100000` | `0.029807` | `0.000000` | `FAIL` |
| R05 top-3 component boxes | box | `0.271747` | `0.016568` | `0.166667` | `0.063224` | `0.000000` | `FAIL` |

The strongest automatic method was `R03_automatic_union_box`: mean box IoU `0.744785`, but its improvement over size-matched random boxes was only `0.025965` and win rate was `0.200000`, below the preregistered R1 thresholds.

Raw matching mask Dice was `0.178754`. SAM-refined Dice was not measured because R1 failed before R2.

## Upper Bound And Recovery

GT region upper bounds remain evidence that the SAM backbone can respond to privileged region prompts:

- SAM-B P4 GT box Dice: `0.727637`
- SAM-L P4 GT box Dice: `0.723478`

Automatic region recovery is not applicable because R2 was not run.

## Consequence

Phase 3 stops at:

`STOP_GLAS_REGION_PROPOSAL_FAILURE`

Therefore:

- no SAM-B R2 automatic region inference;
- no SAM-L replication;
- no R3 oracle matrix;
- no router training;
- no testA/testB access.

## Claim Boundary

Allowed claim:

`Frozen matching evidence and sparse local-peak candidates did not produce region proposals that beat preregistered R1 controls on GlaS strict validation; the automatic region prompt route stops before SAM refinement.`

Not allowed:

- automatic region prompting works;
- SAM refinement gain;
- region action oracle gap;
- prompt-type router improvement;
- formal testA/testB result.

## Artifacts

- `reports/phase3_interpretation_audit/interpretation_summary.json`
- `reports/phase3_glas/split/region_split_report.json`
- `reports/phase3_glas/region_candidate_calibration/candidate_pilot_summary.json`
- `reports/phase3_glas/gate_r1_region_proposal/r1_summary.json`
- `reports/phase3_gate_status.md`
- `reports/phase3_compute_ledger.csv`
- `paper_phase3/main.tex`
- `paper_phase3/build/main.pdf`
- `reports/phase3_paper_assets.json`

## Compute

Phase 3 GPU-hours: `0.029398` out of the `40` GPU-hour budget.

testA/testB status: locked and unused.

Paper source/PDF status: `PASS_8_PAGE_BUILD`.

Paper PDF pages: `8`.

Paper PDF SHA256: `B75339E91F5F9BCE1F77DA8FA84E3E1F9095DB2623954D78E376B9FB14B0DD2A`.

LaTeX log scan: `PASS_NO_OVERFULL_UNDEFINED_MISSING_OR_RERUN_WARNINGS`.

Rendered PNG pages: `8`.

## Reproduction

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_phase3_interpretation.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_glas_phase3_region_splits.py --assignments-csv D:\SAM\adapromptset\manifests\phase2\glas\case_assignments.csv --out-dir D:\SAM\adapromptset\manifests\phase3 --report-dir D:\SAM\adapromptset\reports\phase3_glas\split
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_candidate_pilot.py --project-root D:\SAM\adapromptset --dataset-name GlaS --episodes-csv D:\SAM\adapromptset\manifests\phase3\glas_region_references.csv --out-dir D:\SAM\adapromptset\reports\phase3_glas\region_candidate_calibration --max-episodes 165 --device auto --positive-layout local_peaks --positive-limits 5 10 20 --min-positive-distance 2.0
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_glas_phase3_gate_r1.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_phase3_paper_assets.py --project-root D:\SAM\adapromptset
cd D:\SAM\adapromptset\paper_phase3
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```
