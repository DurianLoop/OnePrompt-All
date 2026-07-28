# AdaPromptSet Phase 3 Gate Status

Date: 2026-07-19
Branch: `phase3-glas-region-prompts`
Protocol: `adapromptset.phase3.glas_region_prompt.v1`

| Gate | Status | Notes |
| --- | --- | --- |
| Phase 2D historical state | `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED` | Preserved as the operational state under P3 grouped-point viability. |
| Interpretation audit | `PHASE3_INTERPRETATION_AUDIT_COMPLETE` | CPU-only audit of existing Phase 2D artifacts completed; no new SAM inference. |
| Scientific pivot | `PIVOT_GLAS_REGION_PROMPT_REQUIRED` | Grouped sparse point prompting failed; privileged box upper bound indicates region interface remains viable. |
| Phase 3 40/15 split | `PASS_PHASE3_GLAS_REGION_40_15_SPLIT` | 40 calibration, 15 internal selection, 165 reference episodes, no testA/testB access. |
| Region bank v1 | `PREREGISTERED` | R00-R09 automatic actions and U01-U03 privileged upper bounds frozen before new SAM inference. |
| Gate R1 proposal quality | `STOP_GLAS_REGION_PROPOSAL_FAILURE` | No automatic region proposal family met the preregistered R1 box or mask route thresholds. |
| Gate R2 automatic region prompt viability | `BLOCKED_BY_R1_FAILURE` | SAM-B region prompt inference is not allowed because R1 failed. |
| Gate R3 region action oracle | `BLOCKED_BY_R1_FAILURE` | R3 requires R2 pass, which cannot be reached. |
| Prompt-type router | `PROHIBITED_BY_R1_FAILURE` | No region action utility matrix or router training is allowed. |
| Formal testA/testB | `PROHIBITED_BY_R1_FAILURE` | testA/testB remain locked and unused. |

Current Phase 3 final state:

`STOP_GLAS_REGION_PROPOSAL_FAILURE`

## Gate R1 Evidence

| Family | Route | Metric | Improvement | Win rate | Recall | Empty | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R00 direct matching mask | mask | `0.178754` | `-0.416321` | `0.066667` | `0.099628` | `0.000000` | `FAIL` |
| R03 automatic union box | box | `0.744785` | `0.025965` | `0.200000` | `0.790664` | `0.000000` | `FAIL` |
| R04 largest-component box | box | `0.019410` | `0.002199` | `0.100000` | `0.029807` | `0.000000` | `FAIL` |
| R05 top-3 component boxes | box | `0.271747` | `0.016568` | `0.166667` | `0.063224` | `0.000000` | `FAIL` |

R03 achieved high absolute box IoU and foreground recall, but it failed R1 because its improvement over the size-matched random box was only `0.025965` and win rate was only `0.200000`, below the required `0.10` improvement and `0.70` win rate.
