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
| Gate R1 proposal quality | `NOT_RUN` | Must run before any SAM region prompt inference. |
| Gate R2 automatic region prompt viability | `BLOCKED_UNTIL_R1_PASS` | SAM-B only unless preregistered replication criteria are met. |
| Gate R3 region action oracle | `BLOCKED_UNTIL_R2_PASS` | Uses only frozen automatic actions. |
| Prompt-type router | `BLOCKED_UNTIL_R3_CONTINUE` | PPO/DQN prohibited. |
| Formal testA/testB | `PROHIBITED_UNTIL_FINAL_METHOD_FREEZE` | testA/testB remain locked and unused. |

Current Phase 3 state:

`PHASE3_INTERPRETATION_AUDIT_COMPLETE`

Next allowed action after preregistration commit:

Run internal parameter calibration and Gate R1 proposal-quality evaluation without changing the frozen bank or grid.
