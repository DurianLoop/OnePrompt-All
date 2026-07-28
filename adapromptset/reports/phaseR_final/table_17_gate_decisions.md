| Gate | Decision / Status | Result | Reason |
|---|---|---|---|
| R1 | PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED | PASSED | 5/5 conditions passed |
| R2 | PHASER_GATE_R2_CONSENSUS_NO_GAIN | FAILED | 7/8 conditions; cond3 variance_reduction_ge_25pct FAILED (-5.06%) |
| R3 | BLOCKED | BLOCKED | R2 failed; directive section 12 |
| R4 | BLOCKED | BLOCKED | R2 and R3 both failed; directive section 13 |
| GlaS | BLOCKED | BLOCKED | R2 failed; directive section 14 |
| Formal test | LOCKED_UNUSED | LOCKED | Directive section 22: no test access |

Final Phase R state: PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC
