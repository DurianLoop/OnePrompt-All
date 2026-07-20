# Table 18: Final Gate Decision Table

| Gate | Name | Status | Rationale |
|---|---|---|---|
| X1 | Historical bank audit | HISTORICAL_AUDIT_COMPLETE | Old 12-action bank; strongest fixed = B_top5_pos3neg (Dice=0.3712) |
| X2 | Candidate universe adequacy | CANDIDATE_UNIVERSE_INADEQUATE | even at K+=100, component coverage is 0.433 < 0.5; problem is matcher/candidate generator, not policy |
| X3 | Dense prompt viability | DENSE_PROMPT_VIABILITY_ESTABLISHED | K=5 -> K=100 Dice gain = 0.097, component recall gain = 0.108; dense prompts improve both Dice and recall |
| X4 | Old action bank adequacy | OLD_ACTION_BANK_INADEQUATE | A_large best_fixed=0.5411 vs A_small best_fixed=0.3713; gain=0.1699 >= 0.05 |
| X5 | Search oracle exceeds bank oracle | SEARCH_ORACLE_EXCEEDS_BANK_ORACLE | search=0.7256 - bank=0.6119 = 0.1137 >= 0.01 |
| X6 | Reopen policy learning with expanded action space | REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE | All 6 conditions met: X1 historical audit complete, X2 candidate universe inadequate, X3 dense prompt viability established, X4 old action bank inadequate, X5 search oracle exceeds bank oracle, X6 actionable expansion direction identified. |
