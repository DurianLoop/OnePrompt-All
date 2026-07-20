# Gate E Status

| Check | Status | Notes |
|---|---|---|
| action bank manifest | PASS_RESCUE_SUBSET | name=prompt_set_bank_v2 actions=13 valid=12 invalid=1 |
| shared candidate pool | PASS | actions are built from candidate_pilot_candidates.csv using positive-then-negative index order |
| action cost accounting | PASS | sam_calls equals valid prompt group count; invalid actions cost 0 |
| full baseline matrix | PENDING | KMeans, local-peak, SynPo-style, GF-SAM-style, safe anchors not yet implemented |
