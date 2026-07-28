# Baseline Registry

Date: 2026-07-19

All future tracks must compare fairly against route-appropriate baselines under the same input, reference, budget, and test-exposure conditions.

## Required Families

| Family | Role | Current implementation status |
| --- | --- | --- |
| Fixed Top-1 / Top-K | point prompt baselines | Partial MVP action-bank evidence exists for MoNuSAC; route-specific refresh required. |
| local maxima / local peaks | candidate baseline | Used in GlaS Gate C/Phase 3; frozen for that route. |
| fixed positive/negative | point baseline | MoNuSAC MVP subset exists. |
| KMeans grouping | structured point baseline | Required for future route; current Phase 3 paper marks as not part of R1. |
| GF-SAM-style graph grouping | closest deterministic grouping baseline | Required if point/group route is revisited on fresh protocol. |
| SynPo-style positive/negative | strong medical negative-prompt baseline | Required if negative prompting is claimed. |
| direct matching map | prompt-free baseline | R00 exists for GlaS Phase 3; broader route requires protocol-specific implementation. |
| automatic union box / mask prompt | region baselines | R03/R06/R07/R08 registered; only R03/R04/R05/R00 evaluated in R1 before stop. |
| random spatial/scale controls | leakage/control checks | Size-matched random boxes used in R1; full-image/random-mask controls require Phase 4 audit. |
| alternative-reference control | control boundary | C2 relabeled scientifically in Phase 3. |
| GT point/box/mask upper bounds | privileged upper bound | P4 GT box exists for Phase 2D; must never be automatic method. |
| supervised comparator | upper-bound/different-input condition | nnU-Net may be cited only with clear input-condition boundary. |

No third-party method receives a numeric main-table row unless the same protocol and input conditions can be reproduced or bounded honestly.
