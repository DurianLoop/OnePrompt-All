# Formal Test Lock Registry

Date: 2026-07-19

| Formal test | Status | Required before first access | Prohibited after first access |
| --- | --- | --- | --- |
| GlaS testA | `LOCKED_UNUSED` | `docs/test_freeze_glas.md`, frozen final config, hashes, code commit, metrics, qualitative sampling, cost/statistics plan | Threshold/action/model/router/seed/postprocess changes. |
| GlaS testB | `LOCKED_UNUSED` | testA report complete with no method changes; same frozen method | Treating testB as rescue validation. |
| DDR test | `LOCKED_UNUSED_CANDIDATE_FRESH_DOMAIN` | DDR four-intent protocol freeze, train/valid-only selection, frozen final method, hashes | Lesion-class tuning or threshold changes after test. |
| MoNuSeg formal use | `NOT_FORMAL_BY_DEFAULT` | re-audit official split/patient identity and define role | Using as sole proof of submission readiness. |
| IDRID | `READ_ONLY_NON_TUNABLE` | proof of unobserved subset if any formal claim is desired | Any tuning or new method-selection influence. |

No formal test may be used to decide between Routes A/B/C.

## 2026-07-19 DDR Lock Check

Route B DDR development manifests were generated only for `train` and `valid`.
The builder records `formal_test_access=false`, `test_split_requested=false`, and `test_files_read=false`.
DDR test remains unavailable until `docs/test_freeze_ddr.md`, frozen final config, code commit, metrics, statistics, qualitative sampling, and cost accounting are complete.
