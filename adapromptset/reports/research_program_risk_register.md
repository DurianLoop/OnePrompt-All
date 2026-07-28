# Research Program Risk Register

Date: 2026-07-19

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Validation overfitting from repeated GlaS strict-val inspection | high | Use Phase 4 only read-only; do not add actions/grids on same strict validation. |
| Formal test leakage | high | Require dataset-specific test-freeze docs, frozen configs, and hash reports before first access. |
| Claim expansion from privileged upper bounds | high | Keep GT upper-bound rows separate from automatic methods in all tables and claims. |
| PDF mistaken for submission readiness | high | Enforce publication readiness criteria and mock review before `SUBMISSION_READY`. |
| Route A lacks separability or predictability | medium | Pivot automatically to Route B or C after registered gate failure. |
| Route B becomes "more experiments only" | medium | Require benchmark metrics, taxonomy, diagnostic baseline, and non-trivial cross-domain finding. |
| Route C collapses to threshold+morphology | medium | Require learned or structured mechanism beyond simple thresholding. |
| Insufficient fresh data | high | Use DDR train/valid/test discipline; request user approval before downloads. |
| Compute drift | medium | Log all runs in `reports/research_program_compute_ledger.csv`; pause at budget thresholds. |
| Untracked historical assets confuse provenance | medium | Stage/commit only scoped research-program artifacts; report untracked legacy state when relevant. |
