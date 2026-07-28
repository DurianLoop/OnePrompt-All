# Phase T Risk Register

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Identified Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Holdout may not confirm instability (WP1 fail) | Medium | High | Freeze protocol before run; honest negative-result reporting; Benchmark-only route retained; no formal-test access on fail |
| R2 | Cross-dataset may not replicate (WP2 insufficient directional support) | Medium | Medium | Per-dataset separate judgment; report honestly; at least 2 datasets required for benchmark submission |
| R3 | External methods may not be reproducible (license / code / weights missing) | High | Medium | Comparator tiers (Tier 1–4); Option B (1 external + 3 strong internal families) fallback; Option C lowered positioning |
| R4 | M1 / M2 may not pass validation Gate | Medium | Medium | Two parallel routes (M1 + M2); Benchmark-only route is the fallback if both fail |
| R5 | GPU budget insufficient (195 GPU-hour cap) | Low | High | Checkpoints at 40/90/140/195; mandatory reuse of DINOv2 / SAM / matching caches; CPU-only where possible |
| R6 | Data leakage (train / validation / holdout / test role contamination) | Low (with controls) | Critical | Reuse Phase Y patient-level splits; `scripts/audit_reference_exposure.py` on every manifest; hash verification before holdout |
| R7 | Patient overlap between reference pool and queries | Low (with controls) | Critical | Patient-disjoint reference pool enforced in `docs/phaseT_reference_protocol.md`; reference-query pair constraint audited per episode |

## Risk Monitoring

- After each WP Gate, update this register with realized risks.
- Any risk that materializes triggers a decision-tree re-evaluation.
- Critical risks (R6, R7) trigger immediate program pause if realized.
- R3 is the highest-likelihood risk; the Option B / Option C fallback must be
  kept viable throughout WP2b.

## Risk Realization Log

(Updated as risks materialize during Phase T execution. Empty at
preregistration.)

| Date | Risk ID | Realized? | Action Taken |
|---|---|---|---|
| 2026-07-23 | — | — | Preregistration; no risks realized yet |
