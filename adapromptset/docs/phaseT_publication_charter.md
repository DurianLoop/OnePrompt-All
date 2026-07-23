# Phase T Publication Charter

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Paper Positioning

Phase T produces a **Benchmark + risk-aware mitigation** paper. The paper is
NOT a SOTA segmentation paper, NOT a method-only paper, and NOT a formal-test
success claim. The primary contribution is a reproducible measurement of
reference-induced instability plus a practical, risk-aware mitigation that
works under the frozen DINOv2-L + SAM-B pipeline.

Recommended paper question (directive §1):

> How much does the choice of a valid reference affect visual-reference
> segmentation, how should this instability be measured, and can a simple
> risk-aware strategy mitigate it without retraining the frozen matcher and
> segmenter?

## Three-Layer Contribution

1. **Phenomenon and Metrics**
   - Reference-induced instability phenomenon.
   - Correct variance decomposition `V_total = V_query + V_ref`.
   - Cross-reference / reference-resampling protocol.
   - Tail-risk metric suite (CVaR, CFR, worst-subset, best-worst gap).

2. **System Benchmark**
   - Multiple datasets (MoNuSAC primary + GlaS / MoNuSeg / DDR secondary).
   - Multiple reference counts k∈{1,3,6}.
   - Multiple prompting / matching baselines (B1–B7 internal families +
     external GF-SAM / SynPo / ProSAM / MAUP / SAMIX / AlignSAM / PPD).
   - Cost accounting and failure-boundary analysis.

3. **Practical Mitigation**
   - Mean consensus as a strong, single-SAM-call baseline.
   - M1 Dynamic Reference Budget (k∈{1,3,6} selection).
   - M2 Instability-Aware Selective Prediction (stable / unstable /
     catastrophic-risk classification + abstention).
   - Lightweight reliability model only if needed; reference weighting is NOT
     the sole method (Track B already showed NO_GAIN).

## Work Packages and Priorities

| WP | Name | Priority | Gate |
|---|---|---|---|
| WP1 | Confirmatory Benchmark (MoNuSAC holdout=40) | MUST run first | §5.6 (10 conditions) |
| WP2 | Cross-Dataset + Cross-Method Replication | After WP1 pass | §6.4 + §7.3 |
| WP3 | Risk-Aware Mitigation (M1 + M2) | After WP2 ≥1 directional support | §9.6 + §10.5 |
| WP4 | Paper Construction + Mock Review | After WP1/WP2/WP3 | §20 (12 conditions) |

## Gate Structure

```
WP1 (holdout confirm) ── pass ──► WP2 (cross-dataset + external)
                                    │
                                    ├── ≥1 extra dataset directional support
                                    │
                                    ▼
                                  WP3 (M1 + M2)
                                    │
                                    ├── validation pass
                                    │
                                    ▼
                                  WP4 (paper + mock review)
                                    │
                                    ├── 12 conditions met
                                    │
                                    ▼
                            PHASET_PAPER_CANDIDATE_READY
```

Routing rules (directive §2):

1. WP1 must complete first.
2. WP2 runs after WP1 passes.
3. WP3 runs after WP2 yields at least one extra-dataset directional support.
4. If WP1 or WP2 fails, formal test is NOT accessed.
5. If WP3 fails, the Benchmark / failure-boundary paper route is retained.
6. Only WP1 + WP2 + WP4 all pass allows writing `PAPER_CANDIDATE_READY`.

## Prohibited Outcomes

The following outcomes are PERMANENTLY PROHIBITED regardless of Gate results:

1. **No SOTA claims.** No leaderboard comparison; formal test is locked.
2. **No formal test access.** Formal test remains locked for the entire
   Phase T program; access requires a separate Phase U freeze protocol and
   explicit user approval (see `docs/phaseT_test_lock_notice.md`).
3. **No result-driven parameter tuning on holdout.** After the holdout hash
   is verified and the run begins, no parameter may be retuned based on
   holdout observations. The holdout runs exactly once per frozen method.

Additional prohibited outcomes (see `docs/phaseT_claim_registry.md`):

- No "reference instability solved" claim.
- No "universal across datasets" claim.
- No "ProSAM/MAUP/SAMIX fully outperformed" claim.
- No "reference weighting is model-agnostic" claim.
- No "mean Dice alone proves robustness" claim.
- No "submission-ready" / "publication-completed" claim until §20 passes.

## Programmatic Constraints

- Total compute cap: **195 GPU-hours**.
- Checkpoints: 40 / 90 / 140 / 195 GPU-hours.
- Statistical unit = query/case; reference episodes are repeated measurements.
- Holdout may be observed exactly once per frozen method.
- Cross-dataset (GlaS / MoNuSeg / DDR) only after WP1 holdout pass.
- No internet downloads without user approval.
- Mandatory reuse of DINOv2 features, matching maps, candidate pools, SAM
  output cache, and Phase R/S manifests.

## Core Principles

1. Strict data-leakage control: train / validation / holdout / test roles
   frozen at preregistration.
2. No validation-set overfitting: hyperparameters selected on train only
   (inner CV where applicable).
3. No post-hoc threshold tuning after Gate evaluation.
4. Honest negative results: a Gate failure is a scientific finding, not a bug.
5. No manufactured success via threshold lowering, holdout peeking, or claim
   inflation.
6. The success criterion is a clear, evidence-backed, submittable paper — not
   a more complex model.
