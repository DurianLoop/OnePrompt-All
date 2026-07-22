# Phase S Publication Readiness Criteria

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Method Paper Candidate

Must satisfy ALL:
- New method (not a re-implementation of existing work)
- Validation + holdout pass
- At least one cross-dataset direction consistent
- Strong baseline comparison (B1-B6 + Track D baselines if applicable)
- Stability metric reported (within-query variance, CVaR, CFR)
- Failure analysis included
- Ablation study included (§16)
- Cost / runtime reported
- Novelty audit passed (§13)

## Benchmark Paper Candidate

Must satisfy ALL:
- Reference instability phenomenon stable across ≥2 datasets
- New metric or protocol introduced
- At least 2-3 datasets included
- Multiple prompt/matching methods compared
- Failure taxonomy included
- At least one predictive baseline
- Reproducible manifests

## Hybrid Paper Candidate (Preferred)

> New reference-instability benchmark + deterministic baseline + learned
> reliability method + cross-dataset analysis.

Must satisfy the union of method and benchmark criteria.

## Failure-Boundary Paper Candidate

If all method tracks fail but Track A benchmark is valid:
- Clear negative result with mechanistic explanation
- Generalizable failure boundary (what does NOT work and why)
- Stable phenomenon (instability is real even if not mitigable)
- Cost of failure quantified

## Mock Review Checklist

Before declaring any paper readiness:
1. Novelty audit (§13) — differentiation from GF-SAM, SynPo, AlignSAM, Point
   Prompt Defender, ProSAM, MAUP, SAMIX, few-shot segmentation, visual
   reference segmentation, selective prediction, robust aggregation, set
   learning.
2. Statistical rigor — bootstrap CI, Holm correction, query-level units.
3. Reproducibility — one-command reproduction, frozen configs, commit hashes.
4. Claim boundary — no prohibited tokens; claims match Gate evidence.
5. Cost transparency — GPU-hours, SAM calls, runtime.
6. Limitation disclosure — what is NOT claimed.

## Publication Readiness States

- `PHASES_METHOD_PAPER_READY`
- `PHASES_BENCHMARK_PAPER_READY`
- `PHASES_HYBRID_PAPER_READY`
- (implicit) `PHASES_NEGATIVE_RESULT_PAPER_READY` (failure-boundary paper)
- `PHASES_RESEARCH_PROGRAM_EXHAUSTED` (no publishable direction found)
