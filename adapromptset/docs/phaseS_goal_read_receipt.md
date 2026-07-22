# Phase S Goal Read Receipt

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`
Directive: `D:\SAM\adapromptset\research.md`

## Acknowledgement

CX acknowledges receipt of the Phase S directive and the `/goal` mode execution
mandate. Phase S is a multi-track reference-robust segmentation research
program with four registered tracks (A/B/C/D), a 210 GPU-hour cap, and a strict
preregistration requirement.

## Starting Scientific State (from Phase X/Y/R)

```text
Reference instability confirmed
+ deterministic consensus utility gain confirmed
+ stability reduction unresolved
+ learned reliability scientifically motivated
```

Phase R is NOT treated as a failed direction; it established that:
- Reference instability is real (88.75% queries best-worst gap ≥ 0.05).
- Deterministic consensus improves mean Dice (+0.0460) and catastrophic rate.
- Variance reduction is unresolved (consensus increased variance by 5.06%).

## Four Registered Tracks

1. **Track A** — Metric Correction and Stability Benchmark (must run first)
2. **Track B** — Learned Reference Reliability and Robust Aggregation (primary method route)
3. **Track C** — Set-Conditioned Robust Prompting (only if B has signal and plateaus)
4. **Track D** — Direct Reference-Conditioned Segmentation (fallback if prompt interface fails)

## Execution Order (section 24)

1. Read Phase X/Y/R artifacts ✓
2. Create Phase S branch ✓
3. Create multi-direction registry (this commit)
4. Complete metric correction audit (Track A §6.3)
5. Execute Track A benchmark (§6.4) and Gate (§6.5)
6. Track Selection (§10)
7. Train Track B reliability models if signal exists
8. Decide Track C after B plateaus
9. Run Track D if B/C fail but matching evidence has value
10. Freeze holdout after validation pass
11. Run holdout once
12. Cross-dataset after holdout pass
13. Generate tables, figures, ablations, failure analysis
14. Novelty audit
15. Mock review
16. Final research summary
17. Output unique state

## Principle

Per directive section 24:
- Do NOT stop all research just because one Gate fails.
- Do NOT immediately claim publishable just because of small positive results.
- The real goal: find a scientifically clear, well-evidenced, differentiated
  research direction that can form a paper; if none exists, document why.
