# Phase AB Submission Readiness Assessment

**Date**: 2026-07-28

## Overall Assessment: PHASEAB_PAPER_CANDIDATE_READY (with caveats)

## Gate Criteria

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Integrity closure pass | PASS | WP0 resolved all null model conflicts |
| 2 | Fresh confirmatory cohort | FAIL | No fresh cohort available |
| 3 | Power analysis | PASS | N=75 recommended, current N=80 |
| 4 | Confirmatory pass | N/A | Blocked by #2 |
| 5 | >=3/5 pipelines support | PASS | 5/5 directional support |
| 6 | >=1 external dataset support | PARTIAL | MoNuSeg boundary, IDRID boundary |
| 7 | >=1 boundary dataset | PASS | Both MoNuSeg and IDRID are boundary |
| 8 | Baseline provenance pass | PASS | WP4 passed |
| 9 | All prohibited claims removed | PASS | 15 claims retired |
| 10 | Full RAMP removed from main | PASS | In appendix only |
| 11 | RRN in Appendix only | PASS | AUROC=0.573 |
| 12 | Risk-coverage curves complete | PASS | All coverage levels |
| 13 | AURC as primary endpoint | PASS | Frozen in config |
| 14 | Venue-neutral full draft | PASS | 17 paper files generated |
| 15 | Mock review no fatal blockers | PASS | All concerns addressable |

## Recommended Venue Strategy

Given the exploratory nature and lack of confirmatory cohort, the paper is best suited for:

1. **Workshop venues** (e.g., CVPR workshops, MICCAI workshops) — most appropriate
2. **Benchmark tracks** (e.g., NeurIPS Datasets & Benchmarks) — good fit
3. **Lower-tier conferences** (e.g., BMVC, WACV) — if strong rebuttal is prepared
4. **Not recommended**: CVPR/ICCV/NeurIPS main conference — requires confirmatory evidence

## Strengths
- Clear problem formulation (RRE)
- Simple, practical method
- Honest about limitations
- Systematic evaluation across pipelines and datasets
- Well-documented reproducibility

## Weaknesses
- No confirmatory cohort
- Limited to medical imaging domain
- Method is a heuristic (no learning)
- Cross-dataset results are boundary cases
- Small evaluation scale (80 queries)

## Recommended Actions Before Submission
1. Add at least one more external dataset (if available without new downloads)
2. Run a simple learned baseline as comparison (e.g., logistic regression on consistency features)
3. Add qualitative examples (high vs. low consistency queries)
4. Compute full risk-coverage curves for all methods
5. Prepare a strong rebuttal addressing the 7 reviewer concerns
