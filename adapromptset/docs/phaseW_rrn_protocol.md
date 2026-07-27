# Phase W — RRN Protocol

**Date**: 2026-07-27

## Status: ALREADY COMPLETED

RRN grouped 5-fold CV was completed in Phase V. Results at:
- reports/phaseV_rrn_grouped_cv/rrn_grouped_cv_results.json

## Key Results

- AUROC: 0.573 ± 0.099 (range: 0.437-0.712)
- Dice@Top-3 (RRN): 0.446 ± 0.055
- Dice@Top-3 (Similarity): 0.430 ± 0.052
- Dice@Top-3 (Oracle): 0.496 ± 0.056
- Improvement over similarity: +0.015

## Gate Decision: PHASEW_RRN_APPENDIX_ONLY

- AUROC (0.573) < 0.65 threshold for main text
- Dice@Top-3 improvement (+0.015) > +0.010 threshold, but marginal
- High cross-fold variance (AUROC: 0.437-0.712)
- Already in Appendix A of v5 paper