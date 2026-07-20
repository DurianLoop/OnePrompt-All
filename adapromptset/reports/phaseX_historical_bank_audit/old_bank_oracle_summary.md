# Phase X Old Bank Oracle Summary

Date: 2026-07-20

## Historical finite-bank oracle (PIVOT_ORACLE_ANALYSIS anchor)

- Bank: `configs/actions/prompt_set_bank_v2.json`
- Bank hash: `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`
- Valid actions: 12
- Episodes: 10
- Strongest fixed action: `B_top5_pos3neg`
- Strongest fixed mean Dice: 0.371238
- Strongest fixed mean IoU: 0.228341
- Strongest fixed mean AJI: 0.228341
- Finite-bank oracle mean Dice: 0.376250
- Finite-bank oracle mean AJI: 0.232161
- Finite-bank Dice gap: 0.005011
- Finite-bank AJI gap: 0.003820
- Median per-episode regret: 0.000319
- Max per-episode regret: 0.018556
- Near-oracle episode fraction (tol=0.005): 0.6
- Top oracle action fraction: 0.5
- Oracle action distribution: {"A_top5_one_group": 1, "B_top3_pos3neg": 1, "B_top5_pos1neg": 2, "B_top5_pos3neg": 5, "R_split5_2groups_localneg": 1}
- Historical Phase 5 status: `PIVOT_ORACLE_ANALYSIS`

## Top-K absolute Dice

- Top-1 mean Dice: 0.029210558137448394
- Top-3 mean Dice: 0.2560657753898481
- Top-5 mean Dice: 0.3259841511612571
- Top-10 status: `INVALID_REQUIRES_10_POSITIVES`

## Bank coverage limits

- Max K+ tested: 5
- Max K- tested: 3
- Max SAM calls tested: 5
- K+ in {10, 20, 30, 50, 75, 100}: never tested
- K- in {10, 20, 30, 50}: never tested
- Groupings tested: G_ONE_GROUP, G_ONE_POINT_PER_GROUP, G_FIXED_CHUNK_3_2
- Groupings *not* tested: G_KMEANS_*, G_SPATIAL_GRID, G_GRAPH, G_CANDIDATE_COMPONENT
- Call budgets *not* tested: 10, 20

## Conclusion

The historical `0.005` gap is **not** a saturation proof. It is the gap between
two low-absolute-performance configurations (mean Dice ~0.37) inside a narrow
12-action bank. It cannot rule out adaptive value under an expanded action
space.

Renamed historical claim: `FINITE_BANK_ORACLE_GAP_SMALL`.
