# Dense Prompt Scaling Protocol (Phase X, Section 8)

Date: 2026-07-20

## Positive selection strategies (section 8.1)

P_TOPK, P_LOCAL_PEAKS, P_SPATIAL_NMS, P_SPREAD, P_COMPONENT_BALANCED, P_DIVERSITY_GREEDY

Definitions:

- `P_TOPK`: take top-K candidates by contrast score.
- `P_LOCAL_PEAKS`: take local maxima of the contrast score map.
- `P_SPATIAL_NMS`: spatial non-maximum suppression on top candidates.
- `P_SPREAD`: maximize spatial dispersion among high-score candidates.
- `P_COMPONENT_BALANCED`: balance selection across candidate spatial
  connected components.
- `P_DIVERSITY_GREEDY`: greedily pick the highest-scoring candidate least
  redundant with already-selected points.

Query GT must *not* be used for positive selection.

## Negative selection strategies (section 8.2)

N_NONE, N_BOTTOMK, N_HIGH_BACKGROUND, N_BOUNDARY_HARD, N_LOCAL_CONFUSION, N_DIVERSITY

## Positive count grid (section 8.3)

1, 3, 5, 10, 20, 30, 50, 75, 100

If the candidate pool has fewer than the requested K, use the actual maximum,
record the shortfall, never duplicate points, never insert fake points.

## Negative count grid (section 8.4)

0, 1, 3, 5, 10, 20, 30, 50

## Grouping families (section 8.5)

G_ONE_GROUP, G_ONE_POINT_PER_GROUP, G_FIXED_CHUNK_5, G_FIXED_CHUNK_10, G_FIXED_CHUNK_20, G_KMEANS_3, G_KMEANS_5, G_KMEANS_10, G_KMEANS_20, G_SPATIAL_GRID, G_GRAPH, G_CANDIDATE_COMPONENT

Definitions follow section 8.5: `G_ONE_GROUP` puts all points in one SAM call;
`G_ONE_POINT_PER_GROUP` makes one SAM call per point; `G_FIXED_CHUNK_n` splits
into chunks of n; `G_KMEANS_k` clusters spatially into k groups;
`G_SPATIAL_GRID` groups by a fixed grid; `G_GRAPH` groups by graph relation;
`G_CANDIDATE_COMPONENT` groups by candidate-pool spatial connected components.

## SAM-call budgets (section 8.6)

1, 3, 5, 10, 20

## Total point budgets (section 8.7)

5, 10, 20, 50, 100, 150

## Staged search (section 9)

Stage S1: single-variable K+ scaling
- selection = `P_TOPK`, negative = `N_NONE`, grouping = `G_ONE_GROUP`, calls = 1
- vary K+ only.

Stage S2: grouping scaling
- pick K+ in {10, 30, 50, 100} from S1; sweep grouping family.

Stage S3: negative scaling
- fix S2's best positive/grouping; sweep K- and negative strategy.

Stage S4: joint coarse grid
- 3 positive selectors x 3 K+ x 3 negative settings x 4 groupings x 3 call
  budgets.

Stage S5: successive halving
- all actions run on a small train-only audit subset; eliminate actions that
  are clearly dominated on Dice, component recall, merge rate, SAM calls, and
  runtime; survivors enter full validation. Formal test must not influence
  elimination.

## Fairness rules

Different methods must be compared under matched call budget, matched point
budget, or on an explicit Pareto curve. Per section 16, every action must
report requested vs actual valid positive/negative counts, group count, SAM
calls, wall-clock runtime, VRAM, failed calls, and fusion cost.
