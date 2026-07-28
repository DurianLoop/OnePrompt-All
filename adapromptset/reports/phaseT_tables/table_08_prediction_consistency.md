# Table 8 — Prediction consistency (WP1 holdout)

Prediction-IoU consistency (J_pred) across resampling seeds and across reference subsets within an episode, by prompt config.

| Config | k | J_pred type | mean J_pred | n |
| --- | --- | --- | --- | --- |
| P20-N1 | 1 | cross_seed_b1_single | 0.5854 | 40 |
| P20-N1 | 6 | cross_seed_consensus_k6 | 0.6438 | 40 |
| P20-N1 | 1 | within_episode_k1 | 0.5734 | 400 |
| P30-N0 | 1 | cross_seed_b1_single | 0.6294 | 40 |
| P30-N0 | 6 | cross_seed_consensus_k6 | 0.7089 | 40 |
| P30-N0 | 1 | within_episode_k1 | 0.6254 | 400 |
| P50-N0 | 1 | cross_seed_b1_single | 0.7320 | 40 |
| P50-N0 | 6 | cross_seed_consensus_k6 | 0.8041 | 40 |
| P50-N0 | 1 | within_episode_k1 | 0.7211 | 400 |

**Notes:**
- J_pred = expected IoU between predictions from two reference subsets.
- cross_seed_b1_single: J_pred across the 10 resampling seeds at k=1 (instability of single-reference choice).
- cross_seed_consensus_k6: J_pred across the 10 seeds at k=6 (consensus stability across seeds).
- within_episode_k1: J_pred across the 6 single references within one episode (per-query reference instability).
- Consensus raises cross-seed J_pred in every config (e.g., P50-N0: 0.7320 → 0.8041). Gate cond8 (consensus J_pred ≥ single J_pred) PASSES.
