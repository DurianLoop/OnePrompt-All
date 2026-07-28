# Phase AB Method Freeze

## Primary Method: Prediction Consistency + Coverage-Based Abstention

### Prediction Consistency Risk Score

For query q and m references, each producing a mask M_{q,i}:

C_pred(q) = (2/(m(m-1))) * sum_{i<j} IoU(M_{q,i}, M_{q,j})

R_PC(q) = 1 - C_pred(q)

### Implementation Details

- Reference count: 6 (K=6)
- Reference sampling: random, seed=42
- Matcher: DINOv2-L (frozen)
- Segmenter: SAM-B (primary)
- Binary threshold: 0.5
- Empty-mask IoU: 0.0
- Empty vs non-empty IoU: 0.0
- Computation scale: original image resolution
- Postprocess: none
- Mask fusion: mean consensus
- Catastrophic threshold: Dice < 0.1
- Coverage levels: 50%, 60%, 70%, 80%, 90%, 95%, 100%
- Tie handling: random

### Selective Prediction Protocol

1. Use m references to generate m masks
2. Compute R_PC(q) for each query
3. Rank queries by risk (descending)
4. At coverage C, accept (1-C)*100% lowest-risk queries
5. For accepted queries, use mean-consensus prediction
6. Abstain on remaining queries

### Primary Endpoint

AURC (Area Under Risk-Coverage curve) — lower is better

### Secondary Endpoints

- CFR@80 (catastrophic failure rate at 80% coverage)
- Selective Dice@80
- AUROC (risk ranking quality)

### Confirmatory Comparators

- Random Ranking
- Matching Entropy
- Candidate Jaccard
- Prompt Overlap
- Prediction Consistency (our method)
- M2-Instability (learned baseline)
- M2-Catastrophic (supervised upper bound, diagnostic only)