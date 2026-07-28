# Table 14: Runtime/VRAM/Calls (Compute Summary)

| Stage | Mean Runtime (ms/episode) | Mean SAM Calls | VRAM | Notes |
|---|---|---|---|---|
| S2 single-variable K+ (K=20) | 26.1 | 1 | SAM-B vit_b (~360 MB) | 1 SAM call per episode |
| S3 negative scaling (K- best) | 42.4 | 1.00 | SAM-B vit_b (~360 MB) | 1 SAM call per episode |
| Grouping scaling (G_ONE_GROUP, K=20) | 51.0 | 1.00 | SAM-B vit_b (~360 MB) | Lowest cost grouping at peak Dice |
| S6 nested bank (A_large best_fixed) | 43.3 | 1.00 | SAM-B vit_b (~360 MB) | 91 actions evaluated; 1 deployed call |
| Search oracle (Gate X5) | N/A (search loop) | 558.5 | SAM-B vit_b (~360 MB) | Search budget ~558 SAM calls; 1 deployed call |
