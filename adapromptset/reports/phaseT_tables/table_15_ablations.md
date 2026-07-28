# Table 15 — Ablations

Ablation studies on prompt config, reference count k, M1 policy class, and M2 selective-prediction model class.

| Ablation axis | Setting | Effect on Dice / instability | Takeaway |
| --- | --- | --- | --- |
| Prompt config (k=1, WP1 holdout) | P20-N1 | V_ref_frac = 0.2627 (highest) | Adding 1 negative point (P20-N1) raises reference instability. |
| Prompt config (k=1, WP1 holdout) | P30-N0 | V_ref_frac = 0.2110 | More positive points without negatives has intermediate instability. |
| Prompt config (k=1, WP1 holdout) | P50-N0 | V_ref_frac = 0.1264 (lowest) | Denser positive sampling reduces reference-induced instability. |
| Reference count k (P30-N0, WP1 holdout) | k = 1 | V_ref_frac = 0.2110 | Single-reference sampling is the most unstable setting. |
| Reference count k (P30-N0, WP1 holdout) | k = 3 | V_ref_frac = 0.1843 | Consensus over 3 references already halves V_ref_frac. |
| Reference count k (P30-N0, WP1 holdout) | k = 6 | V_ref_frac = 0.1590 | Full 6-reference consensus reaches the lowest V_ref_frac. |
| M1 policy class (λ_c=λ_r=0) | baseline k=6 → 0.4586 Dice | Reference (no policy) | Always-k=6 is the strongest baseline; policy must beat it on cost or on Dice at lower cost. |
| M1 policy class (λ_c=λ_r=0) | threshold → 0.4571 Dice (avg_k = 5.63) | Dice ≈ k=6 but cost ≈ k=6 (no saving) | Handcrafted threshold does not unlock a useful cost-quality trade-off. |
| M1 policy class (λ_c=λ_r=0) | logistic → 0.4552 Dice (avg_k = 5.88) | Slightly below k=6, cost still near k=6 | Linear model cannot identify a low-risk low-k subset. |
| M1 policy class (λ_c=λ_r=0) | gradient_boosting → 0.4516 Dice (avg_k = 4.98) | Lower Dice, lower avg_k — dominated by k=6 | Nonlinear model trades Dice for cost but misses the cs4 25% target. |
| M2 model class (catastrophic-trained) | M2-0/M2-1 threshold | AUPRC_cat = 0.0831 | Threshold baselines tie on metrics — their features collapse to the same ranking. |
| M2 model class (catastrophic-trained) | M2-2 logistic regression | AUPRC_cat = 0.4338 (best) | Linear model is the best catastrophic-risk detector; selected for holdout. |
| M2 model class (catastrophic-trained) | M2-3 gradient boosting | AUPRC_cat = 0.3997 | Nonlinear model is competitive but slightly below logistic. |
| M2 model class (catastrophic-trained) | M2-4 small MLP | AUPRC_cat = 0.4167 | Small MLP matches logistic; not selected to keep model complexity minimal. |

**Notes:**
- Three ablation axes are reported: (a) prompt config, (b) reference count k, (c) model class within M1 and M2.
- Prompt-config and k ablations are run on the WP1 holdout (frozen protocol); M1/M2 model-class ablations are run on the WP3 validation split.
- M1 ablation uses λ_c = λ_r = 0 to isolate the model-class effect from the utility-weighting effect; the selected λ grid in §9.5 did not produce any non-zero λ that beat λ=0, so the policy collapses to model-class choice.
- M2 ablation shows logistic regression is sufficient — adding capacity (GBM, MLP) does not improve catastrophic AUPRC by a meaningful margin.
