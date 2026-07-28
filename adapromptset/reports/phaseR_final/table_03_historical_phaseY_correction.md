| Metric (Phase Y / R0 audit) | Value | Correction / Interpretation |
|---|---|---|
| Mean best-worst Dice gap (6 refs) | 0.1874 | Reference instability is large at per-query level |
| Mean reference std | 0.0687 | Exceeds R1 cond2 threshold (0.02) |
| % queries best-worst gap >= 0.05 | 82.9% | Exceeds R1 cond1 threshold (30%) |
| Exact action agreement (A vs B) | 0.0488 | Best action is heavily reference-dependent |
| K+ bin agreement (A vs B) | 0.3902 | Coarse action family only mildly stable |
| Median Spearman rho | 0.7450 | Rank correlation moderate but not high |
| Raw oracle gap | 0.0669 | Inflated by Winner's Curse |
| Stable oracle gap | 0.0061 | Small aggregate gap — but NOT absence of instability |
| Optimism fraction | 90.96% | Most of raw gap is selection bias |
| Diff-from-fixed rate | 0.0000 | Oracle picks same action family; instability is in output, not selection |

Source: phaseR_historical_audit/phaseR_hypothesis_summary.md, phaseY_claim_correction.md. R0 was a read-only CPU audit; the small stable oracle gap (0.0061) was an aggregate artifact, not evidence that reference instability is absent.
