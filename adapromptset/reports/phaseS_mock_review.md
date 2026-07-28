# Phase S — Mock Peer Review

Date: 2026-07-22
Paper type: Benchmark + failure-boundary paper (per directive §21)
Suggested title: "Reference-Induced Instability in Visual-Reference Segmentation: A Benchmark and Failure-Boundary Analysis"

## Summary

This paper investigates reference-induced instability in visual-reference segmentation using a frozen DINOv2-L matcher and SAM-B segmenter on MoNuSAC (41 validation queries). The authors correct a metric definition error in prior work (Phase R cond3, which mixed cross-query and within-query variance axes), establish a reference-instability benchmark with tail-risk metrics, and systematically test three method tracks: (B) learned reference reliability with GBM-based weighting, (C) set-conditioned prompting with pixel-wise agreement gating and stable candidate scoring, and (D) direct reference-conditioned segmentation (not started due to unmet start conditions). All method tracks fail to improve over the B2_mean baseline (val Dice 0.4756), leading to a benchmark + failure-boundary paper. The key positive findings are: (1) reference instability is real (V_ref_frac = 0.1376–0.3312), (2) it is predictable from test-time-safe features (Spearman = 0.8832 on train, 0.4781 on validation), but (3) it cannot be mitigated through reference weighting or set conditioning. The bottleneck is identified at the SAM point-prompt interface.

## Strengths

1. **Rigorous metric correction.** The identification of Phase R cond3's axis-mixing error (var_single on query×ref axis = V_query+V_ref; var_consensus on query axis = V_query) is a genuine methodological contribution. The corrected V_ref decomposition cleanly separates case difficulty from reference-induced variance — a distinction prior visual-reference segmentation work has not made explicit.

2. **Preregistered multi-track protocol.** The 4-track design (A/B/C/D) with explicit gates, a 210 GPU-hour cap, frozen data splits, a claim registry with allowed/prohibited claims, and a mandatory novelty audit + mock review is exemplary research hygiene. The claim registry (10 prohibited tokens including "SOTA", "solved", "universal") is a novel self-regulation mechanism.

3. **Systematic negative results.** The honest documentation of Track B failure (delta Dice = -0.0024 despite Spearman = 0.478 utility prediction) and Track C failure (all 26 non-zero hyperparameter combinations degrade Dice) is valuable. The finding that "predictability does not imply deployability" is a non-trivial negative result for the reference-selection paradigm.

4. **Tail-risk metric suite.** The inclusion of CVaR_0.1, catastrophic failure rate (Dice<0.1), worst-subset Dice, and worst-reference gap — not just mean Dice — provides a more complete picture of reference instability than prior work.

5. **Failure-boundary identification.** The prompt-interface bottleneck finding (Track D start condition 3 not met because SAM refinement *helps*: consensus Dice 0.48 vs raw matching map <0.3) reframes the problem: the instability is not in the matcher but in how SAM processes point prompts.

## Weaknesses

1. **Single dataset.** Only MoNuSAC is tested. Directive §21 requires ≥2–3 datasets for a benchmark paper. GlaS and DDR are mentioned as locked/blocked but not evaluated. The benchmark's generalizability is unverified.

2. **No holdout validation.** Holdout=40 is never used because no method track passed. While this is protocol-compliant (directive §11 requires a method PASS before holdout), it means the benchmark claims rest entirely on the validation set (n=41). A reviewer may ask: is the instability an artifact of this particular split?

3. **CPU-only scale.** 0 GPU-hours used; 6779 SAM calls on CPU. While the relative comparisons hold, the absolute Dice numbers (0.4756) are low compared to GPU-run SAM benchmarks. The paper should clarify that this is a *relative* benchmark, not a SOTA claim (which is prohibited anyway).

4. **No head-to-head with related methods.** ProSAM, MAUP, SAMIX, GF-SAM, SynPo are discussed in the novelty audit but not run under the same protocol. The novelty audit is conceptual, not empirical. A reviewer may ask: would ProSAM's progressive prompting reduce the instability you measure?

5. **Track C limited to C3/C4.** C1 (DeepSets) and C2 (Set Attention) were not implemented because C3/C4 showed no signal. While this is efficient (directive §8.3: complexity limit), it leaves open the possibility that higher-capacity set models could help — though the train-val gap (0.037) suggests overfitting risk.

## Questions for Authors

1. How would the V_ref decomposition change on a dataset with higher inter-query difficulty variance (e.g., GlaS with smaller lesions)?
2. The Track B utility prediction Spearman drops from 0.916 (train) to 0.478 (validation). Is this generalization gap a feature-distribution shift, or overfitting? Have you examined feature stability?
3. Track C's C3 (pixel-wise agreement gate) with α=0 reduces to B2_mean. What does the monotonic Dice degradation with increasing α tell us about the structure of the disagreement signal?
4. The prompt-interface bottleneck finding rests on "raw matching map Dice <0.3 vs consensus Dice 0.48". Could you provide per-query evidence that SAM refinement consistently helps, rather than just on average?
5. The mock review scores this paper Weak Reject. What is the minimal additional experiment (within the 210 GPU-hour cap) that would move it to Accept?

## Soundness: 6/10

The metric correction and benchmark are sound. The negative results are well-supported. However, the single-dataset limitation and absence of holdout reduce confidence in the generality of the claims. The CPU-only execution is acceptable for a relative benchmark but limits absolute performance claims.

## Presentation: 7/10

The 20 tables and preregistered protocol are well-organized. The claim registry is a notable presentation innovation. However, the 24 required figures (directive §15) are not generated in this deliverable, and the final research summary could benefit from a clearer narrative arc connecting Phase X→Y→R→S.

## Contribution: 6/10

The metric correction (V_ref decomposition) is a genuine methodological contribution. The negative result (predictability ≠ deployability) is valuable. However, the single-dataset scope and absence of any successful method limit the paper's impact. As a benchmark + failure-boundary paper, it is borderline — the boundary is well-documented but not yet validated across datasets.

## Recommendation: Weak Reject / Borderline

The paper has a sound core (metric correction + honest negative results) but does not meet the §21 benchmark-paper bar (≥2–3 datasets) in its current form. **Resubmit with cross-dataset validation (GlaS or DDR) to confirm the reference-instability phenomenon generalizes.** If the cross-dataset V_ref_frac ≥ 0.10 and best-worst gap ≥0.05 on ≥30% queries, the benchmark claim strengthens to Accept-level.

The failure-boundary finding (prompt-interface bottleneck) is the most novel contribution and should be emphasized in the resubmission.
