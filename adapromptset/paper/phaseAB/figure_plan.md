# Figure Plan

## Figure 1: Risk-Coverage Curves (Primary Pipeline)
- X-axis: Coverage (0.5 to 1.0)
- Y-axis: Catastrophic Failure Rate (CFR)
- Lines: Prediction Consistency, Random, Entropy, Candidate Jaccard, Prompt Overlap
- Show: Prediction consistency substantially below all baselines

## Figure 2: Method Schematic
- Flow diagram: Query + References → Segmentations → Pairwise IoU → Consistency Score → Risk Ranking → Abstention
- Highlight: No additional model calls, no GT needed

## Figure 3: Cross-Pipeline Summary
- Bar chart: AURC and CFR@80 reduction across 5 pipelines
- Color by support level (strong/directional/none)

## Figure 4: Cross-Dataset Boundary Analysis
- Scatter: Prediction consistency vs. per-reference Dice variance
- Points colored by dataset
- Show: Strong correlation (rho) for each dataset
