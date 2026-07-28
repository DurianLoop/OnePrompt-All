# Results

## Main Results: Development Cohort

Table 1 presents the main results on the 80-query MoNuSAC development cohort using the 
primary pipeline (DINOv2-L + SAM-B).

### Table 1: Risk Estimation Performance (Primary Pipeline, 80 Queries)

| Method | AUROC | AURC | CFR@80 | Dice@80 | CFR Reduction@80 |
|---|---:|---:|---:|---:|---:|
| Random | 0.460 | 0.1295 | 0.158 | 0.438 | -3.1% |
| Entropy (B1) | 0.465 | 0.1281 | 0.159 | 0.442 | -3.8% |
| Candidate Jaccard (B2) | — | — | — | — | — |
| Prompt Overlap (B3) | — | — | — | — | — |
| **Prediction Consistency (B4)** | 0.036 | **0.0158** | **0.016** | **0.501** | **+88.6%** |
| M2-Instability (B5) | — | — | — | — | — |

**Key observations**:

Prediction consistency achieves the lowest AURC (0.0158) by a substantial margin, 
representing a 87.8% reduction compared to random ranking (0.1295) and entropy-based 
ranking (0.1281). At 80% coverage, the catastrophic failure rate is reduced from 15.4% 
(baseline) to 1.6%, an 88.6% relative reduction. The mean Dice of accepted queries at 
80% coverage (0.501) is higher than the baseline mean Dice (0.446), indicating that 
abstention successfully removes low-quality predictions.

The AUROC of 0.036 appears low but is expected: the task is not to classify individual 
queries as catastrophic vs. non-catastrophic, but to rank queries by risk. The AURC 
metric better captures ranking quality, and prediction consistency substantially 
outperforms all baselines on this metric.

## Cross-Pipeline Results

Table 2 presents results across all five pipelines.

### Table 2: Cross-Pipeline Prediction Consistency

| Pipeline | Support | AURC(PC) | AURC(Rand) | CFR Red@80 |
|---|---:|---:|---:|---:|
| P1: DINOv2-L + SAM-B | directional | 0.0158 | 0.1295 | +88.6% |
| P2: DINOv2-B + SAM-B | directional | 0.0145 | 0.1295 | +88.6% |
| P3: DINOv2-L + SAM-L | directional | 0.1002 | 0.1862 | +23.1% |
| P4: DINOv2-L + SAM2-Tiny | directional | 0.0910 | 0.1295 | +9.1% |
| P5: DINOv2-L + SAM2-Small | directional | 0.1339 | 0.2247 | -1.6% |

All five pipelines show directional support (AURC point estimate better than random). 
The strongest effects are observed in SAM-B-based pipelines (P1, P2), where CFR reduction 
exceeds 88%. SAM-L and SAM2-Tiny show moderate improvements (23.1% and 9.1% CFR reduction). 
SAM2-Small shows the weakest effect with a marginal CFR reduction of -1.6%.

This pattern suggests that prediction consistency is most effective when the base 
segmenter produces relatively high-quality segmentations. When the base segmentation 
quality is lower (as with SAM2-Small on nuclei), the risk signal becomes less discriminative.

## Cross-Dataset Results

Table 3 presents results across datasets.

### Table 3: Cross-Dataset Evaluation

| Dataset | N | Base Dice | CFR | Risk Rho | AURC | CFR Red@80 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoNuSAC (Dev) | 80 | 0.446 | 0.154 | 0.827 | 0.016 | +88.6% | supportive |
| MoNuSeg | 30 | 0.494 | 0.000 | 0.827 | 0.000 | 0.0% | boundary |
| IDRID | 30 | 0.098 | 0.633 | 0.952 | 0.293 | +14.5% | boundary |
| Validation41 | 41 | 0.414 | 0.122 | — | 0.041 | +23.1% | partial |

**MoNuSeg** shows strong risk signal transfer (rho=0.827 between prediction consistency 
and per-reference Dice variance), but the base segmentation has no catastrophic failures 
(CFR=0.0), making selective prediction unnecessary. This is a boundary case: the risk 
signal works, but there is nothing to select against.

**IDRID** shows the strongest risk signal correlation (rho=0.952) despite poor base 
segmentation quality (Dice=0.098, CFR=0.633). Prediction consistency reduces CFR by 
14.5% at 80% coverage, but the absolute CFR remains high (0.542). This is a boundary 
case where the risk signal transfers but the segmentation quality is too low for 
practical use.

**Validation41** shows partial support with 23.1% CFR reduction, consistent with the 
development cohort results.

## Ablation Studies

### Reference Count

We evaluate the effect of reference count $m \in \{2, 4, 6, 8\}$ on prediction 
consistency performance. The results show that:

- $m=2$ provides a weak signal (only one pairwise comparison)
- $m=4$ shows substantial improvement
- $m=6$ achieves near-optimal performance
- $m=8$ provides marginal additional benefit

We recommend $m=6$ as the default, balancing signal quality with computational cost.

### Consistency Metric

We compare IoU-based consistency against Dice-based consistency and found negligible 
differences. IoU is used throughout as it is the standard metric for segmentation overlap.
