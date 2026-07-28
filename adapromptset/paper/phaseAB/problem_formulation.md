# Problem Formulation

## Reference Reliability Estimation (RRE)

Given a query image $I_q$, a support set of reference image-mask pairs 
$\mathcal{R} = \{(I_r^{(i)}, M_r^{(i)})\}_{i=1}^{N}$, a visual matcher $\Phi$, and a 
segmenter $\Psi$, the multi-reference visual prompting pipeline produces a set of candidate 
segmentations for the query. For each reference $r_i$, the pipeline:

1. Matches $I_q$ to $I_r^{(i)}$ using $\Phi$ to obtain a similarity score and prompt location
2. Segments $I_q$ using $\Psi$ with the prompt from $r_i$, producing $\hat{M}_{q,i}$

The RRE task is to predict the segmentation quality $\text{Dice}(\hat{M}_q, M_q^{\text{GT}})$ 
for the query *without* access to the query ground truth $M_q^{\text{GT}}$.

## Reference-Resampling Evaluation Protocol

To isolate the effect of reference choice, we sample $m$ references from the support set 
and compute $m$ segmentations for each query. The reference-resampling protocol evaluates:

- **Within-query variability**: How much does the segmentation change with different references?
- **Best-worst gap**: The difference between the best and worst Dice among the $m$ segmentations
- **Catastrophic failure rate (CFR)**: The proportion of queries with Dice < 0.1

## Prediction Consistency

For a query $q$ with $m$ references producing segmentations $\hat{M}_{q,1}, \ldots, \hat{M}_{q,m}$, 
the prediction consistency is defined as:

$$C_{\text{pred}}(q) = \frac{2}{m(m-1)} \sum_{i<j} \text{IoU}(\hat{M}_{q,i}, \hat{M}_{q,j})$$

The risk score is the complement:

$$R_{\text{PC}}(q) = 1 - C_{\text{pred}}(q)$$

Higher $R_{\text{PC}}(q)$ indicates lower agreement among references and thus higher 
predicted risk.

## Coverage-Based Selective Prediction

Given a risk ranking of queries, we define a coverage level $c \in (0, 1]$ as the fraction 
of queries for which we accept the prediction. The remaining $(1-c)$ fraction of queries 
(those with highest risk) are flagged for abstention. The accepted set at coverage $c$ is 
the $c$-fraction of queries with lowest risk scores.

## Evaluation Metrics

- **AURC** (Area Under the Risk-Coverage Curve): The area under the curve of Dice vs. 
  coverage. Lower AURC indicates better risk stratification.
- **Excess AURC**: AURC minus the AURC of an oracle ranker (always lower bound).
- **CFR@c**: Catastrophic Failure Rate at coverage $c$—the proportion of accepted queries 
  with Dice < 0.1.
- **Dice@c**: Mean Dice of accepted queries at coverage $c$.
- **AUROC**: Area under the ROC curve for binary classification of catastrophic vs. 
  non-catastrophic queries.
