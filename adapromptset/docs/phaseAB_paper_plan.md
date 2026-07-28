# Phase AB Paper Plan

## Target Venue

Venue-neutral (initially CVPR/ICCV/NeurIPS level, but must acknowledge exploratory nature)

## Paper Type

New Task Formulation + Simple Method + Systematic Evaluation

## Title

**Primary**: When References Disagree: Prediction Consistency for Selective Visual Prompting

**Alternative**: Prediction Consistency as a Simple Risk Signal for Multi-Reference Visual Prompting

## Contributions (3)

1. **RRE Problem Definition**: Reference Reliability Estimation with reference-resampling evaluation protocol
2. **Simple Risk Signal**: Prediction Consistency — GT-free, no additional training, query-level risk ranking
3. **Systematic Evaluation**: Coverage-risk trade-off across 5 pipelines and 3 datasets, with honest boundary analysis

## Paper Structure

1. Introduction (problem, motivation, contributions)
2. Related Work (visual prompting, selective prediction, uncertainty estimation)
3. Problem Formulation (RRE, reference resampling, risk metrics)
4. Method (Prediction Consistency, coverage-based abstention)
5. Experiments (setup, main results, ablation, cross-pipeline, cross-dataset)
6. Failure Analysis (when and why it fails)
7. Limitations
8. Conclusion

## Key Figures

1. Risk-coverage curves (primary pipeline, all baselines)
2. Cross-pipeline summary (AURC, CFR@80 across 5 pipelines)
3. Cross-dataset boundary analysis
4. Method schematic (references → masks → consistency → risk → abstention)

## Key Tables

1. Main results (Development Cohort, all baselines)
2. Cross-pipeline (AURC, CFR@80, support status)
3. Cross-dataset (segmentation quality, risk ranking, selective prediction)
4. Ablation (reference count, consistency metric)

## Appendix

- RRN results (AUROC=0.573)
- Anti-correlation as mechanical artefact (brief note)
- Full RAMP component ablation