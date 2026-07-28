# Experimental Protocol

## Datasets

### Development Cohort: MoNuSAC (80 queries)
The primary development cohort consists of 80 nuclei segmentation queries from the MoNuSAC 
dataset. These queries are drawn from distinct patients/cases not used in the reference pool. 
The cohort was used for method development, threshold selection, and comparative analysis. 
**Important**: This is a development cohort, not a holdout set. All results on this cohort 
should be interpreted as exploratory.

### External Validation Datasets
- **MoNuSeg** (30 queries): Multi-organ nuclei segmentation—same task domain
- **IDRID** (30 queries): Retinal lesion segmentation—different task domain
- **Validation41** (41 queries): Separate MoNuSAC queries used for validation

## Pipelines

Five pipelines are evaluated:
- **P1**: DINOv2-L + SAM-B (primary)
- **P2**: DINOv2-B + SAM-B
- **P3**: DINOv2-L + SAM-L
- **P4**: DINOv2-L + SAM2-Tiny
- **P5**: DINOv2-L + SAM2-Small

## Configuration

- Reference count: 6 per query
- Reference sampling: random, seed 42
- Patient/query disjoint: enforced
- 5 reference seeds per dataset for cross-dataset evaluation
- Query-level aggregation for all metrics

## Statistical Analysis

- Query is the statistical unit
- Bootstrap confidence intervals (10,000 iterations) for paired comparisons
- Holm correction for multiple comparisons across pipelines
- No data leakage between development and external validation
