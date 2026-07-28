# Phase AB Rebuttal Plan

## R1: "The Method Is Too Simple"

**Response Strategy**: Embrace simplicity as a strength.

**Key Points**:
- Simplicity is not a weakness—it is the primary contribution. The method requires no 
  training, no model modification, and no additional forward passes. This makes it 
  immediately deployable.
- The intellectual novelty is in the problem formulation (RRE) and the systematic 
  evaluation protocol, not in the complexity of the solution.
- We explicitly position this as a benchmark/exploratory paper. The goal is to establish 
  a baseline for future, more sophisticated methods.

**Evidence to Add**:
- Cite the "bitter lesson" and recent work showing that simple baselines often outperform 
  complex methods
- Show that more complex methods (Full RAMP, RRN) do not outperform prediction consistency

---

## R2: "No Real New Model"

**Response Strategy**: Reframe contribution as problem + benchmark, not model.

**Key Points**:
- The main contribution is the RRE problem formulation and the reference-resampling 
  evaluation protocol, not a new model architecture.
- The simplicity of the method is a feature, not a bug. Complex methods (RRN, Full RAMP) 
  were evaluated and found to provide no benefit.
- The paper's title and framing explicitly position it as a reliability estimation method, 
  not a new segmentation model.

**Evidence to Add**:
- Include RRN (AUROC=0.573) and Full RAMP (Delta Dice=0.0) results in appendix to show 
  that complex methods don't help
- Clarify that the contribution is establishing a benchmark and baseline

---

## R3: "Multi-Reference Inference Cost"

**Response Strategy**: Acknowledge cost, argue value proposition.

**Key Points**:
- The 6x inference cost is a real concern. However, multi-reference inference is already 
  standard practice for improving segmentation quality. We are not adding cost—we are 
  extracting additional value (reliability information) from existing computation.
- The pairwise IoU computation is negligible (15 IoU computations per query, milliseconds).
- For applications where reliability is critical (e.g., medical diagnosis), 6x inference 
  cost is acceptable if it prevents catastrophic failures.

**Evidence to Add**:
- Add a cost-benefit analysis: inference time vs. failure reduction
- Compare against the cost of human review for flagged queries

---

## R4: "Cross-Dataset Instability"

**Response Strategy**: Acknowledge limitations, provide honest analysis.

**Key Points**:
- We are honest about cross-dataset limitations. This is a feature of the paper, not a bug.
- The boundary analysis is pre-registered (Phase AB protocol). We categorize datasets 
  based on pre-defined criteria, not post-hoc.
- MoNuSeg has no catastrophic failures to detect—the method works but is unnecessary. 
  This is an important finding for practitioners.

**Evidence to Add**:
- Reference the pre-registered protocol (docs/phaseAB_cross_dataset_protocol.md)
- Add more datasets if possible

---

## R5: "Selective Prediction Is Just Abstention"

**Response Strategy**: Distinguish from existing work.

**Key Points**:
- The novelty is in the *risk signal*, not the abstention mechanism. We show that 
  prediction consistency—a signal derived from the multi-reference setup itself—is 
  effective for risk estimation.
- The comparison to RRN (AUROC=0.573) actually supports our claim: a learned method 
  performs worse than our simple heuristic on the primary metric (AURC).
- Full risk-coverage curves are provided in the supplementary material.

**Evidence to Add**:
- Include full risk-coverage curves (all coverage levels)
- Clarify that AUROC is not the primary metric because it ignores ranking quality 
  among non-catastrophic queries

---

## R6: "Limited Confirmatory Evidence"

**Response Strategy**: Acknowledge honestly, frame as benchmark.

**Key Points**:
- We are transparent about the lack of a confirmatory cohort. The paper is explicitly 
  positioned as exploratory/benchmark.
- Power analysis is provided to guide future confirmatory studies.
- The evaluation protocol (reference-resampling, query-level statistics) is designed 
  to be reproducible.

**Evidence to Add**:
- Include power analysis in the main paper (or appendix)
- Provide clear guidance for future confirmatory studies

---

## R7: "Uncertainty Estimation Comparison"

**Response Strategy**: Distinguish from model uncertainty.

**Key Points**:
- Prediction consistency captures *reference-induced* uncertainty, which is orthogonal 
  to model uncertainty.
- Standard uncertainty methods (MC dropout, ensembles) require model modification or 
  multiple forward passes. Our method requires neither.
- The two types of uncertainty are complementary. We acknowledge this as future work.

**Evidence to Add**:
- Discuss the relationship between reference-induced uncertainty and model uncertainty
- Add a note about complementary signals as future work
