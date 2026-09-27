# ICASSP 2027 submission package

This package is an evidence-locked, anonymous ICASSP 2027 draft for the Biomedical Imaging & Signal Processing / Machine Learning track.

## Proposed paper

**Response-Gated Background Prompting for Reference-Guided Nuclei Segmentation**

The paper studies a deliberately narrow question: after foreground--background matching produces a shared positive prompt set for a frozen DINOv2--SAM pipeline, can an observable SAM response decide whether a small background-point budget should be accepted?  This avoids unsupported claims of universal prompt policies, model agnosticism, or SOTA. A reproduced one-fold FoB adaptation is preserved as a negative-transfer diagnostic, not presented as a fair head-to-head comparison.

## Package map

- `paper/`: anonymous, locally compiled two-column manuscript and bibliography. It is content-complete but must be rebuilt with the official ICASSP class before upload.
- `paper/figures/`: evidence-locked main and qualitative figures.
- `evidence/`: source CSV/JSON snapshots used by the manuscript.
- `baselines/fob_source/` and `runs/fob_monuseg_fold0/`: copied/compatibility-documented FoB source plus its pre-registered one-fold train, selection, and one-time test record.
- `scripts/verify_evidence.py`: deterministic cross-file integrity checks.
- `evidence/verification_report.json`: result of the current verification run.
- `docs/research_assessment.md`: hypothesis, protocol, results, failure boundaries, and remaining pre-submission risks.
- `docs/acceptance_readiness.md`: evidence-based decision record separating submission readiness from an unprovable numeric acceptance promise.
- `docs/baseline_eligibility_audit.md`: reasoned exclusion of an incompatible historical GBMSeg-style artifact from the nested-CV comparison table.
- `docs/next_cohort_protocol.md`: GlaS feasibility exclusion and the completed pre-registered PanNuke field-level zero-shot protocol.
- `evidence/pannuke_zero_shot_summary.json`: complete 2,540-query PanNuke result, including the mixed-sklearn version isolation record; its scope is field-level rather than patient-disjoint.
- `evidence/pannuke_support_sensitivity_summary.json`: descriptive five-support breakdown; it is explicitly not a support-selection analysis.
- `evidence/pannuke_tissue_sensitivity_summary.json`: descriptive public-tissue-code breakdown, including indeterminate strata; it is not a subgroup-selection result.
- `evidence/nested_instance_metrics_summary.json`: diagnostic object-recall/merge-error analysis that constrains the manuscript to a union-mask Dice claim.
- `evidence/end_to_end_latency_summary.json`: CUDA-synchronized 30-episode DINOv2-L + SAM ViT-B latency record with declared GPU, input size, and no-query-label policy.
- `SUBMISSION_CHECKLIST.md`: human actions still required before upload.

## Reproduce the package audit

```powershell
D:\conda\python.exe -B .\icassp\scripts\verify_evidence.py
```

The audit checks the primary means, paired effects, confidence intervals, inference units, numerical-consistency claim, and that every copied figure exists. It validates the paper package, not the original GPU training/inference pipeline.

## Honest readiness statement

The underlying evidence is credible enough for a focused conference draft: 37-query five-fold nested CV, patient-disjoint external evaluation, a pre-registered 2,540-query field-level PanNuke diagnostic, paired bootstrap CIs, a device-consistency audit, and explicit negative-transfer results. In PanNuke, fixed N5 is harmful but the version-isolated frozen gate improves N0 by 0.00470 Dice (95% CI [0.00402, 0.00542]); this supports selective rejection of a harmful fixed policy, not patient-level generalization. A reproducible one-fold FoB adaptation also failed to transfer (outer Dice 0.00477) and is retained only as a diagnostic boundary. It is not a guaranteed acceptance: the remaining material risks are a fully protocol-matched multi-fold comparator and confirmation on another patient-disjoint pathology cohort. The draft therefore makes only the claims supported by the packaged evidence.
