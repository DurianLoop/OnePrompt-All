# Title Candidates (Directive §19)

The final title is chosen at WP4 based on which contributions actually pass
their Gates. All three candidates below are taken verbatim from directive
§19. Two additional candidates are offered for reviewer-orientation; the
selected title will be one of these five.

## Candidate Titles

1. **Reference-Induced Instability in Visual-Reference Segmentation: A
   Cross-Reference Benchmark and Failure-Boundary Analysis**
   (directive §19, default for the Benchmark + failure-boundary route; fits
   the actual outcome: WP1 confirmed, WP2 mixed, M2 pass, M1 negative.)

2. **Beyond a Single Reference: Measuring and Mitigating Reference-Induced
   Risk in Prompted Segmentation**
   (directive §19, Hybrid-route candidate; emphasizes the M2 selective
   prediction mitigation.)

3. **RefShift: A Benchmark for Reference Resampling, Tail Risk, and
   Selective Prompting**
   (directive §19, Hybrid-route candidate; short memorable name aligned with
   the resampling protocol.)

4. **When Valid References Disagree: A Holdout-Confirmed Benchmark and
   Dual-Model Selective Prediction for Reference-Conditioned Segmentation**
   (Additional candidate; foregrounds the dual-model M2 contribution and the
   anti-correlation finding that forces it.)

5. **Reference Risk in Frozen Prompted Segmentation: Variance, Tail Loss,
   and Abstention**
   (Additional candidate; foregrounds the frozen DINOv2-L + SAM-B pipeline
   and the risk-coverage framing.)

## Selection Rationale

The Phase T outcome is: WP1 holdout 10/10 PASS, WP2 MoNuSeg strong support
(9/9), WP2 IDRiD FAIL (4/5 directional), M2 selective prediction PASS
(6/6 holdout), M1 dynamic budget NO_GAIN. Because the program produced one
positive mitigation (M2) plus a documented negative result (M1) plus a
cross-dataset failure boundary (IDRiD), the paper is a *benchmark +
failure-boundary + one positive mitigation* paper rather than a
benchmark-only paper. Candidate 1 is the default and is recommended: it
signals the benchmark and failure-boundary framing without over-claiming a
mitigation, and it does not collide with any prohibited claim (no "SOTA",
no "instability solved", no "universal across datasets"). Candidate 2 is
the fallback if reviewer feedback favors a mitigation-forward framing.

## Prohibited Tokens (claim registry)

None of the candidates contain "SOTA", "state-of-the-art", "solved",
"eliminated", "universal", "submission-ready", or "publication-completed".
