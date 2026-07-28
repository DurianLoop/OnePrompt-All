# Research Track Selection

Date: 2026-07-19

Phase 4 audit state: `PHASE4_AUDIT_SUPPORTS_BENCHMARK_TRACK`

Selected primary track: `Route B -- Reference Promptability Benchmark`

Selected fallback track: `Route A -- Promptability-Aware Routing`

Deferred track: `Route C -- Direct Reference Segmentation`

## Scoring

| Track | Role | Score | Evidence-based reason |
| --- | --- | ---: | --- |
| B Reference Promptability Benchmark | primary | 90 | Phase 4 exposes a control and measurement problem: high candidate precision, privileged box upper bounds, and high absolute R03 box IoU do not establish automatic prompt success. |
| A Promptability-Aware Routing | fallback | 78 | It unifies the failures conceptually, but legal promptability labels and separability evidence are not yet established. |
| C Direct Reference Segmentation | deferred | 70 | R00 is precise but too low-recall for immediate direct segmentation; a real dense mechanism would be required. |

## Decision

Route B is selected because the strongest current evidence is diagnostic rather than method-positive. The project should first build a reusable promptability benchmark with matched controls, privileged upper bounds, prompt representations, failure taxonomy, and calibration baselines across fresh allowed data.

Route A remains the fallback if benchmark evidence shows separable promptability labels and a predictable viability signal. Route C remains deferred unless direct matching evidence improves under a new structured dense mechanism.

No formal test data were used for this selection.
