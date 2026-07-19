# Route B Benchmark Metric and Control Protocol v1

Date: 2026-07-19

Route: `B -- Reference Promptability Benchmark`

Config: `configs/experiments/route_b_promptability_benchmark_v1.json`

## Scope

This protocol freezes the measurement layer for Route B before any new matcher or segmenter inference. It does not define a final method and does not authorize formal test access.

## Row Families

Route B tables must separate the following row families:

- automatic rows: prompts or masks generated from frozen, test-time-safe reference/query evidence;
- matched-control rows: random, scale-matched, full-image, or alternative-reference controls that expose non-reference shortcuts;
- privileged upper-bound rows: GT point/box/mask prompts used only to measure prompt-interface headroom;
- direct-matching rows: segmentation directly from matching evidence, before the prompt interface;
- diagnostic baseline rows: calibrated predictors or simple baselines used to explain failure, not to claim final method superiority.

Automatic rows must not use query GT, formal-test labels, post-hoc action search, or test-tuned thresholds.

## Required Metrics

| Metric | Definition | Claim boundary |
| --- | --- | --- |
| Promptability Gap | `U_privileged(q,t) - U_automatic(q,t)` | Measures prompt-interface headroom, not automatic success. |
| Localization Advantage | `U_automatic(q,t) - U_matched_control(q,t)` | Measures whether reference evidence beats support-matched controls. |
| Interface Gain | `U_segmenter(q,t) - U_raw_proposal(q,t)` | Measures whether the prompt interface improves or degrades matching evidence. |
| Oracle Adaptation Value | `max_t U(q,t) - max_t mean_q U(q,t)` | Measures possible benefit from per-query representation choice. |
| Promptability Reliability | Calibration of predicted viability against observed viability | Supports diagnostic prediction only after train/valid calibration evidence. |

## Required Prompt Representations

- point;
- multi-point;
- grouped point;
- box;
- multi-box;
- hard mask;
- soft mask;
- direct matching mask;
- privileged GT point;
- privileged GT box;
- privileged GT mask.

## Required Controls

- spatial random point/control;
- size-matched random box;
- full-image box;
- alternative-reference control;
- strongest fixed representation;
- raw matching baseline;
- privileged GT upper bounds.

## Development Gate B1

Before Route B development inference:

1. DDR train/valid manifests must pass.
2. DDR test must remain locked.
3. Metrics, row families, controls, and prompt representations must be frozen in config.
4. Output schemas and statistical units must be declared.
5. No formal-test claim may be made.

Passing this gate only authorizes development-only evidence generation. It does not mean `TRACK_B_BENCHMARK_READY`, because benchmark readiness still requires multi-dataset breadth, model/representation combinations, statistics, baseline predictors, and a non-trivial cross-domain finding.
