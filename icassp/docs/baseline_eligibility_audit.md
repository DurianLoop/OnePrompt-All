# Baseline eligibility audit

## Decision

No additional external-method number was added to the ICASSP main table from the existing workspace. The available GBMSeg-style artifact is useful historical context but is not eligible for the paper's five-fold nested-CV comparison.

## Audited artifact

`D:\SAM\runs\research_direction\baselines\gbmseg_style_adapted_test14`

The source builder, `surf/benchmarks/build_prompt_policy_paper_assets.py`, labels this row `GBMSeg-style adapted`, assigns it a `single-reference controlled` protocol, and embeds the warning: “GBMSeg-style is an adapted predecessor baseline, not an official SAT reproduction.” It evaluates 14 held-out MoNuSeg cases, whereas the manuscript's confirmatory evidence is five-fold nested CV over 37 independent outer-test queries with five references each.

## Exclusion rationale

1. Its method is an adaptation rather than the released comparator.
2. Its 14-case, single-reference protocol differs from the 37-query, five-reference nested inference unit.
3. Combining it with the nested table would create an invalid cross-protocol numerical comparison.

The manuscript instead reports the compatible FG-only and FG--BG fixed-prompt ablation within its own frozen protocol, and retains the FoB run only as explicitly non-comparable diagnostic evidence.
