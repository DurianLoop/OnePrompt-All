# FSPL Experiment Completion Gate

This is an evidence gate, not a manuscript draft.  It records the minimum
current-state evidence required before any paper text can describe the method
as an experimentally supported contribution.

| Requirement | Authoritative evidence | Current state | Allowed conclusion |
|---|---|---|---|
| Matched prompt-budget ablation | `artifacts/stage9_monusac_full_fixed_ablation/summary.json` and stage 6 rows | Complete | More positive points and negative prompts help on the fixed MoNuSAC protocol. |
| Prespecified radius-3 comparison | stage 6 and stage 8 summaries/rows | Complete | Fixed spatial diversification improves over matched Top-5 for two independently selected reference seeds. |
| Matcher replication | `artifacts/stage13_dinob_samb_monusac/summary.json` | Complete | The fixed effect is not tied only to DINOv2-L. |
| Segmenter replication: SAM-L | `artifacts/stage16_dinol_saml_monusac/summary.json` | Complete | The fixed effect is present with SAM-L. |
| Segmenter replication: SAM-H | `artifacts/stage18_dinol_samh_monusac/summary.json` plus balanced action CSV | Complete | Spread r=3 improves 0.4439 versus 0.4289 for Top-5 on the same 303 episodes. |
| Paired uncertainty reporting | `artifacts/fixed_layout_effects.json` | Complete for all five rows | SAM-H paired gain is +0.0149, 95% CI [+0.0036, +0.0265]; every completed fixed-rule interval is positive. |
| Clean adaptive policy test | `artifacts/stage17_clean_adaptive_layout/summary.json` | Complete, negative | The current adaptive radius selector does not transfer from clean MoNuSAC development to IDRiD. |
| Qualitative protocol | `qualitative_manifest.json`, `qualitative_panels.json`, and `qualitative_rendered/render_summary.json` | Complete | All three predeclared queries and all three references are rendered without metric-based filtering; visual QA found complete, legible panels. |
| Reproducible validation | targeted FSPL tests and `stage18_dinol_samh_monusac.validation.stdout.log` | Complete | The stage-18 validator confirms exactly 303 paired episodes for each fixed rule. |

## Current decision

**The evidence gate now supports a bounded fixed-rule paper.**  Across the two
reference seeds, two DINOv2 matcher sizes, and three SAM segmenters (B/L/H),
the fixed spread-r=3 action improves paired MoNuSAC Dice over the matched
Top-5 baseline.  This is not evidence for adaptive-radius transfer,
cross-domain superiority, or superiority over trained medical segmenters.  The
adaptive-policy branch is a recorded negative result and cannot be retuned
against the observed IDRiD test data.

## Automatic next checks

1. Preserve the completed artifacts and use only the bounded fixed-rule claim
   in any manuscript draft.
2. Report the clean adaptive-policy result as a negative transfer finding; do
   not run another selector choice against the observed IDRiD test data.
3. Keep the paper scope to the evaluated MoNuSAC protocol, rather than
   presenting it as a universal or cross-domain segmentation method.
