# PanNuke four-budget stress test — protocol

## Purpose

This is a cross-dataset **fixed-action** test of whether the four background
budgets used by the MoNuSeg adaptive policy have a consistent raw effect on
PanNuke. It deliberately does not fit, tune, or select a policy on PanNuke.

## Frozen cohort and procedure

- 2,540 non-empty PanNuke fold-1 query tiles;
- five pre-registered PanNuke fold-2 support tiles per query;
- frozen DINOv2-L foreground/background matcher, 40 positive points, `lambda=0.75`, SAM ViT-B;
- the existing N0/N5 raw outputs remain unchanged in `pannuke_zero_shot_v1`;
- `run_pannuke_extra_budgets.py` adds only N10/N20 with exactly the same matching score map and positive points;
- the query annotation is opened only after N10 and N20 predictions have both been decoded.

## Decision rule

No target-domain labels may affect prompt construction, action selection, or
model parameters. The completed analysis reports only fixed-budget query-level
means and paired 10,000-resample bootstrap intervals. Per-episode oracle is a
diagnostic ceiling, never a deployed policy.

## Interpretation guardrail

This test can establish whether the **action-space premise** transports to
another pathology dataset. It cannot establish that a MoNuSeg-trained adaptive
selector transfers: that stronger claim is separately tested by the frozen
MoNuSAC source-fold ensemble and currently has no significant gain.

## Completed result

All 2,540 queries and 12,700 support-query episodes completed. Fixed N0 was
best (Dice 0.49267); fixed N10 and N20 were significantly worse than N0 by
-0.01529 (95% CI [-0.01721, -0.01345]) and -0.01952 ([-0.02184, -0.01726]),
respectively. This is a retained counterexample: larger negative budgets are
not a universal improvement, reinforcing the need for per-episode action
selection rather than a global budget rule.
