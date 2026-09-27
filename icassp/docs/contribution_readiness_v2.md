# Contribution readiness audit v2 — no manuscript claims

## What is now supported

**Core contribution.** In a frozen DINOv2-L + SAM ViT-B reference-guided
nuclei pipeline, selecting an optional background-prompt budget per episode
from `{0,5,10,20}` is superior to both the prior two-action selector and every
fixed budget on MoNuSeg nested outer tests.

| Requirement | Evidence | Verdict |
|---|---|---|
| Leakage-safe selection | 5 folds; predictors train on inner train and hyperparameters choose on 30 inner-validation episodes before outer scoring | supported |
| Main efficacy | 37 outer queries × 5 references; four-action vs two-action +0.00315, CI [0.00074, 0.00633] | supported |
| Not just a fixed budget | best fixed N5: four-action +0.01234, CI [0.00338, 0.02234] | supported |
| Action-space is used | N10/N20 selected on 26/185 outer episodes | supported |
| Headroom interpretation | oracle is +0.00702 above deployable policy; labelled diagnostic only | supported |
| Reproducibility | independent verification report passes 8 checks | supported |
| Seed stability | original plus three independent ExtraTrees seeds all retain positive two-action CI lower bounds | supported |

## Crucial negative evidence retained

1. Direct tree-spread risk gating and cross-reference consensus did **not**
   improve the two-action selector; neither is a claimed method.
2. On full PanNuke fixed-budget stress testing (2,540 queries, 12,700
   episodes), N0 is best; N10 and N20 significantly harm Dice. The result
   rejects a universal "more background points" story.
3. Frozen MoNuSeg-to-MoNuSAC four-action ensemble has no significant gain over
   the existing two-action external reference. The method is not yet a
   cross-domain adaptive-policy claim.
4. Completed FoB five-fold diagnostic uses SAM-H and a different prompt
   representation, so it remains non-protocol-matched and cannot support a
   direct numerical superiority claim.

## Honest novelty position

The defensible gap is **response-level adaptive prompt-budget selection across
more than a binary optional intervention**, tested with nested action selection
and fixed-budget counterfactuals. It is not uncertainty calibration, universal
prompt scaling, cross-domain policy transfer, instance-separation improvement,
or a claim of superiority to FoB.

## Remaining publication-risk reducers

1. A protocol-matched released comparator, or a public implementation adapted
   under the exact same support/query and SAM backbone setup.
2. One patient-disjoint pathology cohort with trainable four-budget outcomes;
   PanNuke lacks patient IDs and is therefore a stress test only.
3. A second patient-disjoint pathology cohort with trainable four-budget outcomes remains the main evidence gap.

The current evidence is materially stronger than a one-dataset prompt ablation,
but it cannot honestly substantiate an 80% acceptance-probability guarantee.
