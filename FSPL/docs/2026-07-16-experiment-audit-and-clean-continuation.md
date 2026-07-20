# Experiment Audit and Clean Continuation Protocol

## Purpose

This note records which observations are confirmatory, which are exploratory,
and how the remaining experiments must be conducted before they can support a
paper claim.  It supersedes any wording that would imply that a separation
radius was selected without looking at a held-out test score.

## Confirmed evidence already available

All runs below use frozen matching and segmentation models, fixed reference
sets, and mean Dice over reference-query episodes.

| Question | Dataset / protocol | Evidence | Status |
|---|---|---|---|
| Do more positive points help? | MoNuSAC, seed 20260716, 303 held-out episodes/action | Top-1: 0.0907; Top-3: 0.3266; Top-5: 0.3814 | Confirmatory fixed-rule ablation |
| Do negative prompts help? | Same protocol | Top-5/no-negative: 0.3814; Top-5/3-negative: 0.4196 | Confirmatory fixed-rule ablation |
| Does a prespecified spread action help? | MoNuSAC, two independent 3-reference seeds, 303 held-out episodes/seed | radius 3: +0.0330 and +0.0354 versus matched Top-5/3-negative | Confirmatory only for the preselected radius-3 action |
| Is the layout action tied to DINOv2-L? | MoNuSAC, seed 20260716, DINOv2-B + SAM-B | 0.3835 to 0.4344 (+0.0508) | Confirmatory matcher-backbone replication |
| Does the current offline policy beat fixed spread? | MoNuSAC, seed 20260716 | policy 0.4444; fixed spread-radius-3 0.4526 | Negative result; do not claim policy superiority |
| Does the clean adaptive policy transfer out of domain? | Clean MoNuSAC development pool, then the once-read IDRiD test | validation policy 0.4056 versus best fixed d2 0.4049; IDRiD policy 0.0725 versus fixed Top-5 0.0958 | External negative result; terminate this adaptive branch without retuning |

## Exploratory observations that must not select the final policy

After the first MoNuSAC held-out comparison had been inspected, additional
radius-2 and radius-4 actions were evaluated on the same official test cases.
The resulting sequence (0, 1, 2, 3, 4 patch radii) was 0.4196, 0.4196,
0.4474, 0.4526, and 0.4629 Dice.  The radius-4 number is useful for hypothesis
generation, but it is **not** an unbiased held-out estimate for choosing the
final radius or an adaptive action set.

MoNuSeg also had its official test scores inspected for radius 0 and 3 before
the radius-2/radius-4 transfer check.  Its small radius-2 difference
(0.3794 versus 0.3788) is therefore descriptive only, not a clean external
selection result.

## Consequences for the paper

1. The primary supported method claim is the prespecified fixed
   radius-3 spatial-layout action, not an automatically learned prompt policy.
2. The clean adaptive-radius experiment completed with the locked MoNuSAC
   development pool and the once-read IDRiD test.  Its small validation gain
   (0.4056 versus the fixed d2 rule's 0.4049) did not transfer: the frozen
   policy scored 0.0725 on IDRiD versus 0.0958 for fixed Top-5.  This branch is
   a negative result, not a policy to tune or promote.
3. IDRiD's test labels have now been observed.  They must not be reused to
   change the action set, feature set, regressor, or hyperparameters, nor to
   claim a retried adaptive result.  Any future adaptive-policy study needs a
   fresh predeclared external split/dataset and a separate development pool.
4. The present manuscript must label the radius sweep as exploratory, retain
   the fixed radius-3 test result as the confirmatory result, and avoid an
   adaptive-policy headline claim.

## Continuation sequence

1. Finish the remaining frozen matcher x segmenter compatibility check
   (DINOv2-L + SAM-H) with the already fixed Top-5 versus radius-3 rules.
   This check changes no prompt-policy choice and is independent of the
   terminated adaptive-radius branch.
2. Preserve the completed clean adaptive result as a negative finding.  Do
   not make another adaptive-policy selection using IDRiD, MoNuSAC test, or
   MoNuSeg test labels already seen in this worktree.
3. Produce preselected qualitative cases and prompt-geometry statistics
   without choosing images based on a method advantage.
4. Write the paper around the conclusions actually supported by these staged
   experiments: fixed spatially diversified prompts, with the adaptive branch
   omitted from headline claims or reported plainly as a failed transfer.

## Reporting language

Use "spatially diversified matching prompts" for the current validated
contribution.  Do not use "state of the art", "universal plug-and-play", or
"adaptive prompt policy" as headline claims unless the remaining clean
experiments establish those statements.

## Locked clean development pool

The audit excludes all 63 MoNuSAC training cases used by the seed-20260716
multi-reference development protocol and the three seed-20260717 reference
cases.  This leaves **143 of 209** training cases that have not appeared in a
reference, policy-training, validation, or qualitative role.  The clean
adaptive-layout development run will use only these 143 cases, with a newly
fixed random seed (20260718) and a three-reference / 48-training-query /
12-validation-query partition.  The remaining 80 cases are not used by that
run.

The action set for this *new* policy experiment is locked before it is run:
`top5_n3`, `spread5_n3_d2`, `spread5_n3_d3`, and `spread5_n3_d4`.  Radius 2
and radius 4 entered this action set from prior exploratory work; therefore
the new policy will not be judged on previously inspected MoNuSAC or MoNuSeg
test masks.  Its final out-of-domain evaluation will use the previously
unread IDRiD testing split (27 cases), with three IDRiD training references
fixed before the test masks are read.  IDRiD contains 54 valid local training
pairs and 27 valid local testing pairs; an auxiliary unmatched
`IDRiD_14_sam_result` image is excluded by stem matching.

The deterministic preselected references are MoNuSAC `train_009`,
`train_118`, and `train_205`; the held-out IDRiD reference images are
`IDRiD_20`, `IDRiD_47`, and `IDRiD_50`.  They are generated solely by sorting
case IDs and applying `random.Random(20260718)` before any new action utility
or test metric is computed.

The three qualitative IDRiD test panels are also locked before evaluation:
`IDRiD_68`, `IDRiD_75`, and `IDRiD_81`.  Their manifest is
`artifacts/stage17_clean_adaptive_layout/qualitative_manifest.json`; it stores
only file identities at this stage, not any test Dice or method output.
`artifacts/stage17_clean_adaptive_layout/qualitative_panels.json` now expands
those three queries across all three fixed IDRiD references (nine
reference-query views).  Each view prescribes a visible `top5_n3` baseline and
the frozen policy action, but contains no Dice, utility, ranking, or
result-based case filtering.  It is therefore a rendering instruction, not a
new selection step.

The renderer has now produced the three corresponding PNG panels in
`artifacts/stage17_clean_adaptive_layout/qualitative_rendered/`, with every
locked reference-query view shown.  A visual QA pass confirmed that the
reference, query ground truth, matched Top-5, and frozen-policy columns are
present and legible, with no metric or utility displayed in the panels.

## Completed clean adaptive-layout result (stage 17)

The completed end-to-end stage-17 runner produced 576 clean MoNuSAC
development-training rows, 144 validation rows, and 324 IDRiD test rows: four
locked actions for each of 144, 36, and 81 reference-query episodes
respectively.  Its final `summary.json` records the locked protocol, all fixed
rule means, the selected leaf size, and the IDRiD result.  Before that full
run finished, `scripts/finalize_stage17_saved_rows.py` was used only to make a
reporting interruption recoverable from saved utility rows without rereading
an image, mask, matcher, or segmenter.  The completed runner summary is now
the authoritative end-to-end artifact; the postprocessor remains available
for a future interrupted reporting step.

On the clean validation episodes, the ExtraTrees selector chose
`min_samples_leaf=2` and scored 0.4056 Dice.  The fixed rules were 0.3820
(Top-5), 0.4049 (spread d2), 0.4014 (spread d3), and 0.4031 (spread d4).
On the once-read IDRiD test, the corresponding fixed rules were 0.0958,
0.0917, 0.0533, and 0.0425, while the frozen selector scored 0.0725.  The
selector is therefore not externally supported and must not be tuned against
this test.  The useful conclusion is bounded and negative: the present
development features do not transport a radius-selection policy across this
MoNuSAC-to-IDRiD shift.

## Paired fixed-layout effects

The fixed radius-3 action is evaluated against Top-5 on the *same* 303
reference-query episodes in every completed replication.  The deterministic
paired bootstrap report is `artifacts/fixed_layout_effects.json` (10,000
resamples; seed 20260719); it rejects a result table with a missing or
duplicated action for an episode.  All completed 95% intervals for
`spread5_n3 - top5_n3` are positive:

| Frozen matcher + segmenter / reference seed | Mean Dice gain | 95% paired bootstrap interval | Win / tie / loss |
|---|---:|---:|---:|
| DINOv2-L + SAM-B / 20260716 | +0.0330 | [+0.0194, +0.0467] | 125 / 110 / 68 |
| DINOv2-L + SAM-B / 20260717 | +0.0354 | [+0.0203, +0.0514] | 115 / 117 / 71 |
| DINOv2-B + SAM-B / 20260716 | +0.0508 | [+0.0353, +0.0674] | 131 / 101 / 71 |
| DINOv2-L + SAM-L / 20260716 | +0.0152 | [+0.0064, +0.0250] | 108 / 111 / 84 |
| DINOv2-L + SAM-H / 20260716 | +0.0149 | [+0.0036, +0.0265] | 110 / 111 / 82 |

This supports a bounded fixed-rule conclusion: spatial diversification has a
positive paired effect across the five completed matcher and segmenter
replications on MoNuSAC, including SAM-H.  It does not establish
adaptive-radius transfer, a cross-domain claim, or superiority over trained
medical segmentation systems.  The current adaptive selector remains a
recorded external negative result, not an alternative policy to tune.
