# Spatially Diversified Prompts for Frozen Matching-Based Foundation Segmentation

## Abstract

Reference-conditioned segmentation can generate point prompts for a frozen
foundation segmenter by matching a labelled reference image to an unlabelled
query.  A common implementation takes the highest-scoring locations
independently, which can concentrate a fixed prompt budget in one local region.
We study whether the *spatial layout* of automatically generated prompts
matters when the matcher, prompt budget, and segmenter are otherwise fixed.
Our deterministic spread rule traverses the same foreground-confidence ranking
as Top-5, but accepts a positive location only when it is at least three
frozen-grid units from previously accepted positives; it retains five positive
and three negative prompts through a deterministic fallback.  On the locally
processed MoNuSAC held-out protocol, DINOv2-L plus SAM-B improves from 0.4196
to 0.4526 mean Dice over the matched Top-5 baseline.  The paired gain is
+0.0330 (95% bootstrap CI [+0.0194, +0.0467]) across 303
reference-query episodes.  The same fixed action has positive paired intervals
in four additional completed replications spanning a second reference seed,
DINOv2-B, and SAM-L/H.  A clean-development adaptive selector is evaluated
once on external IDRiD after its action set is locked; it scores 0.0725 Dice,
below the 0.0958 Top-5 rule.  We therefore present adaptive selection as a
negative transfer result rather than as the contribution.  The evidence
supports a bounded conclusion: for this frozen MoNuSAC protocol, spatially
diversifying matched prompts is preferable to independently choosing the top
five locations at the same prompt budget.

**Keywords:** promptable segmentation; reference-conditioned matching; spatial
diversification; frozen foundation models; paired evaluation

## 1. Introduction

Promptable foundation segmenters such as the Segment Anything Model (SAM)
accept sparse point prompts and can be used without task-specific parameter
updates [1].  In reference-conditioned use, a labelled image-mask pair can
define foreground and background appearance, while a visual feature extractor
produces a dense matching map on the query.  Self-supervised DINOv2 features
provide one such frozen visual representation [2].  This pipeline shifts an
important decision upstream of the segmenter: how should a small, automatic
prompt budget be laid out on the query?

The usual Top-*K* rule chooses each high-confidence positive independently.
That is attractive for its simplicity, but it can spend multiple points on a
single locally redundant region.  We isolate this design choice without adding
training to the matcher or segmenter.  The proposed action keeps the candidate
ranking and number of positive and negative prompts unchanged, then enforces a
minimum separation only among selected positives.

This paper makes three evidence-bounded contributions:

1. A deterministic spatially diversified prompt action that differs from the
   matched Top-5 baseline only in positive-point layout.
2. A paired, held-out evaluation in which every fixed action is run on the
   same reference-query episodes and uncertainty is reported by a deterministic
   paired bootstrap.
3. Five completed MoNuSAC replications across reference seeds, DINOv2 matcher
   sizes, and SAM-B/L/H, together with an explicitly negative external test of
   the current adaptive selector.

We do not claim cross-domain transfer, clinical utility, state-of-the-art
performance, superiority to trained medical segmenters, or successful adaptive
prompt selection.

## 2. Method

### 2.1 Frozen reference matching

Given a labelled reference image-mask pair \((r,m_r)\) and a query image
\(q\), frozen DINOv2 patch features define foreground and background prototypes
by weighted pooling inside and outside \(m_r\).  Let \(S_f\) and \(S_b\) be the
query patch cosine-similarity maps to those prototypes.  The fixed ranking map
is

\[
C = z(S_f) - z(S_b),
\]

where \(z\) is within-map standardization.  Positive candidates are traversed
from high to low \(C\); negative prompts are the three lowest \(C\) locations
after excluding selected positives.  The matcher stays frozen in every
experiment.

### 2.2 Matched prompt-layout actions

Both actions use five positive and three negative points:

| Action | Positive selection | Negative selection |
|---|---|---|
| `top5_n3` | First five locations in descending \(C\) | Bottom three values of \(C\), excluding positives |
| `spread5_n3` | Descending \(C\), accepting a location only if it is at least three patch-grid units from accepted positives | Identical to `top5_n3` |

If the separation constraint leaves fewer than five positives, the remaining
slots are filled from the same descending ranking.  Thus the action changes
only layout: the score construction, prompt count, negative prompts, and frozen
segmenter are matched.  Coordinates are scaled to the input image and passed to
SAM; the highest-scoring returned mask is evaluated.

### 2.3 What is and is not adaptive

We also developed an ExtraTrees selector over a finite set of fixed-layout
actions using clean MoNuSAC development episodes.  Its leaf size and action set
were locked before one external IDRiD evaluation.  This selector is not the
method contribution: its external result is negative (Section 3.4).  No IDRiD
mask, Dice value, or target-domain result is used to select a new policy or
radius after that test.

## 3. Experimental Protocol

### 3.1 Data, frozen components, and episode construction

The main evidence uses the locally processed MoNuSAC split, a multi-organ
nuclei segmentation benchmark [3].  It contains 209 training cases and 101
held-out test cases in this processing pipeline.  The primary comparison uses
frozen DINOv2-L/14 matching and frozen SAM-B.  Three training references are
fixed before held-out evaluation; each of the 101 queries is paired with each
reference, yielding 303 reference-query episodes per action.  A second,
independently fixed reference seed repeats this protocol.

The compatibility replications preserve the two fixed prompt actions and the
same paired episode principle while changing either the matcher (DINOv2-B) or
the segmenter (SAM-L or SAM-H).  Every reported row contains 303 action rows
for `top5_n3` and 303 for `spread5_n3`.  The stage-18 SAM-H artifact is
validated for balanced action rows before it enters the report.

### 3.2 Statistical unit and reporting

For a configuration, the Dice difference `spread5_n3 - top5_n3` is computed
within each identical reference-query episode.  We report mean Dice, the mean
paired difference, win/tie/loss counts, and a 95% paired bootstrap interval
using 10,000 resamples with seed 20260719.  An episode is the reporting unit;
patches and prompts are not treated as independent samples.

### 3.3 Geometry and qualitative protocols

The geometry analysis measures each episode's mean positive nearest-neighbour
distance on the frozen 37 x 37 DINO patch grid.  It is computed from selected
points and needs no query mask at inference.  This verifies that the proposed
action changes layout at the matched budget; it is not a test-time radius
selection procedure.

The qualitative protocol locks three IDRiD query identities
(`IDRiD_68`, `IDRiD_75`, and `IDRiD_81`) before evaluation.  Each rendered
panel contains all three fixed IDRiD references (`IDRiD_20`, `IDRiD_47`, and
`IDRiD_50`), the query ground truth, the matched Top-5 output, and the frozen
policy output.  No Dice, utility, ranking, or result-based case filter selects
these panels.

## 4. Results

### 4.1 Matched-budget primary result

For DINOv2-L plus SAM-B under reference seed 20260716, spatial diversification
raises mean Dice from 0.4196 to 0.4526 (+0.0330).  The paired 95% bootstrap
interval is [+0.0194, +0.0467], and spread wins, ties, and loses 125, 110, and
68 episodes, respectively.  Because the negative prompts and point counts are
identical, this result isolates the positive-point layout choice.

### 4.2 Replication over matcher, segmenter, and reference seed

**Table 1. Matched fixed-layout replication on MoNuSAC (303 paired episodes
per row).**

| Combination / reference seed | Top-5 + 3 neg. | Spread r=3 + 3 neg. | Gain |
|---|---:|---:|---:|
| DINOv2-L + SAM-B / 20260716 | 0.4196 | 0.4526 | +0.0330 |
| DINOv2-L + SAM-B / 20260717 | 0.4008 | 0.4362 | +0.0354 |
| DINOv2-B + SAM-B / 20260716 | 0.3835 | 0.4344 | +0.0508 |
| DINOv2-L + SAM-L / 20260716 | 0.4507 | 0.4659 | +0.0152 |
| DINOv2-L + SAM-H / 20260716 | 0.4289 | 0.4439 | +0.0149 |

**Table 2. Paired uncertainty for the fixed-layout effect.**

| Combination / reference seed | 95% paired bootstrap CI | Win / tie / loss |
|---|---:|---:|
| DINOv2-L + SAM-B / 20260716 | [+0.0194, +0.0467] | 125 / 110 / 68 |
| DINOv2-L + SAM-B / 20260717 | [+0.0203, +0.0514] | 115 / 117 / 71 |
| DINOv2-B + SAM-B / 20260716 | [+0.0353, +0.0674] | 131 / 101 / 71 |
| DINOv2-L + SAM-L / 20260716 | [+0.0064, +0.0250] | 108 / 111 / 84 |
| DINOv2-L + SAM-H / 20260716 | [+0.0036, +0.0265] | 110 / 111 / 82 |

All five intervals are positive.  The smallest observed gain occurs with
DINOv2-L plus SAM-H, where the interval remains above zero.  These replications
support compatibility with the evaluated frozen components; they do not show
that any arbitrary matcher or segmenter will benefit.

### 4.3 Mechanism check: the action changes point layout

For the primary DINOv2-L + SAM-B configuration, the mean positive
nearest-neighbour distance rises from 7.460 to 9.673 patches.  The paired
change is +2.213 patches, with a 95% interval of [+1.861, +2.579].  The other
three reported geometry checks have positive changes of +2.133, +2.443, and
+2.213 patches.  Together with the identical prompt budget, this confirms that
the Dice difference is consistent with a spatial-layout mechanism rather than
with more prompts.

### 4.4 Clean adaptive-policy external negative

The development selector was frozen after clean MoNuSAC development and then
evaluated once on the external IDRiD protocol.  On 81 reference-query episodes,
the selector scores 0.0725 Dice.  The fixed Top-5 rule scores 0.0958; the three
fixed spread actions with radii 2, 3, and 4 score 0.0917, 0.0533, and 0.0425.
The present matching statistics and geometry features therefore do not support
adaptive radius selection across this shift.  We record this result rather than
retuning the selector or treating the observed target scores as a development
signal.

### 4.5 Predeclared qualitative panels

Figure 1 comprises the rendered panels
`IDRiD_68_panel.png`, `IDRiD_75_panel.png`, and `IDRiD_81_panel.png` in
`artifacts/stage17_clean_adaptive_layout/qualitative_rendered/`.  Each has
three rows, one for each locked reference; its four columns show the reference
with mask, query ground truth, matched Top-5, and frozen policy result.  The
panels expose success and failure modes without outcome-based case selection.
They are qualitative protocol evidence, not a per-image performance claim.

## 5. Discussion and Limitations

The central result is deliberately narrow.  At the same five-positive,
three-negative prompt budget, a deterministic separation rule changes the
geometry of matched points and improves paired held-out MoNuSAC Dice in five
completed frozen-model replications.  The effect is not explained by a larger
prompt budget, since the candidate ranking, negative prompts, and counts are
shared.

Several limitations bound this result.  The main quantitative evidence is one
locally processed MoNuSAC protocol and binary foreground masks.  The evaluated
compatibility settings establish neither universal backbone behavior nor
cross-domain generalization.  The current adaptive selector fails on the
once-read external IDRiD test, so no adaptive-transfer claim is warranted.
Future work should preregister an external protocol before selecting a
separation policy, evaluate additional reference sets and datasets, and test
whether a selector can be developed without reusing external test feedback.

## 6. Reproducibility

The manuscript's numerical authority is
`paper/fixed_rule_evidence_manifest.md`.  The paired report is
`artifacts/fixed_layout_effects.json`; the readable table is
`artifacts/evidence_tables.md`; the balanced SAM-H validation output is
`artifacts/stage18_dinol_samh_monusac.validation.stdout.log`; and the
qualitative render manifest is
`artifacts/stage17_clean_adaptive_layout/qualitative_rendered/render_summary.json`.

Validate the manuscript table against the artifact report with:

```powershell
D:\conda\python.exe -B -m scripts.validate_fixed_rule_manuscript `
  --manuscript-path paper\spatial_layout_matching_prompts_draft.md
```

Run the focused regression suite with:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\conda\python.exe -m pytest -q -p no:cacheprovider `
  tests\test_statistics.py tests\test_analyze_fixed_layout_effects.py `
  tests\test_qualitative.py tests\test_build_stage17_qualitative_panels.py `
  tests\test_render_stage17_qualitative.py tests\test_build_evidence_tables.py `
  tests\test_validate_fixed_rule_manuscript.py
```

## References

1. Alexander Kirillov et al. *Segment Anything*. ICCV, 2023.
   https://arxiv.org/abs/2304.02643
2. Maxime Oquab et al. *DINOv2: Learning Robust Visual Features without
   Supervision*. arXiv:2304.07193, 2023.
   https://arxiv.org/abs/2304.07193
3. Ruchika Verma et al. *MoNuSAC2020: A Multi-Organ Nuclei Segmentation and
   Classification Challenge*. IEEE Transactions on Medical Imaging, 2021.
   https://monusac-2020.grand-challenge.org/
