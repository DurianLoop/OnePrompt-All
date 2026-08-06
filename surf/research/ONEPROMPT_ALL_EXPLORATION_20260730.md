# OnePrompt-All: first natural-image exploration

## Protocol

- Data: `D:\SAM\dataset\20张\same_class_20`, 20 LVIS banana scenes, 9
  annotated target instances per image.
- Prompt simulation: one positive point at the centroid of one target
  instance. The ground truth is used only to simulate and score the prompt.
- Metrics: instance precision/recall/F1 at IoU 0.5, union IoU, seed-mask IoU,
  and absolute instance-count error.

## Mechanisms tested

1. **DINO point clusters -> sparse SAM prompts** (existing Matcher stages).
   On image 01 it produced one retained instance from 9. This is unsuitable for
   touching instances because high-similarity DINO patches form one spatial
   cluster.
2. **SAM automatic proposals -> DINO prototype ranking -> NMS**. This is the
   current leading baseline. The prompted object yields a SAM exemplar; SAM
   generates candidate masks; DINO scores each candidate using the exemplar's
   patch prototype. No class name, reference image, or target GT is used in the
   selection stage.
3. **DINO + Lab colour fusion**. This was rejected as the primary ranker:
   appearance changes within a banana bunch can make the fusion over-select the
   seed and under-select the other fruit.

## Evidence so far

| Setting | Scope | Instance F1 | Recall | Precision | Union IoU |
|---|---:|---:|---:|---:|---:|
| SAM-B proposal oracle coverage | image 01 | — | 9/9 proposal recall | — | mean best IoU 0.918 |
| SAM-B + DINOv2-B proposal ranker | all 20 | 0.615 | 0.572 | 0.726 | 0.549 |
| SAM-L + DINOv2-B | first 2 | 0.563 | 0.611 | 0.596 | 0.509 |
| SAM-B + DINOv2-L | first 2 | 0.565 | 0.556 | 0.700 | 0.485 |

The all-20 SAM-B/DINOv2-B run used 16x16 AMG points and no crop layers for
stable operation alongside the currently occupied GPU. It has mean seed IoU
0.794 and mean absolute count error 2.75. Nine of 20 images have F1 >= 0.8;
six are below 0.4.

## Decision

Keep **SAM automatic proposals + learned feature ranking** as the main line.
Do not yet promote a fixed DINO threshold, DINO-L, SAM-L, or colour fusion:
the current data show no reliable gain from those substitutions.

The next experiment should replace one-prototype thresholding with
**proposal-graph propagation**: start from the prompted proposal, create a
mutual-kNN graph of SAM proposal descriptors, add a background ring as negative
evidence, and retain a calibrated connected component before final NMS. This
targets the confirmed bottleneck (same-class proposal selection), while retaining
SAM's strong per-instance proposal coverage.

## SAM3 one-point experiment (new leading result)

SAM3 was then tested as a native **one point -> multi-instance mask** model.
Unlike the original SAM automatic-mask path, one positive click on image 01
directly returned nine separate candidate masks.  The final point-only
mechanism is deliberately small:

1. Send one positive click to SAM3 (no class text, no reference image, no
   external mask).
2. Keep SAM3's independent instance masks after score thresholding and mask
   NMS.
3. Reject only fragment masks whose area is below 0.1% of the input image.
   This is a class-agnostic quality floor, not a target-count or category rule.

| Setting | Scope | Instance F1 | Recall | Precision | Union IoU |
|---|---:|---:|---:|---:|---:|
| SAM3, raw click, threshold 0.50 | same-class 20 | 0.771 | 0.817 | 0.773 | 0.720 |
| SAM3, raw click, threshold 0.45 | same-class 20 | 0.786 | 0.878 | 0.746 | 0.713 |
| **SAM3 + fragment-quality floor** | **same-class 20** | **0.815** | **0.878** | **0.793** | **0.721** |
| SAM3 + DINOv2-B hard consistency filter | same-class 20 | 0.720 | 0.689 | 0.813 | 0.621 |
| SAM3 + DINOv2-B soft NMS ordering | same-class 20 | 0.815 | 0.878 | 0.793 | 0.721 |
| Same fixed SAM3 configuration, no retuning | random-category 20 | 0.596 | 0.652 | 0.640 | 0.583 |

The target benchmark (20 dense same-class banana images) therefore exceeds the
requested 0.8 F1 threshold.  However, the held-out random-category run shows
that the result does **not** yet generalize to arbitrary object categories:
SAM3 can still under-detect objects such as bowls/wheels and over-detect on
condiment/signboard scenes.  DINO hard filtering was tested and rejected
because it removes visually varied true instances.

The recommended research direction is now **SAM3 native candidate generation
plus seed-aware refinement**, not the earlier SAM-B/DINO proposal ranker.  A
next robust version should derive soft positive/negative compatibility from the
prompted candidate and use it only to resolve ambiguous SAM3 candidates; it
must preserve the native SAM3 recall rather than hard-filter it.

The soft-NMS variant lets DINO resolve the order of overlapping SAM3 candidates
without vetoing any candidate.  It preserved the target-benchmark result, while
the stronger DINO filter hurt recall.  Its 20 prediction-vs-GT sheets and a
contact sheet are under
`D:\SAM\runs\oneprompt_all\sam3_dino_soft_nms_viz20\`.

## Reproducibility

`research/evaluate_same_image_proposals.py` runs the current baseline. Large
multi-image runs are intentionally executed in short independent batches because
the machine has another process occupying most GPU memory.

`research/evaluate_sam3_onepoint.py` runs the SAM3 one-point evaluation.  The
best target-benchmark command is:

```powershell
& D:\SAM\conda_envs\sam_gpu\python.exe -B research\evaluate_sam3_onepoint.py `
  --dataset <same_class_20> `
  --checkpoint D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt `
  --threshold 0.45 --nms-iou 0.5 --min-area-ratio 0.001 `
  --out-dir D:\SAM\runs\oneprompt_all\sam3_onepoint_t045_area001
```

## Prompt-location robustness (five clicks per image)

The headline one-point result used a fixed reproducible click.  To measure the
actual interaction sensitivity, each of the 20 same-class images was then run
with five different points.  Each point was sampled from the interior of a
different target instance; the model still received only that single point.

| Method | 100 single-click trials | Mean F1 | F1 std | Mean P | Mean R | Mean within-image F1 std |
|---|---:|---:|---:|---:|---:|---:|
| SAM3 + fragment floor + DINO soft NMS | 20 x 5 | **0.827** | 0.204 | 0.790 | 0.913 | 0.073 |
| + DINO canonical re-click | 20 x 5 | 0.802 | 0.226 | 0.766 | 0.903 | 0.103 |

The multi-click mean remains above 0.8, but the standard deviation is material:
image 03 ranges from F1 0.000 to 0.941.  DINO canonicalization was a genuine
single-user-click refinement: SAM3 first produced a seed mask containing the
click, DINO selected a visually central token inside it, and SAM3 was called
once more.  It improved 16 / 100 click trials but worsened 23 / 100 (including
some already excellent points), so it is **rejected as a default**.

This points to a better next design: keep the original SAM3 click output as an
anchor, generate a small set of local virtual clicks only when SAM3 confidence
or candidate-count stability is low, then use DINO to *fuse agreement* across
the candidates rather than replacing the user point with a single canonical
point.  This should make DINO a conservative ambiguity resolver and preserve
the strong original-click outputs.

Reproducibility outputs:

- `D:\SAM\runs\oneprompt_all\sam3_dino_multiclick5\` (accepted multi-click baseline)
- `D:\SAM\runs\oneprompt_all\sam3_dino_canonical_multiclick5\` (rejected canonicalization ablation)

## DINO same-image propagation ablation

A second DINO mechanism was tested: from the clicked SAM3 seed mask, pool a
DINO prototype; take three separated high-similarity DINO patches outside the
seed as internal virtual queries; ask SAM3 at those locations; then aggregate
all returned instance candidates by NMS.  This respects the one-user-click
interaction contract, but is more aggressive than the soft-NMS use of DINO.

| Setting | F1 | Precision | Recall | Union IoU | Decision |
|---|---:|---:|---:|---:|---|
| Native SAM3 + quality floor + DINO soft NMS | 0.815 | 0.793 | 0.878 | 0.721 | keep |
| DINO propagation, three virtual points | 0.779 | 0.714 | 0.911 | 0.722 | reject |

The propagation gained recall but created too many false instances.  DINO is
therefore useful for *finding* possible siblings, but not yet for accepting
them.  The next version must impose a conservative agreement rule (for
example, accept a propagated candidate only when both an original-click SAM3
mask and a virtual-click SAM3 mask agree spatially, then use DINO only to pair
the two masks).  This prevents a single DINO high-similarity background patch
from expanding the final instance set.

## Automatic background evidence and relative DINO filtering

The workbench now uses two automatic DINO-selected background points.  On 400
independent valid-click trials this was the best tested background setting:
mean F1 **0.833**, precision **0.834**, recall **0.884**.  These are internal
model prompts, not additional user actions.

I also tested a seed-relative final-mask filter: retain a SAM3 candidate only
when its DINO cosine similarity lies within a fixed margin of the clicked seed.
The small 40-click ablation shows why it is not enabled by default:

| Method | Mean F1 | Precision | Recall | Decision |
|---|---:|---:|---:|---|
| Two DINO background points, no DINO veto | **0.855** | 0.868 | 0.878 | keep |
| Relative DINO margin 0.20 | 0.853 | **0.912** | 0.847 | reject as fixed default |
| Relative DINO margin 0.10 | 0.720 | 0.914 | 0.642 | reject |

The fixed filter reduces obvious wrong-category masks, but also removes true
instances with illumination, scale, or pose variation.  The next experiment
therefore needs an *adaptive agreement* criterion rather than any universal
DINO similarity threshold.

## Independent 200-image broccoli benchmark

`D:\SAM\dataset\200张\same_class_200` is an independent LVIS broccoli set
(200 images, 8--10 annotated target instances per image).  It was not used in
the earlier 20-image banana study.  Each image was evaluated with five
independently sampled interior GT points.  During inference the model received
only one point; GT was used solely for point simulation and scoring.

On an evenly spaced 20-image development slice (40 prompts), extra DINO
background points were harmful: fixed `n=0/1/2/4` yielded F1
`0.677/0.671/0.662/0.588`.  A contrast-gated adaptive policy with an upper
bound of six still selected 3.65 negatives on average and fell to 0.550.  The
cause is semantic: the exterior of a clicked broccoli often contains another
broccoli, not background.  The deployed dense-scene default is therefore
`n=0`, rather than the previously arbitrary fixed two points.

The selected configuration is SAM3 one-point candidates, DINO soft ordering,
no automatic negatives, and a 0.2% image-area fragment floor.  Its full
200 x 5 run is in
`D:\SAM\runs\oneprompt_all\broccoli200_final_5click_n0_area002\`:

| Images | Prompts | F1 | Precision | Recall | Mean per-image click std |
|---:|---:|---:|---:|---:|---:|
| 200 | 1,000 | 0.694 | 0.688 | 0.734 | 0.068 |

This is deliberately reported as a failed generalization target, not hidden:
28/200 images have mean F1 below 0.5 and only 60/200 reach 0.8.  It motivates
the next mechanism: DINO-guided positive propagation must be accepted only
when it agrees with an independently generated SAM3 candidate, rather than
using background negatives or a global DINO threshold.

## Candidate-set consensus attempts (not deployed)

To address prompt sensitivity and candidate acceptance directly, I evaluated a
three-response consensus: the original user point, a DINO-selected interior
seed point, and an Euclidean-distance interior seed point.  All virtual points
were constrained to the initial SAM3 seed, so no second user interaction or
external reference was used.  Candidate masks were IoU-clustered across
responses.

On the same 40-prompt broccoli development slice, accepting clusters observed
in at least two responses produced F1 0.656; accepting every cluster produced
0.668.  Neither exceeded the single-response 0.687 result.  Strict agreement
loses occluded/fragmented true instances; loose agreement restores them but
also restores false candidates.  The mechanism is retained as a diagnostic
ablation only (`research/evaluate_sam3_consensus.py`), not deployed.

As a recall-oriented fallback, SAM ViT-B automatic proposals plus DINO ranking
was also rechecked on ten broccoli images.  It obtained F1 0.338 and is not a
viable replacement for SAM3 native candidates.

The evidence now points to the concrete next mechanism: learn an
episode-local candidate graph with nodes = SAM3 masks and edges = DINO
appearance plus SAM3 overlap/stability.  Selection needs a *set-level*
objective (coverage of disjoint target-like regions, penalty for duplicate and
fragment nodes), calibrated on a held-out multi-category development set.  It
cannot be responsibly solved by another global threshold on this single
broccoli class.
