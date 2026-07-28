# V4 Independent Integrity Audit

Date: 2026-07-28

Scope: `paper/phaseU/full_draft_v4.pdf`, its archived V4 material in `VV4/`, and the corresponding active code in `src/adapromptset/scripts/`.

## Decision

**`V4_NOT_SUBMISSION_SAFE; V4_USEFUL_ONLY_AS_AUDITED_RESEARCH_SCAFFOLD`**

The PDF is a real, internally consistent artifact: the live and archived V4 PDFs have the same SHA-256 (`36d5...3e44`), and many reported values can be read back from archived result files. This does **not** validate the paper's central method or its stronger scientific claims. The available evidence establishes artifact replay, not an independent end-to-end reproduction from frozen raw inputs.

There is no direct evidence that numbers were invented as arbitrary constants. There is, however, direct code evidence that key V4 method results are based on simulated per-reference Dice and on post-hoc features computed from already evaluated predictions. That makes the active-intervention and deployment claims invalid as written.

## What Is Real And Potentially Worth Keeping

| Asset | Audit verdict | Permitted use |
|---|---|---|
| Reference reliability estimation (RRE) problem | Plausible and useful, but novelty needs literature verification | Motivation / benchmark framing |
| Reference-resampling protocol | Traceable in code and result CSVs | Development benchmark protocol |
| DINOv2-L + SAM-B top-K Dice around 0.446 | Artifact replay match (`0.446243...`) | Historical development result, not independent test evidence |
| Reference-dependent variability and catastrophic episodes | Supported by archived episode-level outputs | Failure characterization with an explicit development-only qualifier |
| Prediction Consistency follow-up | A separate later direction, not evidence for V4 | Candidate replacement subject to a fresh audit |

## Blocking Findings

### P0-1: RAMP's claimed intervention is not actually executed

In `run_phaseU_ramp.py`, `evaluate_ramp_on_episodes()` selects a prompt configuration and reference budget but consumes the precomputed `best_config` rows. It does not rerun the matcher or segmenter after choosing the supposedly adaptive configuration. The code then constructs the value being fused with:

```python
per_ref_dice = np.random.normal(mean_dice, std_dice, len(features))
```

(`VV4/02_CODE_ORIGINAL/exact_v4_code/run_phaseU_ramp.py:267`). The ablation path does the same around line 363. Consequently, reported Dice changes for weighting, fusion, prompt adaptation, and reference budgeting are not measurements of those executed actions.

The project’s own repair script explicitly records the defect: it says the original component ablation simulated per-reference Dice rather than using the raw CSV (`run_phaseV_fix_component_ablation.py:1-14`). The original RAMP result JSON also shows exactly identical Dice for all five ablation variants in every pipeline. These figures and Table 8 must not be cited.

### P0-2: RAMP is a post-hoc, in-sample selective-prediction analysis

The method loads `phase2_episode_instability.csv`, whose features include prediction consistency and Dice-derived labels after SAM has already run. The two gradient-boosting risk models are fit on the same episode table later used for evaluation. The available implementation has no independent train/validation/test partition or nested evaluation for the primary RAMP claim.

Therefore it cannot support an online, training-free, calibration-free, or prospective reference-selection claim. At most, it supports an exploratory post-hoc ranking analysis on the development cohort.

### P0-3: The PDF's central universality claim is contradicted by later analysis

The V4 source calls the anti-correlation a “universal property” (`paper/phaseU/full_draft.tex:113-117`) and presents a universality section (`:664`). Later query-level results retained in VV4 find significance only for SAM-B pipelines; SAM-L fails Holm correction and both SAM2 pipelines are non-significant. The null analysis also reproduces 94% of the observed correlation with a random-beta construction.

This does not mean the association is false everywhere. It means the V4 interpretation as a structural, architecture-wide mechanism is not supported. Any salvaged paper must state it as a pipeline-specific empirical association and make the null result central rather than incidental.

### P0-4: Several comparisons are absent, proxy-only, or not comparably configured

CycleSAM, H-SAM, and SAM-Med2D numbers lack reproducible prediction artifacts; MC Dropout and Deep Ensemble are explicitly proxies; GF-SAM and MAUP use SAM-H while V4's method uses SAM-B. The cross-domain natural-image evidence is only fold 0, not the advertised full four-fold protocol.

No ranking, SOTA, “identical conditions,” or external-generalization claim may survive from these rows.

## Major Provenance And Reproducibility Risks

| Finding | Evidence | Consequence |
|---|---|---|
| PDF archives are byte-identical | Direct SHA-256 check | Good archival integrity for the PDF only |
| Current `.tex` is not proven to be the source of that PDF | `.tex` modification time is later than PDF; the VV4 decision itself says the ZIP source differs from current `.tex` | Do not claim source-to-PDF reproducibility |
| 60 “reproductions” are mostly artifact replay | `WP8` calls 55 results `Level1_ArtifactReplay`; five inaccessible | Do not call them independent replication |
| Claim registry is not an evidence registry | 48/92 claims are class D; the matrix has no code/data/report artifact for early core claims such as C-V4-0001 to C-V4-0003 | Claim count must not be mistaken for validation coverage |
| Archive is not standalone executable | `run_phaseU_wp1_segmenter_audit.py --help` fails: missing `run_phaseT_wp1_holdout` | Exact code needs repository context and a frozen environment manifest |
| Archived and active RAMP/WP1 scripts hash-identically | Direct SHA-256 checks | The defects above are in the claimed V4 execution code, not introduced by the archive |

## Evidence Grade By Claim Type

| Claim type | Grade | Boundary |
|---|---|---|
| “The V4 PDF and result artifacts exist” | A | Directly established |
| “Some historical numbers match archived CSV/JSON” | B | Artifact replay only |
| “Reference choice can matter in the V4 development cohort” | B/C | Needs a frozen cohort and end-to-end replay for stronger use |
| “RAMP actively improves segmentation/prompting” | F | Directly invalidated by code path and simulation |
| “Anti-correlation is universal/structural” | F | Contradicted by later pipeline-specific analysis and null control |
| “External methods are fairly compared” | D/F | Missing data, proxies, and backbone mismatch |
| “Cross-domain generalization” | D | Only a single fold is available |

## Minimum Gates Before Any Rebuilt Paper

1. Freeze a commit/tag containing paper source, raw data manifests, checkpoint hashes, environment lockfile, and deterministic episode manifests. Build the PDF in that environment and record its hash.
2. Re-run one prespecified primary pipeline from raw images and masks to per-query predictions, without reading any final-test labels during model/risk design. Save raw predictions and per-call provenance.
3. Replace all simulated Dice and report-only decisions with executed matcher/SAM calls. Compare every intervention against the identical action with only one factor changed.
4. Evaluate risk selection with grouped, nested splits at the query/patient level. Features generated from masks may be used only under an explicit post-hoc protocol.
5. Hold out a fresh confirmation cohort before selecting thresholds, variants, figures, or narrative. Report uncertainty intervals at the query/patient unit.
6. Re-run natural-image evaluation over all four folds, or remove it. Keep only external comparators for which predictions, checkpoints, and backbone-matched protocols are available.
7. Write the paper only from a claim ledger that maps each sentence/table/figure to a frozen artifact and labels it `confirmatory`, `development`, `proxy`, or `unavailable`.

## Recommended Research Position

Do not repair V4 by editing qualifiers around the existing tables. The honest route is a new, narrower paper: a benchmark/failure-boundary study of reference sensitivity in frozen visual prompting, with RRE as the problem formulation. A method paper is justified only after a real intervention passes the gates above on fresh data.

The current Phase 4+ DDR work is a better fit for this discipline: it already separates automatic rows from privileged controls and keeps formal test access locked. Its results should not be merged with V4 until they have an explicit common protocol and a new preregistered claim ledger.
