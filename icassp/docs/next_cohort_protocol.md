# Next-cohort feasibility and frozen protocol

## Decision

Do not use the locally present GlaS data as the next confirmatory cohort. The available strict patient-disjoint construction has 7 training cases from 3 patients and 1 validation case after excluding every official-train patient that overlaps the official test patients. This is insufficient for response-gate training or threshold selection. The calculation is frozen in `D:\SAM\runs\research_direction\prompt_policy_splits\glas_patient_disjoint\protocol_metadata.json` and no GlaS official-test image is read by this ICASSP package.

The next viable *field-level* external diagnostic is PanNuke. It is a public H&E nuclei dataset with three official folds, 256 by 256 RGB tiles, instance masks, tissue labels, and a CC-BY-NC-SA-4.0 distribution license. The public dataset card lists 7,901 tiles and 189,744 labelled nuclei; its three parquet folds total about 833 MB. Source: https://huggingface.co/datasets/RationAI/PanNuke (dataset card accessed 2026-08-09). It has no patient identifier in the public card, so this is explicitly not a patient-disjoint claim.

## Pre-registered zero-shot diagnostic

The protocol below is written before download or inspection of PanNuke pixel/mask values.

| Item | Frozen choice |
|---|---|
| Method | Existing DINOv2-L + SAM ViT-B foreground--background contrast. First emit model-independent N0/N5 records for the full cohort. Then use the five already saved nested MoNuSeg gates and their existing majority-vote rule, loading folds 0--1 under sklearn 1.7.2 and folds 2--4 under sklearn 1.9.0; no PanNuke fitting, threshold tuning, or post-processing selection. |
| Query set | Every PanNuke `fold1` tile with at least one union-mask foreground pixel. |
| Support bank | The first five non-empty `fold2` tiles in ascending SHA-256(image bytes) order. Each query is evaluated against all five, then aggregated to a query-level mean. |
| Target | Union of all instance masks, including every nuclear category; the dead-tissue category is excluded only when it has no nucleus instance mask. |
| Actions | The already frozen N0 / N5 candidates and the MoNuSeg-trained gate; no new candidate action is introduced. |
| Metric and inference unit | Binary Dice; mean over five supports is calculated before bootstrap over independent query tiles. |
| Comparator | FG-BG N0, fixed N5, and the frozen learned gate. No claim against FoB. |
| Statistics | 10,000 paired bootstrap resamples, seed 20260809. Report CIs and win/tie/loss at query-tile level. |
| Failure rule | If any dependency cannot consume 256 by 256 data without changing the frozen method, report `BLOCKED_INTERFACE` rather than resizing, retuning, or silently changing the protocol. |

## Claim boundary

A completed run can support only a zero-shot, field-level morphology/tissue-transfer statement. It cannot establish patient-disjoint generalization, a PanNuke-optimal policy, or broad clinical robustness. It will be added to the manuscript only if the entire frozen protocol executes and the paired evidence is internally consistent; otherwise the paper retains its current scope.

## Interface checkpoint — 2026-08-09

The frozen DINOv2-L + SAM ViT-B interface passed on one fixed fold2-to-fold1 pair using exactly `K=40`, `lambda=0.75`, N0, and N5. It verifies RGB/mask loading, 256-to-518 internal resizing, feature extraction, point coordinates, and SAM decoding. The generated one-pair values are deliberately classified as smoke-only in `evidence/pannuke_interface_smoke.json`; they do not contribute to a PanNuke mean, model choice, or manuscript claim. The raw full-cohort runner is memory bounded and persists only N0/N5 measurements. The later gate step is version-isolated because the frozen artifacts were produced by two sklearn releases; its decision inputs are the two SAM scores, the two predicted areas, and the fixed action difference, never Dice or a query mask. See `evidence/pannuke_gate_environment_audit.json`.
