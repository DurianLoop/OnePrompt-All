# AdaPromptSet Open Questions and Gate Extensions

> Purpose: this document lists the remaining uncertainties before a full goal-mode implementation. It also records how to extend experiments when a gate fails, without tuning on already observed test data or overstating the claim.

Date: 2026-07-17
Workspace: `D:\SAM`

Status update after user/Codex decision file: `AdaPromptSet_Codex_Response.md` resolves the confirmation items below for goal-mode execution. This document remains useful as a historical uncertainty register and as the gate-failure extension policy, but open questions in Section 2 should no longer block Phase 0/1 unless new local evidence contradicts the resolved answer.

Resolved execution answers:

- Main target: attempt adaptive policy, but accept deterministic/action-oracle/oracle-analysis pivots when gates reject policy learning.
- Primary data: MoNuSAC main development; MoNuSeg sanity/historical; GlaS first clean extension.
- IDRID: read-only/non-tunable boundary result; no new policy or threshold decisions may use it.
- DDR: Stage 1 lesion-all union infrastructure only; Stage 2 four-intent protocol is conditional extension.
- CPM-17: deferred until main evidence is strong and complete data are available.
- Compute: MVP review budget capped at 120 GPU-hours with per-phase ledgers.
- Segmenter: MVP uses DINOv2-L + SAM-B only.
- External local data: allowed after manifest/license/source/hash/exposure logging; no network without explicit approval.
- Paper: experiment package first, claim-free skeleton until gates produce evidence.

New evidence from Phase 1 audit:

- GlaS has the expected 85 train / 60 testA / 20 testB source split, but patient-disjoint official-test-preserved roles leave only 8 role=train cases and 77 source-train cases marked `excluded_overlap`. Treat GlaS as a strict confirmatory morphology-shift extension, not as a robust early policy-training source, unless a later predeclared protocol justifies a different statistical unit without test leakage.

---

## 1. Local Dataset Inventory

| Dataset/source | Local path | Observed state | Best use | Blocker before formal use |
|---|---|---|---|---|
| MoNuSAC | `code\datasets\MoNuSAC\processed` | 1264 files; already used by FSPL fixed-rule and policy experiments | MVP main development and pilot oracle-gap | Rebuild AdaPromptSet manifest and preserve prior test-observation boundaries |
| MoNuSeg | `code\datasets\MonuSeg\official` | 286 files, 190 image-like files | sanity check, historical alignment, small cross-check | Confirm case/patient IDs and official split handling |
| GlaS | `external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` | train 85 / testA 60 / testB 20 image-mask pairs confirmed | first morphology-shift extension | Run or migrate `build_glass_strict_policy_splits.py`; confirm license/source hash |
| DDR lesion segmentation | `code\ddr_datasets\DDR-dataset-01\lesion_segmentation` | train 383, valid 149, test 225 images; EX/HE/MA/SE labels | eye-lesion boundary test and domain-specific candidate study | Decide lesion-all vs four-intent protocol; define mask merge and metrics |
| IDRID | `code\datasets\IDRID` | train 55 images/54 masks, test 27/27 | recorded boundary test and comparison to older negative result | Do not retune on observed test; decide if any fresh split remains valid |
| nnU-Net Dataset902 | `runs\nnunet\nnUNet_raw\Dataset902_OCSAMPhase3Stratified` | 80 training cases; IDRID + MonuSeg stratified derivative | supervised strong baseline / sanity comparator | Keep it read-only for comparison; do not leak into policy feature selection |
| CPM-17 | `external\research_data\raw\CPM17` | only `cpm15` 15 images + 15 `.mat` labels confirmed; train/test folders incomplete | future external nuclei extension | Fix or redownload to match actual manifest; parse `.mat` instance labels |
| FSS-1000 / COCO2014 | `code\datasets\FSS-1000`, `code\datasets\COCO2014` | local split stubs only in current scan | non-medical matcher sanity only | Not suitable for medical paper claims |

---

## 2. Points Needing User Confirmation

1. Main scientific target:
   - Should the first full goal pursue a positive adaptive policy result, or is a bounded fixed-rule/action-oracle paper acceptable if gates reject policy learning?

2. Primary dataset:
   - Should MoNuSAC remain the main development dataset, with GlaS as first extension, or should GlaS become the main policy-learning target because it has a clean 85/60/20 official structure?

3. IDRID usage:
   - Is IDRID allowed only as a recorded negative/boundary result, or do you want a fresh strictly separated protocol if one can be justified from unobserved data?

4. DDR protocol:
   - Should DDR labels be merged into one lesion foreground mask, or treated as four separate intents: EX, HE, MA, SE?

5. CPM-17:
   - Should we spend effort repairing/downloading CPM-17, or drop it until the main AdaPromptSet evidence is already strong?

6. Claim tolerance:
   - If the policy does not beat the best fixed action but a deterministic action is stable, should the paper pivot to deterministic spatial/layout prompting?

7. Compute budget:
   - What is the maximum acceptable pilot/full run budget in GPU-hours before a gate should stop expansion?

8. Segmenter scope:
   - Should MVP stay DINOv2-L + SAM-B only, or should SAM-L/H replication be included in the first goal from the beginning?

9. External data policy:
   - May the goal mode use existing local `external\research_data` and, if CPM-17 is incomplete, request network/download approval?

10. Paper target:
   - Is the desired output an experiment package first, or a complete paper artifact only after positive gates pass?

---

## 3. Gate Extension Rules

### Gate A: Data Integrity Fails

Allowed rescue:
- Build a strict manifest repair script.
- Exclude incomplete datasets from the formal table.
- Move the dataset to exploratory appendix only.

Not allowed:
- Treat unpaired images as valid image-mask pairs.
- Infer patient isolation when metadata is missing.
- Use official test images as reference pool.

### Gate C: Candidate Quality Fails

Allowed rescue:
- Add domain-specific candidate generators on train/validation only.
- For pathology: multi-prototype foreground/background, local density peaks, stain-normalized features.
- For eye lesions: color/brightness-aware candidates and lesion-class-specific masks.
- Compare DINOv2-B/L and local pathology encoders as matcher ablations.

Stop condition:
- If candidate precision is not above random and coverage does not improve with pool size after one redesign, do not train policy on that dataset.

### Gate D: Segmenter Execution Fails

Allowed rescue:
- Validate GT-center privileged prompts.
- Switch SAM-B/L/H only under the same frozen protocol.
- Try SAM2 only as a separately costed extension.

Not allowed:
- Hide segmenter failures by selecting only easy visual cases.
- Mix different fusion/postprocess rules across methods.

### Gate F: Oracle Gap Fails

Allowed rescue:
- Expand action bank once with predeclared actions: spread/radius variants, group-specific negatives, graph groups, safe anchors, GF-SAM-style grouping.
- Repeat pilot on GlaS or DDR if the failure may be dataset-shape specific.

Current status:
- 2026-07-17: The one-time MoNuSAC Rescue-F1 subset has already been executed as `prompt_set_bank_v2` with split/group-specific-negative actions. It still failed Gate F (`dice_oracle_gap=0.0050`, below the authoritative 0.010 threshold), so no additional MoNuSAC policy training or post-hoc action redesign is allowed under the current evidence. The Phase 5 checkpoint status is `PIVOT_ORACLE_ANALYSIS`; the remaining valid options are a predeclared new-domain pilot (for example GlaS/DDR) or a fixed-rule/action-oracle pivot.
- 2026-07-17: Fixed-rule/action-oracle pivot analysis has been executed in `reports/fixed_oracle_analysis/`. It does not promote a deterministic method: best fixed is `B_top5_pos3neg`, mean oracle gap is 0.005011, near-oracle episode fraction is 0.6, top oracle action fraction is 0.5, and recommended claim is `oracle_analysis_only`.
- 2026-07-17: A predeclared strict GlaS validation smoke has been executed in `reports/candidate_pilot_glas/` and `reports/oracle_gap_pilot_glas/`. Candidate quality passed on one strict validation episode, but Gate F remains `PENDING_INSUFFICIENT_EPISODES` because the run has 1 episode and the gate minimum is 10. This evidence only proves that the extension path runs; it cannot reopen policy training or support a GlaS formal claim.

Pivot:
- If fixed actions are near-oracle, convert the paper to fixed-rule or action-oracle analysis.
- If one action dominates oracle, promote it as deterministic method and stop complex policy training.

### Gate H: Policy Learning Fails

Allowed rescue:
- Train simpler binary/ordinal gates: use spread vs top-k, add negatives or not, low/high budget.
- Try regression/pairwise/listwise once.
- Add domain-conditioned heads only as a controlled ablation.

Pivot:
- If oracle gap exists but policy cannot rank actions, write representation/oracle analysis.
- If policy only learns action priors, report a calibrated action classifier rather than a set-utility predictor.

### Gate I: Formal Test Fails

Allowed rescue:
- Run predeclared external validation on a dataset not used for method selection, such as GlaS if still untouched.
- Report failure subsets and domain mismatch.

Not allowed:
- Retune action bank, budget, thresholds, candidate generation, or policy architecture on the failed test.
- Re-sample reference-query pairs from the same observed test and claim a larger sample size.

---

## 4. Recommended Goal-Mode Starting Scope

If there are no objections, the safest full goal is:

1. Build `AdaPromptSet-MVP` as an independent package under `D:\SAM\adapromptset`.
2. Migrate/reuse FSPL modules instead of rewriting.
3. Build strict manifests for MoNuSAC, MoNuSeg, and GlaS.
4. Run audit/smoke on MoNuSAC and GlaS.
5. Run 10-20 query oracle-gap pilots.
6. Decide whether to continue policy learning, pivot to deterministic fixed-rule, or expand the action bank.

This gives the project a real escape hatch: it can still produce a defensible result even if the adaptive policy hypothesis fails.
