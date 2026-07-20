AdaPromptSet_Codex_Response.md


# AdaPromptSet Decisions and Implementation Directives

> **Purpose:** Formal response to the remaining open questions before goal-mode implementation. This document is authoritative for Codex unless a later written instruction supersedes it.
>
> **Date:** 2026-07-17  
> **Workspace:** `D:\SAM`  
> **Target package:** `D:\SAM\adapromptset`

---

## 0. Executive Decision

Proceed with **AdaPromptSet-MVP** as a gated and reproducible experiment package.

### Current empirical status overlay

This document was written as the execution directive before goal-mode implementation. As of the current local evidence, Gate F has already been evaluated on the MoNuSAC MVP action bank and the one allowed rescue bank. Both runs failed the minimum oracle-gap criterion, and `reports/decision_checkpoint/phase5_decision.md` records:

```text
PIVOT_ORACLE_ANALYSIS
```

Therefore, the "pursue adaptive-policy hypothesis first" instruction below should be read as the original gated route, not as permission to continue MoNuSAC policy training after Gate F failure. The currently allowed MoNuSAC directions are:

- fixed-rule / action-oracle analysis;
- representation analysis of why the action bank is near-oracle;
- separately predeclared new-domain pilots, such as GlaS, without reopening MoNuSAC policy training.

The currently prohibited directions are:

- policy training on MoNuSAC;
- post-hoc MoNuSAC action-bank redesign based on the observed failed Gate F result.

The primary hypothesis remains:

> A lightweight reference-conditioned utility model can select a structured positive/negative prompt set that gives a better segmentation-quality–cost trade-off than fixed prompt rules under the same frozen matcher, segmenter, candidate pool, reference episodes, and SAM-call budget.

A positive policy result is the main target, but it is not a required outcome. Three final directions are acceptable:

1. **Adaptive-policy paper:** the learned selector beats the strongest fixed action at matched cost and recovers a meaningful fraction of the oracle gap.
2. **Deterministic-method paper:** one auditable fixed action is stable and near-oracle, so the complex policy is removed.
3. **Oracle/representation analysis:** a useful oracle gap exists, but current observable features cannot predict the best action reliably.

No test-driven redesign is allowed. The final claim must follow the gate results rather than the desired narrative.

---

# 1. Decisions on the Ten Open Questions

## 1.1 Main Scientific Target

**Decision:** pursue the adaptive-policy hypothesis first, while registering deterministic and oracle-analysis pivots in advance.

The project must separate four questions:

1. Does query-specific prompt allocation matter?
2. Is there a meaningful action oracle?
3. Can test-time-safe features predict useful actions?
4. Does the learned or deterministic result survive a frozen test?

Required routing:

| Result | Direction |
|---|---|
| Oracle gap exists and policy recovers it | Continue AdaPromptSet policy |
| Oracle gap exists but ranking fails | Pivot to representation/oracle analysis |
| One fixed action is near-oracle | Pivot to deterministic prompting |
| No oracle gap after one approved redesign | Stop policy learning on that dataset |

Do not redefine success after observing formal test results.

---

## 1.2 Primary Dataset

**Decision:** assign different scientific roles instead of forcing one dataset to serve every purpose.

| Dataset | Role |
|---|---|
| **MoNuSAC** | main development dataset, candidate/action design, pilot oracle-gap, ablations |
| **GlaS** | first clean morphology-shift extension and preferred confirmatory dataset |
| **MoNuSeg** | sanity check and historical alignment |
| **DDR** | conditional domain-boundary extension after pathology gates pass |
| **IDRID** | recorded non-tunable boundary/negative result |
| **CPM-17** | deferred external nuclei extension |

MoNuSAC remains the development dataset because the local implementation already exists and dense nuclei create a genuine prompt-allocation problem. However, prior MoNuSAC observations must be documented. Previously inspected cases or evaluation partitions cannot be represented as a clean untouched test.

GlaS should remain untouched as long as possible. Verify whether testA or testB has ever influenced previous decisions. If both are untouched, use the official train set for fitting/development, testA for the first frozen evaluation, and testB as the stricter morphology-shift evaluation.

---

## 1.3 IDRID Usage

**Decision:** IDRID is read-only, non-tunable evidence.

Do not use observed IDRID test behavior to choose:

- candidate generators;
- negative-point rules;
- action-bank entries;
- budget thresholds;
- policy features;
- model architecture;
- reference sampling.

A new confirmatory IDRID protocol is allowed only if an entirely unobserved case-level subset can be proven and documented in `protocol_history.md` and `data_exposure_log.md`. If this cannot be demonstrated, IDRID may appear only as a historical negative result, robustness boundary, or exploratory appendix result.

---

## 1.4 DDR Protocol

**Decision:** use two stages.

### Stage 1: lesion-all union, infrastructure only

Merge EX, HE, MA, and SE into one binary foreground mask only for:

- loader and mask-merge validation;
- coordinate tests;
- candidate diagnostics;
- segmenter smoke tests;
- runtime estimation.

Do not use this union task as the preferred formal scientific claim.

### Stage 2: four reference-defined intents

Treat EX, HE, MA, and SE as separate tasks. For each intent:

- the reference mask contains only that lesion class;
- the query target is the same class;
- metrics are reported per class and macro-averaged;
- lesion-size-stratified results are required;
- empty-target handling is fixed before formal evaluation.

Begin with a **positive-query protocol** containing queries with at least one target lesion. A mixed-present/absent protocol may be added later with an explicit no-target/no-prompt action.

---

## 1.5 CPM-17

**Decision:** defer repair and download.

Do not spend MVP effort on CPM-17 until:

- the MoNuSAC/GlaS candidate and oracle gates pass;
- the policy or deterministic direction is known;
- the main evidence is reproducible;
- the source and license are confirmed;
- network access is explicitly approved.

The local `cpm15` subset may be used only for parser development or exploratory visualization. It must not be called formal CPM-17 evaluation.

---

## 1.6 Claim Tolerance

**Decision:** a deterministic pivot is acceptable and preferred over retaining unnecessary policy complexity.

Promote a deterministic method when validation shows that:

1. one action is oracle-selected for at least 70% of queries, **or**
2. one action is within 0.5 Dice points of the per-query oracle;
3. the learned policy fails to improve that action by at least 0.5 Dice points at equal cost; and
4. the result is stable across multiple reference seeds or fixed reference sets.

A deterministic paper may claim an efficient structured prompt rule or an oracle study showing that adaptation is unnecessary in the tested setting. It may not claim adaptive policy learning.

---

## 1.7 Compute Budget

Use the following default ceilings unless manually changed before execution:

| Phase | Maximum |
|---|---:|
| Data audit and smoke tests | 2 GPU-hours total |
| Candidate-quality pilot | 6 GPU-hours per dataset |
| Initial 10–20 query oracle pilot | 12 GPU-hours per dataset |
| One action-bank redesign and rerun | additional 12 GPU-hours per dataset |
| Policy pilot | 16 GPU-hours total |
| Full MoNuSAC run | 48 GPU-hours |
| GlaS extension/formal run | 36 GPU-hours |
| One matcher or segmenter extension | 24 GPU-hours |
| **MVP total before review** | **120 GPU-hours** |

Every run must log:

- estimated and actual GPU-hours;
- GPU model;
- wall-clock time;
- peak VRAM;
- matcher calls;
- segmenter calls;
- cache reuse.

Do not start SAM-L/H, SAM2, DDR, or CPM-17 merely because compute remains available.

---

## 1.8 Segmenter Scope

**Decision:** the MVP uses only:

> **DINOv2-L matcher + SAM-B segmenter**

Use this pair for candidate construction, action-bank development, oracle-gap analysis, policy training, and the main ablations.

After the central gates pass, add only one of the following first:

1. SAM-L or SAM2 replication; or
2. one alternative matcher.

SAM-H and a large matcher × segmenter matrix are not MVP requirements.

Every extension must state whether the policy is:

- retrained for the new combination;
- calibrated with target validation labels;
- or transferred frozen.

“Plug-compatible” must not be presented as frozen policy transfer.

---

## 1.9 External Data and Network Policy

Existing local data under `D:\SAM\external\research_data` may be used only after:

- source and license are recorded;
- image-mask pairing is verified;
- file and manifest hashes are saved;
- prior test exposure is documented.

No network access, dataset download, model download, or repository cloning is allowed without explicit approval.

When an asset is missing:

1. write a missing-asset report;
2. record the expected source and structure;
3. stop that branch;
4. continue with available approved assets.

---

## 1.10 Output Priority

**Decision:** produce the reproducible experiment package first, while maintaining a claim-free paper skeleton.

### Before the oracle gate

Required outputs:

- package structure;
- strict manifests;
- audit reports;
- candidate and action code;
- smoke tests;
- pilot cache;
- gate reports;
- LaTeX skeleton without performance claims.

### After the oracle gate

Add:

- oracle-gap tables and figures;
- action-selection analysis;
- frozen policy-training or deterministic-pivot plan.

### After the policy/deterministic gate

Add:

- full validation results;
- ablations;
- reference sensitivity;
- cost analysis;
- qualitative successes and failures.

### After frozen formal test

Produce:

- final tables and figures;
- complete two-column paper;
- compiled PDF;
- reproducibility appendix;
- exact commands and environment information.

Placeholder or simulated values must never be mixed with real experimental results.

---

# 2. Confirmed Scientific Scope

Given a labeled reference pair \((R,M_R)\), query image \(Q\), frozen matcher \(F\), frozen segmenter \(S\), and budget \(B\), select:

\[
A = \{G_1,\dots,G_K\},
\qquad
G_k=(P_k^+,P_k^-),
\]

where each prompt group causes one segmenter call.

The decision is:

\[
A^*
=
\arg\max_{A\in\mathcal A,\ C(A)\le B}
\hat U(A\mid R,M_R,Q).
\]

Here:

- \(\mathcal A\) is a finite, predeclared action bank;
- \(C(A)\) records prompt and call cost;
- utility labels use training/validation ground truth only;
- formal test ground truth is never used for action selection or redesign.

## MVP exclusions

Do not implement initially:

- unrestricted coordinate generation;
- learned coordinate offsets;
- continuous clustering-parameter prediction;
- PPO or DQN;
- iterative SAM-response-conditioned prompting;
- universal transfer claims;
- domain-conditioned heads;
- CPM-17 download;
- SAM-H matrix experiments.

---

# 3. Required Package Structure

Create:

```text
D:\SAM\adapromptset\
├── README.md
├── pyproject.toml
├── configs\
│   ├── data\
│   ├── matcher\
│   ├── segmenter\
│   ├── candidates\
│   ├── actions\
│   ├── policy\
│   └── experiments\
├── adapromptset\
│   ├── data\
│   ├── matching\
│   ├── candidates\
│   ├── actions\
│   ├── segmenter\
│   ├── fusion\
│   ├── utility\
│   ├── policy\
│   ├── metrics\
│   ├── statistics\
│   └── reporting\
├── scripts\
├── tests\
├── manifests\
├── cache\
├── runs\
├── reports\
├── paper\
└── docs\
```

Reuse verified FSPL modules through imports or copied modules with provenance. Do not modify the existing FSPL package in place except for isolated, tested bug fixes.

---

# 4. Mandatory Reproducibility Artifacts

Maintain:

```text
docs/protocol_history.md
docs/decision_log.md
docs/data_exposure_log.md
reports/gate_status.md
reports/compute_ledger.csv
reports/dataset_audit/*.json
reports/dataset_audit/*.md
manifests/*.csv
manifests/*.sha256
```

Every run must save:

- resolved configuration;
- git commit hash;
- environment metadata;
- random seeds;
- dataset manifest hash;
- reference-query episode list;
- action-bank version and hash;
- candidate-generator version;
- matcher and segmenter identifiers;
- query-level metrics;
- selected prompt groups;
- requested and valid prompt counts;
- segmenter calls;
- wall-clock runtime;
- peak VRAM;
- output-mask paths or cache identifiers.

Aggregated scores alone are insufficient.

---

# 5. Gate Definitions

## Gate A — Data Integrity

Pass only when:

- all formal cases have verified image-mask pairs;
- duplicate IDs/content are detected;
- image and mask dimensions are checked;
- label semantics are documented;
- train/validation/test overlap is zero at the strictest available identity level;
- reference pools contain no formal test case;
- manifests and hashes are saved.

Allowed rescue:

- strict manifest repair;
- case exclusion;
- downgrade to exploratory use.

Prohibited:

- loose filename pairing;
- inferred patient isolation without evidence;
- official test images as references;
- missing masks replaced with empty masks;
- incomplete data presented as a full dataset.

---

## Gate B — Protocol and Exposure Integrity

Before modeling:

- record all previous test observations;
- freeze dataset roles;
- freeze reference sampling;
- create immutable episode manifests;
- assign a protocol version.

If exposure cannot be bounded, the dataset becomes development-only and cannot be described as an untouched test.

---

## Gate C — Candidate Quality

Required positive diagnostics:

- point precision;
- enrichment over random foreground prevalence;
- component/instance coverage;
- coverage versus pool size;
- spatial redundancy;
- distance to target components.

Required negative diagnostics:

- background precision;
- hard-background rate;
- target-boundary distance;
- false-negative risk;
- concentration in obvious empty background.

Default pilot warning thresholds:

- top-candidate precision should exceed random foreground prevalence by at least 15 percentage points or 1.5×;
- maximum-pool component coverage should reach at least 60%.

One redesign is allowed:

- pathology: multi-prototypes, local-density peaks, stain-robust features, DINOv2-B/L, one available pathology encoder;
- eye lesions: color/brightness-aware and multiscale candidates.

If precision remains near random and coverage does not improve after one redesign, stop policy work on that dataset.

---

## Gate D — Segmenter Execution

Run privileged diagnostics:

- GT-center prompts;
- GT boxes where meaningful;
- controlled GT positive/negative prompts;
- random and mismatched controls.

Verify:

- coordinate transforms;
- point labels;
- one logged segmenter call per prompt group;
- fixed multimask selection;
- fixed fusion;
- failed-call accounting.

Pass when privileged prompts produce broadly valid target masks and outperform random/mismatched controls. Do not impose one universal Dice threshold.

Allowed rescue:

- repair transforms;
- compare SAM-B with one alternative scale;
- test SAM2 as a separately costed extension.

Prohibited:

- cherry-picking easy images;
- method-specific postprocessing;
- dropping failed calls.

---

## Gate E — Action-Bank Validity

The first action bank should contain a compact set:

- Top-1, Top-3, Top-5, Top-10;
- local-maxima spread;
- fixed positive plus negative;
- KMeans groups;
- group-specific negatives;
- GF-SAM-style graph grouping if reproducible;
- one safe anchor;
- one low-cost and one high-coverage action.

Every action must save:

- positive points;
- negative points;
- groups;
- requested and valid points;
- number of calls;
- parameters;
- actual cost.

All non-public-baseline actions must use the same candidate evidence, segmenter, fusion, and budget accounting.

---

## Gate F — Oracle Gap

For query \(q\):

\[
A_q^{oracle}
=
\arg\max_{A\in\mathcal A,\ C(A)\le B}
M(q,A).
\]

Continue to policy learning only if the validation pilot shows:

1. at least **1.0 Dice point** mean oracle improvement over the strongest global fixed action at matched budget; and
2. at least **25% of queries** prefer an oracle action different from the globally strongest fixed action.

One action-bank expansion is allowed, limited to predeclared categories:

- spread/radius variants;
- group-specific negatives;
- graph groups;
- safe anchors;
- GF-SAM-style grouping;
- one missing cost regime.

Pivot rules:

- fixed near oracle → deterministic method;
- one action dominates → promote it and stop policy work;
- no gap after redesign → stop adaptive work on that dataset;
- gap only on another morphology/domain → narrow the claim accordingly.

---

## Gate G — Utility-Label Reliability

Before policy training, analyze:

- action rank stability across repeated runs;
- utility variance across references;
- fraction of near-tied actions;
- best-versus-second-best margins;
- metric sensitivity;
- failed-output effects.

If rankings are unstable:

- aggregate multiple references;
- use tied or ordinal labels;
- predict coarse decisions;
- report uncertainty.

Do not train a high-capacity ranker on noisy utility labels.

---

## Gate H — Policy Learning

Start with a lightweight test-time-safe model using:

- matching-score statistics;
- foreground/background contrast distribution;
- candidate counts;
- dispersion and density;
- confidence margins;
- prompt-group geometry;
- action cost;
- reference-query feature gap.

Training order:

1. global action prior;
2. multiclass action classifier;
3. pointwise utility regression;
4. pairwise ranking;
5. listwise ranking only if justified.

Policy success on validation requires:

1. at least **0.5 Dice point** improvement over the strongest fixed action at matched cost;
2. at least **30% oracle-gap recovery**;
3. no hidden increase in calls or valid prompts;
4. stability across reference seeds or fixed reference aggregation.

\[
Recovery
=
\frac{M_{policy}-M_{fixed}}
{M_{oracle}-M_{fixed}}.
\]

One rescue sequence is allowed:

- binary/ordinal gates;
- regression versus pairwise versus listwise once;
- domain-conditioned head only as a final controlled ablation.

Failure routes:

- oracle exists but ranking fails → representation/oracle analysis;
- only priors are learned → calibrated action classifier;
- fixed action remains strongest → deterministic pivot;
- GT-correlated features are required → reject the policy.

---

## Gate I — Frozen Formal Test

Before test, freeze and hash:

- code commit;
- configs;
- action bank;
- candidate generators;
- policy checkpoint;
- budgets;
- metrics;
- episode/reference manifests;
- fusion and postprocessing;
- statistical plan.

Run the predeclared formal evaluation once.

If it fails, allowed actions are:

- report failure;
- evaluate a previously declared untouched external dataset;
- analyze failure subsets;
- downgrade the claim.

Not allowed:

- retuning;
- action-bank edits;
- candidate changes;
- architecture changes;
- favorable seed selection;
- test resampling;
- failed-case removal;
- retroactive validation relabeling.

---

# 6. Dataset Directives

## MoNuSAC

Use for development, pilot utilities, ablations, and method analysis. Rebuild the manifest and document all prior exposure. Use query/case—not patch—as the statistical unit.

## GlaS

Run or migrate `build_glass_strict_policy_splits.py`. Verify the 85/60/20 pairs, official split identities, license, hashes, and prior exposure. Preserve testA/testB as untouched whenever possible.

## MoNuSeg

Use for loader/coordinate sanity and historical cross-check. Do not assume patient isolation from filenames.

## DDR

Do not start formal DDR policy work before pathology Gate F and Gate H decisions. Begin with union-mask infrastructure tests, then move to the four-intent protocol.

## IDRID

Read-only. It must not influence method selection.

## nnU-Net Dataset902

Use only as a supervised comparator. Do not leak its learned features, predictions, masks, or split-specific behavior into AdaPromptSet feature/action selection.

## CPM-17

Deferred. Create only a parser TODO, missing-data report, and expected manifest schema.

---

# 7. Initial Action Bank

Use a small, versioned bank such as:

| ID | Action | Positives | Negatives | Grouping | Calls |
|---|---|---:|---:|---|---:|
| A00 | safe anchor | 1 | 0 | one group | 1 |
| A01 | Top-3 | 3 | 0 | one group | 1 |
| A02 | Top-5 | 5 | 0 | one group | 1 |
| A03 | Top-3 + hard negatives | 3 | 2 | one group | 1 |
| A04 | local peaks | capped variable | 0 | one group | 1 |
| A05 | KMeans-3 | 3 centers | 0 | one point/group | 3 |
| A06 | KMeans-3 + negatives | 3 | up to 3 | group-specific | 3 |
| A07 | graph groups | capped variable | 0 | graph components | variable |
| A08 | graph groups + negatives | capped variable | capped variable | graph components | variable |
| A09 | high-coverage spread | capped | 0 | spatial groups | budget limit |

Do not enumerate continuous clustering parameters. Generate the bank from configuration and save its hash.

---

# 8. Metrics and Statistics

Select one primary metric per dataset before policy training.

Required metrics:

- Dice;
- IoU;
- AJI where valid;
- component/object recall;
- merge rate;
- area bias;
- point precision;
- component coverage;
- positive/negative prompt count;
- valid/requested prompt ratio;
- prompt-group count;
- SAM calls;
- runtime;
- peak VRAM.

Use query image or patient/case as the independent statistical unit. Multiple references, patches, and actions for the same query are not independent samples.

Use:

- paired bootstrap confidence intervals;
- per-query paired differences;
- win/tie/loss;
- oracle recovery;
- cost–accuracy Pareto curves;
- predeclared multiple-comparison correction where needed.

---

# 9. Goal-Mode Execution Order

## Phase 0 — Setup

1. Create `D:\SAM\adapromptset`.
2. Initialize package, configs, tests, reports, and paper skeleton.
3. Create protocol, decision, and exposure logs.
4. Inventory reusable FSPL modules.
5. Add config hashing and run metadata.

## Phase 1 — Manifests

1. MoNuSAC strict manifest.
2. MoNuSeg strict manifest.
3. GlaS strict manifest.
4. Pair, dimension, duplicate, overlap, and hash checks.
5. Dataset-role assignment.

**Gates:** A and B.

## Phase 2 — Matcher and candidates

1. Cache DINOv2-L features.
2. Validate mask-to-feature-grid projection.
3. Generate foreground/background evidence.
4. Generate positive and negative pools.
5. Render overlays.
6. Compute candidate diagnostics.

**Gate:** C.

## Phase 3 — Segmenter smoke test

1. GT-center and GT-box diagnostics.
2. Random/mismatched controls.
3. Prompt-group call logging.
4. Freeze multimask selection and fusion.
5. Verify metrics.

**Gate:** D.

## Phase 4 — Action utilities

1. Implement the initial action bank.
2. Freeze 10–20 pilot queries and references.
3. Run all eligible actions under matched budgets.
4. Save the query × action utility matrix.
5. Plot per-query oracle actions.
6. Measure oracle gap and costs.

**Gates:** E, F, and G.

## Phase 5 — Mandatory decision checkpoint

Emit exactly one status:

```text
CONTINUE_POLICY
PIVOT_DETERMINISTIC
PIVOT_ORACLE_ANALYSIS
REDESIGN_ACTION_BANK_ONCE
STOP_DATASET
```

Do not train the policy before this status is recorded.

## Phase 6 — Policy pilot

1. action prior;
2. classifier;
3. regression;
4. pairwise ranking;
5. one controlled rescue if required;
6. matched-cost comparison;
7. oracle recovery.

**Gate:** H.

## Phase 7 — Full validation and extension

Proceed only after a positive or pivot decision:

- full MoNuSAC validation;
- GlaS extension;
- ablations;
- reference sensitivity;
- cost analysis;
- qualitative successes and failures.

## Phase 8 — Frozen formal evaluation

Freeze all artifacts and run the untouched split.

**Gate:** I.

## Phase 9 — Paper

Compile the final two-column paper only after the formal outcome is known.

---

# 10. Gate Reports

For every gate, create:

```text
reports/gates/gate_<letter>_<name>.md
reports/gates/gate_<letter>_<name>.json
```

Each report must contain:

- input hashes;
- commands;
- diagnostics;
- plots;
- pass/fail;
- rule applied;
- allowed next actions;
- prohibited next actions;
- deviations.

No gate may pass solely through qualitative inspection.

---

# 11. Claim Matrix

| Evidence | Permitted claim |
|---|---|
| Candidate quality only | matching evidence generates useful candidates |
| Oracle gap only | one fixed prompt rule is query-suboptimal |
| Validation policy gain | adaptive selection is promising on validation |
| Frozen test gain | AdaPromptSet improves cost-matched selection on the tested protocol |
| Untouched GlaS gain | evidence extends across a morphology/dataset shift |
| Retraining on another model | interface is plug-compatible |
| Same frozen policy succeeds | limited cross-model transfer |
| Deterministic action near oracle | a fixed structured rule is sufficient in this setting |
| No oracle gap | adaptation is not justified for this bank and setting |

Do not use “universal,” “model-agnostic,” “plug-and-play,” or “generalizable” without corresponding frozen-transfer evidence.

---

# 12. Immediate Codex Instructions

Proceed without additional confirmation on these points:

1. Create the independent AdaPromptSet package.
2. Reuse stable FSPL modules with provenance.
3. Build strict manifests for MoNuSAC, MoNuSeg, and GlaS.
4. Create exposure and protocol logs before experiments.
5. Keep DINOv2-L + SAM-B as the MVP pair.
6. Use MoNuSAC for development and GlaS as the first clean extension.
7. Treat IDRID as non-tunable.
8. Defer formal DDR, CPM-17 repair, SAM-H, SAM2, and network access.
9. Run only 10–20 query oracle pilots after Gates A–E pass.
10. Stop at Gate F and emit a decision report before policy training.
11. Enforce the compute ledger and 120 GPU-hour ceiling.
12. Maintain a paper skeleton without premature performance claims.

---

# 13. Final Direction

AdaPromptSet should be implemented as a falsifiable sequence:

1. Is the candidate evidence adequate?
2. Does a meaningful query-specific action oracle exist?
3. Can a lightweight model recover the oracle advantage at matched cost?
4. Does the result survive one untouched formal evaluation?

If any answer is no, follow the registered pivot rather than expanding the search space.
