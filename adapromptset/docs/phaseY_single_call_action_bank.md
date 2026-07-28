# Phase Y — Single-Call One-Group Action Bank

Date: 2026-07-21

Directive section: 6, 7.

## 1. Main-Experiment Interface Lock

All Phase Y main-experiment actions are fixed to:

```text
segmenter    = SAM-B
SAM calls    = 1
prompt groups = 1 (G_ONE_GROUP)
fusion       = none
multimask rule = frozen
postprocess  = frozen
```

Rationale (Phase X findings):

- G_ONE_GROUP dominates at every K+ (Gate X3).
- Multi-group and multi-call strategies cause over-segmentation.
- Strongest fixed uses 1 call.
- Search Oracle's final deployed action also uses 1 call.

## 2. Prohibited in Main Experiment

- 2-100 SAM calls
- one-point-per-group
- mask union
- KMeans multi-group calls
- graph multi-group calls
- SAM-L / SAM2 / MedSAM
- GlaS / DDR
- formal test
- PPO / DQN
- continuous coordinate generation
- GT-guided search outputs as deployable actions

## 3. Full Audit Bank (Frozen)

Config: `configs/actions/phaseY_full_single_call_bank.yaml`

### Dimensions

| Dimension | Values |
|---|---|
| K+ | 5, 10, 15, 20, 25, 30, 40, 50, 75, 100 |
| K- | 0, 1, 3, 5 |
| Positive selector | P_TOPK, P_SPREAD, P_LOCAL_PEAKS |
| Negative selector (K-=0) | N_NONE |
| Negative selector (K->0) | N_BOTTOMK, N_HIGH_BACKGROUND |
| Grouping | G_ONE_GROUP |
| SAM calls | 1 |

### Action Definition

```text
A = (positive_selector, K+, negative_selector, K-, G_ONE_GROUP, 1_call)
```

No query GT is included in the action definition.

### Total Actions

- K-=0: 3 pos × 10 K+ × 1 neg = 30
- K->0 (1,3,5): 3 pos × 10 K+ × 2 neg × 3 K- = 180
- **Total: 210 actions**

### Action ID Scheme

```text
Y_<pos_selector>_K<k+>_N<k->_<neg_selector>_G1_C1
```

Example: `Y_P_TOPK_K20_N1_N_BOTTOMK_G1_C1`

## 4. Compact Policy Bank (Pending Pareto Pruning)

Config: `configs/actions/phaseY_compact_policy_bank.yaml` (placeholder)

The compact bank (12-32 actions) is derived from the full audit bank via
Pareto pruning on **Policy Train Queries only**, after the preregistration
commit. Pruning rules (section 7.2):

1. Drop actions dominated in Dice / calls / runtime by another action.
2. Drop actions clearly inferior across all train folds.
3. Within a family, if Dice diff < 0.003, keep the simpler action.
4. Validation and holdout do NOT participate in pruning.
5. Target compact bank size: 12-32 actions.

Pareto metrics:

- mean Dice
- component recall
- false-positive area
- catastrophic failure rate
- runtime
- point count
- reference stability

## 5. Bank Hash

The full audit bank YAML SHA-256 is recorded in
`reports/phaseY_input_hashes.json`. Any modification after preregistration
invalidates the protocol.

## 6. Output Files

```text
configs/actions/phaseY_full_single_call_bank.yaml     (210 actions, frozen)
configs/actions/phaseY_compact_policy_bank.yaml       (placeholder -> 12-32 actions)
reports/phaseY_action_bank/full_bank_metrics.csv      (pending Y1 utility matrix)
reports/phaseY_action_bank/pareto_pruning.md          (pending)
reports/phaseY_action_bank/compact_bank.json          (pending)
```
