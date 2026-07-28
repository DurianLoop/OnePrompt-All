# Route B DDR Development Protocol v1

Date: 2026-07-19

Route: `B -- Reference Promptability Benchmark`

Dataset: DDR lesion segmentation

Local root: `D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation`

## Purpose

This protocol creates the first fresh-data development entry for Route B. It is a benchmark infrastructure step only: it builds train/valid manifests for the DDR four-intent positive-query task and does not run matcher, segmenter, policy, router, or formal-test inference.

## Split Roles

| DDR split | Route B role | Allowed now | Prohibited now |
| --- | --- | --- | --- |
| `train` | Development/training pool | Manifest construction, hashing, mask profiling, future train-only labels | Formal-test claim, post-test tuning |
| `valid` | Development validation pool | Manifest construction, method selection after preregistered gates | Formal-test claim |
| `test` | Locked formal retinal-domain test | No file or label access before final freeze | Any threshold/action/model/claim selection |

## Four-Intent Task

The formal DDR task has four reference-defined lesion intents:

- `EX`
- `HE`
- `MA`
- `SE`

The current development manifest uses a positive-query protocol: a row is eligible for Route B development only when the image contains a non-empty mask for the selected intent. Empty intent masks are still recorded in all-intent manifests for prevalence accounting, but are excluded from positive-query manifests.

## Manifest Fields

Each row records:

- dataset and split;
- image id;
- lesion intent;
- absolute image and mask paths;
- SHA256 for image and mask;
- image and mask dimensions;
- foreground pixel count and foreground area ratio;
- whether the mask is non-empty;
- protocol role;
- formal-test lock flag.

## Exposure Rules

- DDR test remains `LOCKED_UNUSED`.
- The manifest builder must reject `test` if passed as a requested split.
- No formal test freeze file is created by this protocol; that belongs to a later final-method gate.
- IDRID observed behavior must not tune DDR thresholds, actions, prompt types, or claim wording.
- DDR train/valid may be used for Route B benchmark protocol development only after this manifest gate passes.

## Gate B-DDR-DEV-MANIFEST

Status is `PASS_DDR_ROUTE_B_DEV_MANIFEST` only if:

1. requested splits are exactly within `train` and `valid`;
2. every image has all four intent masks;
3. image/mask dimensions match;
4. every split has at least one positive query for each intent;
5. manifests and report hashes are written;
6. the report records `formal_test_access=false`.

If any item fails, Route B may not run DDR model inference until the manifest issue is fixed or the dataset is excluded from the current development stage.
