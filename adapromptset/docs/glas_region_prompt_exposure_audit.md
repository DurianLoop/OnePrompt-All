# GlaS Region Prompt Exposure Audit

Date: 2026-07-19

## Status

`PASS_PRETEST_BOUNDARY`

## Data Exposure

Phase 3 uses:

- GlaS official train only for 55/30 internal and strict-validation roles;
- existing frozen 30 strict-validation query pool;
- existing frozen 55 utility/reference pool;
- new 40/15 internal split within the 55 utility pool.

GlaS testA/testB are listed only in locked manifests and are not read for Phase 3 method selection.

## Prior Exposure

GlaS testA/testB are not treated as fully blind because prior manifest/profile exposure existed, but there is still no approved use of testA/testB for method selection, threshold selection, region action selection, training, or qualitative sampling.

## Dense Map Availability

The existing committed cache contains candidate-cache manifests, not complete dense matching maps. If dense maps are needed later, they must be regenerated from the frozen DINOv2-L matcher and Gate C configuration, with hashes recorded before use.

## Boundary

No Phase 3 automatic region claim may use:

- query GT for automatic proposal generation;
- testA/testB;
- new matcher;
- new candidate generator;
- strict-validation threshold selection;
- post-hoc action additions.
