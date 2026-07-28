# GlaS Dataset Availability Audit

Date: 2026-07-23
Branch: `phaseT-publication-completion`

## Status: NOT AVAILABLE

The GlaS (Warwick QU) dataset directory exists at:
`D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)`

However, the directory contains only `.DS_Store` files — no image or mask
data is present. Prior phase split builders (`build_glas_phase2_splits.py`,
`build_glas_phase3_region_splits.py`) reference this path, indicating the
data was available during earlier phases but has since been removed or
the extraction is incomplete.

## Impact on WP2

GlaS cannot be used for cross-dataset replication. The minimum benchmark
submission requirement (directive §6.4) is "MoNuSAC holdout confirmed +
at least 2 additional datasets directional support." MoNuSeg and IDRID
will serve as the two additional datasets.

## Historical Exposure

Phase 2D historical Gate D exposure on GlaS strict validation (30 queries)
was run and FAILED. This exposure is acknowledged but cannot be replicated
in Phase T due to data unavailability.
