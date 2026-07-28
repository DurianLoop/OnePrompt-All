# Phase AB Blocker Registry

## Active Blockers

### B1: No Fresh Confirmatory Cohort
- Status: BLOCKED
- Impact: Cannot run WP2 (confirmatory experiment)
- Resolution: Paper is exploratory/benchmark only
- Requires: User-provided fresh data

### B2: SAM-Med2D Checkpoint Unverifiable
- Status: RESOLVED (REMOVED_FROM_MAIN_TABLE)
- Impact: Removed from external baseline comparison
- Resolution: Permanently removed unless full provenance provided

### B3: Pascal-5i/COCO-20i Only Fold-0
- Status: PENDING
- Impact: Cannot claim cross-domain generality to natural images
- Resolution: Run remaining folds or remove natural image claims

### B4: IDRID Segmentation Quality Very Low
- Status: KNOWN LIMITATION
- Impact: Risk ranking may transfer but segmentation not usable
- Resolution: Report as boundary case

## Resolved Blockers

1. Null model contradiction (Phase W bug) → RESOLVED (AA0)
2. SAM-L significance mislabeling → RESOLVED (AA0)
3. Anti-correlation structural claims → RESOLVED (AA7, removed)
4. Full RAMP over-claiming → RESOLVED (removed from main)