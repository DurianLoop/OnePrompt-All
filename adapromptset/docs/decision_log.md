# Decision Log

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-07-17 | Use `AdaPromptSet_Codex_Response.md` as highest-priority directive | It resolves the open questions needed for goal-mode execution | locked |
| 2026-07-17 | Build an independent `src/adapromptset` package while reusing FSPL provenance | Existing FSPL modules provide the nearest tested patterns, but AdaPromptSet needs isolated artifacts and gates | active |
| 2026-07-17 | Treat patient-overlapping source-train cases as `excluded_overlap` instead of train | Preserves official test while avoiding patient leakage | active |
| 2026-07-17 | Interpret 0.5 Dice points as absolute Dice 0.005 | Removes ambiguity between absolute Dice and percentage-point language | locked |
| 2026-07-17 | Use feature-grid random baseline for Gate C candidate precision | AdaPromptSet candidates are DINOv2 patch-grid centers, so pixel-random baseline is not support-matched | locked |
| 2026-07-17 | Export only oracle-analysis paper assets after the MoNuSAC Gate F pivot | Current evidence prohibits MoNuSAC policy training and supports `oracle_analysis_only`, so generated tables/macros must not imply adaptive-policy success | active |
| 2026-07-17 | Keep the current paper build as an oracle-analysis-only skeleton | The submission guard passes for placeholder removal, but PDF compilation and full Phase 15/16 paper acceptance remain open | active |
| 2026-07-17 | Record run metadata as compact hashes plus environment/git summaries | Full git porcelain is too noisy for artifacts; reproducibility requires stable config/input hashes, environment metadata, and a dirty-count/sample instead | active |
| 2026-07-17 | Treat multi-reference episodes as repeated measurements, not independent samples | MoNuSAC uses up to 5 references per query and repeated reference rows would inflate sample size; reports must aggregate at query level and use patient clusters where patients have multiple queries | locked |
| 2026-07-17 | Mark pilot bootstrap evidence pending when patient clusters are insufficient | The current v2 pilot has 2 query-level rows but only 1 patient cluster, so reporting a confidence interval would overstate the evidence | locked |
| 2026-07-17 | Treat pilot paired significance as diagnostic only when statistical units are insufficient | The exact sign-flip p-value can be computed, but the current pilot has too few independent units for a formal significance claim | locked |
| 2026-07-19 | Preserve the original GlaS 6-case Gate D failure as exploratory and run one preregistered 30-query viability confirmation | The all-case rule was too strong for a 6-case execution gate; the confirmation uses all 30 strict-validation queries, keeps testA/testB locked, does not change Gate C/action bank/policy, and freezes thresholds before SAM inference | active-preregistered-not-run |
| 2026-07-19 | Close GlaS Gate D as `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED` after Phase 2D confirmation | SAM-B and SAM-L both failed the preregistered P3 query-level criterion on all 30 strict-validation queries; Gate E/F/G/H/I remain closed and testA/testB remain unused | locked |
| 2026-07-19 | Add Phase 3 interpretation state `PIVOT_GLAS_REGION_PROMPT_REQUIRED` without changing the Phase 2D operational state | Phase 2D supports grouped sparse point prompting failure but not complete SAM-backbone failure, because P4 GT box upper bounds remain strong and C2 is an alternative-reference control | preregistered |
| 2026-07-19 | Freeze the Phase 3 automatic region bank and 40/15 internal split before new SAM inference | Region parameters/actions must be selected only on the 40/15 internal protocol; strict validation cannot add actions or choose thresholds | preregistered |
