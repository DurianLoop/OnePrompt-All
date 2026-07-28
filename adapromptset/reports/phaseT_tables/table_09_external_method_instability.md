# Table 9 — External method instability comparison

Status of the seven priority external comparators listed in Phase T §7.1, with their tier, code availability, reference-guided status, and reproduction disposition under our reference-resampling protocol.

| Method | Venue / year | Tier | Code available | Reference-guided | Reproduced under our protocol | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| GF-SAM | NeurIPS 2024 | 2 | YES | YES | NO (audit only) | Reproducible in principle (Option A); not run in Phase T — research/documentation only |
| SynPo | MICCAI 2025 | 3 | PARTIAL | YES | NO (audit only) | Tier 3 — concept comparison only (no code/weights) |
| ProSAM | arXiv (Bosch Research) 2025 | 3 | NO | YES | NO (audit only) | Tier 3 — concept comparison only (no code/weights) |
| MAUP | MICCAI 2025 | 2 | YES | YES | NO (audit only) | Reproducible in principle (Option A); not run in Phase T — research/documentation only |
| SAMIX | CVPR 2026 | 4 | YES | YES | NO (audit only) | Tier 4 — not applicable to reference-resampling protocol |
| AlignSAM | CVPR 2024 | 4 | YES | NO | NO (audit only) | Tier 4 — not applicable to reference-resampling protocol |
| Point Prompt Defender | CVPR 2026 | 4 | YES | NO | NO (audit only) | Tier 4 — not applicable to reference-resampling protocol |

**Notes:**
- Phase T §7.2 tiers: Tier 1 = code+weights reusable; Tier 2 = official code exists but interface incompatible; Tier 3 = no code/weights; Tier 4 = method not applicable.
- Reproduction status: NO external method was actually downloaded or run in Phase T. The audit is research/documentation only (per §7 and §23.1).
- GF-SAM and MAUP are the two Tier-2 methods with public code; they satisfy Option A of §7.3 (reproduce 2 external methods) pending user authorization to download external code.
- SynPo and ProSAM are Tier-3 (no code) — concept comparisons only, no fabricated numbers.
- SAMIX, AlignSAM, and Point Prompt Defender are Tier-4 (wrong paradigm or not reference-guided) — documented in the novelty audit only.
- reference_instability_results.csv and reproduced_results.csv in phaseT_external_baselines/ contain only headers (no data rows), confirming no external method was run.
