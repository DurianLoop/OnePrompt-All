# Table 17 — Runtime and compute accounting

Runtime, GPU hours, SAM-call count, episode count and split sizes for each Phase T run. All runs are CPU-only (GPU hours = 0).

| Run | Device | Runtime (s) | GPU hours | Total SAM calls | Total episodes | Queries | Seeds | Configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WP1 MoNuSAC holdout (benchmark) | cpu | 0.0000 | 0.0000 | 0 | 0 | 0 | 10 | 3 |
| WP2 MoNuSeg replication | cpu | 656.9990 | 0.0000 | 3440 | 450 | 30 | 5 | 3 |
| WP2 IDRID replication | cpu | 581.2440 | 0.0000 | 3504 | 450 | 30 | 5 | 3 |
| WP3 M1 dynamic budget (train+val) | — | 34.8130 | — | — | — | — | — | — |
| WP3 M2 selective prediction (train+val) | — | 1.3700 | — | — | — | — | — | — |
| WP3 M2 holdout confirmation | — | — | — | — | — | — | — | — |
| WP3 train/val benchmark (Phase 2 raw) | — | N/A (no run_metadata.json) | — | — | — | — | — | — |

**Notes:**
- All Phase T runs were executed on CPU (cuda_available = false); GPU hours = 0 for every run.
- WP1 runtime_seconds = 0.0 in metadata because the WP1 benchmark ingested pre-computed Phase 2 raw per-call predictions (the DINOv2-L + SAM-B inference was performed in Phase 2 and checkpointed to phase2_raw_per_call.csv).
- WP2 MoNuSeg and IDRID ran end-to-end (DINOv2-L + SAM-B inference) in 657 s and 581 s respectively on CPU.
- WP3 M1 (34.8 s) and M2 (1.37 s) run on top of cached episode features — they do not re-invoke SAM.
- WP3 M2 holdout confirmation summary JSON does not contain a runtime_seconds field; the holdout reuses cached predictions.
- phaseT_wp3_train/ and phaseT_validation/ do not contain a run_metadata.json file (only CSV outputs); their cost_scaling.csv files are reported in Table 5 / Table 14 instead.
- Total Phase T compute is dominated by WP2 cross-dataset inference (≈ 20 minutes of CPU).
