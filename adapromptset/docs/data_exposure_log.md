# Data Exposure Log

This log records whether each local dataset may influence AdaPromptSet MVP tuning.

| Dataset | Role | Root | Tuning | Notes |
|---|---|---|---|---|
| MoNuSAC | main development | `D:\SAM\code\datasets\MoNuSAC\processed` | tuning_allowed=True | Primary MVP development dataset; formal test remains locked by gate protocol. |
| MoNuSeg | sanity/historical | `D:\SAM\code\datasets\MonuSeg\official` | tuning_allowed=False | Used for sanity and historical alignment, not for MVP action-bank redesign. |
| GlaS | first morphology-shift extension | `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` | tuning_allowed=False | Extension/confirmation only until MoNuSAC gates pass. |
| GlaS Phase 3 region internal pool | region proposal calibration/selection | `manifests/phase3/glas_region_calibration_40.csv`; `manifests/phase3/glas_region_selection_15.csv` | tuning_allowed=True within official-train 55 pool only | Used only for region parameter/action-family selection under Phase 3; strict validation and testA/testB cannot tune parameters. |
| GlaS Phase 3 strict validation | region gate validation | `manifests/phase2/glas/episodes_val.csv` | tuning_allowed=False | Used for Gate R1/R2/R3 validation after preregistration; cannot choose thresholds or add actions. |
| GlaS testA/testB | locked formal test | `manifests/phase2/glas/locked_test_cases.csv` | tuning_allowed=False | Remains locked and unused until a final Phase 3 method is frozen. |
| Phase 4+ read-only audit | evidence diagnosis only | `reports/phase3_*`; `reports/phase2_glas/*`; historical gate reports | tuning_allowed=False | May diagnose failure and choose a research track; may not create new method outputs or alter frozen conclusions. |
| DDR train/valid | fresh candidate domain | `D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation` | tuning_allowed=True after protocol freeze | May support future Phase 4+ development under a registered track; DDR test remains locked. |
| DDR test | locked formal domain test | `D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation\test` | tuning_allowed=False | Must not be accessed until DDR final method/protocol freeze. |
| DDR Route B train/valid manifest | benchmark development manifest | `manifests/phase4plus/ddr_route_b_{train,valid}_*.csv` | tuning_allowed=True within Route B protocol only | File hashes, dimensions, and positive-query prevalence observed for train/valid; formal-test access remains false. |
| DDR Route B reference/query episodes | benchmark development episodes | `manifests/phase4plus/ddr_route_b_{train,valid}_reference_query_episodes.csv` | tuning_allowed=True within Route B protocol only | Same-intent references selected deterministically; valid references are train-only; reference rows are repeated support views, not independent samples. |
| DDR Route B matcher smoke | development-smoke evidence | `reports/route_b_ddr_development_smoke/` | tuning_allowed=True within Route B protocol only | One valid MA query was evaluated with train support. Query GT used for evaluation only; no SAM inference; no formal-test access. |
| DDR Route B stratified matcher smoke | stratified development evidence | `reports/route_b_ddr_stratified_matcher/` | tuning_allowed=True within Route B protocol only | Sixteen train/valid query-intent units evaluated with DINOv2-L only. Query GT used for evaluation only; no SAM inference; no formal-test access. |
| DDR Route B failure taxonomy | control-aware development taxonomy | `reports/route_b_ddr_failure_taxonomy/` | tuning_allowed=True within Route B protocol only | CPU-only control and morphology labels over the same 16 train/valid rows. Query GT used for evaluation labels only; no formal-test access. |


| MoNuSAC train/validation | Phase X action-space audit | `manifests/episodes/monusac/episodes_train.csv`; `manifests/episodes/monusac/episodes_val.csv` | tuning_allowed=True within Phase X protocol only | Used for candidate coverage scaling, dense prompt scaling, search oracle, and saturation audit. MoNuSAC test remains locked. |
| Phase X preregistration artifacts | CPU-only audit + docs | `docs/phaseX_*.md`; `reports/phaseX_*`; `reports/phaseX_historical_bank_audit/*` | tuning_allowed=False | Read-only summaries of historical artifacts; no new model inference; no formal-test access. |
