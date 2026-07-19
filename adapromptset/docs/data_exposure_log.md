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
