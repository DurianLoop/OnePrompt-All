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
