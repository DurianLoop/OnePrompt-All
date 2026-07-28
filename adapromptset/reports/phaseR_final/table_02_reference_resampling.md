| Parameter | Value | Description |
|---|---|---|
| References per query | 6 | 3 from set A + 3 from set B |
| Reference sets | A / B | Patient-disjoint halves of the reference pool |
| Train resampling seeds | 5 | seeds 0-4; independent reference resamples per query |
| Validation reference assignment | Fixed (1 seed) | validation_AB_fixed.csv; prevents multiple observations |
| Reference pool size | 80 | Total available reference cases |
| Reference pool patients | 18 | Distinct patients in pool |
| Min available (train) | 70 | Worst-case eligible references for any train query |
| Min available (val) | 80 | Worst-case eligible references for any val query |
| Feasible for 5 seeds | True | Pool sufficiency confirmed |

Source: manifests/phaseR/train_resampling_seed_{0..4}.csv, validation_AB_fixed.csv
