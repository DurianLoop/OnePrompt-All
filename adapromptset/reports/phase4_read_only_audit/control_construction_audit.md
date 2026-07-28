# Control Construction Audit

Status: `PASS_READ_ONLY`

- R03 box IoU definition: IoU between the automatic proposal bounding box and the query GT union bounding box, computed by _box_iou(proposal_box, target_box).
- Random box width/height source: The random box uses width and height copied from the corresponding automatic proposal bounding box.
- Random control inherits automatic scale: `_random_box_like(box, seed_text) copies x/y span from the automatic proposal box.`
- Random control uses query GT for generation: `False`
- Query GT usage: Query GT is used only after proposal generation for evaluation of IoU/Dice/recall, not for random-box placement.
- Random samples per query/family: `1`
- Random seed pattern: `phase3-r1:{family}:{query_case_id}`
- Full-image box recorded in original R1 artifact: `False`

Interpretation: R03's high absolute box IoU must be read against a size-matched random control that inherits automatic scale. Its small improvement and low win rate mean localization advantage is not established.
