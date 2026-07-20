# Stage 5 Spatial-Layout Feasibility Record

## Small feasibility result

The initial MoNuSAC training/validation-only run used three references, 12
train queries, six validation queries, and two actions (`top5_n3` and
`spread5_n3`).  It generated 108 finite action-utility rows.

| Validation method | Dice |
|---|---:|
| Fixed `top5_n3` | 0.2323 |
| Fixed `spread5_n3` | 0.2733 |
| Layout policy | 0.2761 |

`spread5_n3` won 7/18 episodes, lost 5/18, and tied 6/18.  The mean paired
gain over `top5_n3` was 0.0410, but the bootstrap 95% interval crossed zero
because the validation sample is small.

## Next check

Run the same pre-specified two-action protocol with three references, 48
train queries and 12 validation queries.  This adds training evidence while
leaving all 101 MoNuSAC test cases unread.  Promote to held-out testing only
when the larger validation set shows a paired mean policy gain exceeding its
paired standard error and nontrivial selection of both actions.

## Medium replication result

The replication used three references, 48 train queries and 12 validation
queries, yielding 360 finite action-utility rows and 36 validation episodes.

| Validation method | Dice |
|---|---:|
| Fixed `top5_n3` | 0.2812 |
| Fixed `spread5_n3` | 0.3059 |
| Layout policy | 0.3035 |

The policy selected `spread5_n3` for 30/36 validation episodes and `top5_n3`
for 6/36. Its paired gain over `top5_n3` was `+0.02233` with paired standard
error `0.01622`; the spread action itself had paired gain `+0.02468` with
standard error `0.01636`. This satisfies the stated promotion rule, while the
bootstrap interval remains close to zero and must be reported honestly.

## Promotion

Proceed to the held-out MoNuSAC test only under the same three references and
two prespecified actions. Refit the chosen ExtraTrees configuration on the
combined train and validation utility rows, then emit one action prediction
per reference/test-query episode. Do not alter candidate spacing, action
definitions, leaf size, or policy feature set after seeing test results.
