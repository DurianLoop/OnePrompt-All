# DDR Protocol Audit

Status: `PASS_PROTOCOL_DEFINED`

Stage 1 is infrastructure only; formal policy training remains prohibited before the predeclared gates allow DDR work.

## Stage 1

- Mask rule: `binary_union_of_EX_HE_MA_SE`
- Allowed use: `infrastructure_only`
- Prohibited use: `formal_scientific_claim`

## Stage 2

- Intents: `EX, HE, MA, SE`
- Reference rule: reference mask contains exactly one lesion intent
- Query rule: positive-query protocol first; query must contain the same target lesion for the selected intent

## Pairing Audit

| Split | Images | Complete cases | EX | HE | MA | SE | Missing labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 383 | 383 | 383 | 383 | 383 | 383 | 0 |
| valid | 149 | 149 | 149 | 149 | 149 | 149 | 0 |
| test | 225 | 225 | 225 | 225 | 225 | 225 | 0 |

## Boundaries

- DDR union masks may be used for loader, merge, coordinate, candidate, segmenter-smoke, and runtime checks.
- DDR four-intent evaluation must be predeclared before any formal run.
- IDRID observed test behavior must not tune DDR candidates, thresholds, actions, or claims.
