# GlaS New-Domain Pilot Summary

Date: 2026-07-17

## Status

This is a predeclared new-domain diagnostic after MoNuSAC Gate F failed. It does not reopen MoNuSAC policy training and does not use GlaS testA/testB.

## Artifacts

- Protocol: `reports/new_domain_pilot_glas_protocol.md`
- Candidate pilot: `reports/candidate_pilot_glas/candidate_pilot_summary.json`
- Oracle pilot: `reports/oracle_gap_pilot_glas/oracle_gap_pilot_summary.json`
- Gate status: `reports/gate_f_status_glas.md`

## Result

- Candidate pilot status: `PASS`
- Candidate pilot episodes: 1 strict GlaS val episode
- Mean point precision: 1.0
- Mean random precision: 0.4
- Mean component coverage: 0.375
- Oracle pilot status: `PENDING_INSUFFICIENT_EPISODES`
- Oracle pilot episodes: 1 strict GlaS val episode
- Gate minimum episodes: 10
- Best fixed action: `R_top5_independent_localneg`
- Mean oracle Dice: 0.7180205482567982
- Mean best fixed Dice: 0.7180205482567982
- Dice oracle gap: 0.0

## Interpretation

The strict GlaS val smoke shows that the current candidate and action execution path can run on the first extension dataset. It cannot decide Gate F because the strict role-level GlaS validation split currently has only one episode. The correct next step, if more GlaS evidence is desired, is a separately labeled train/val diagnostic or a revised predeclared GlaS protocol that preserves testA/testB as untouched.

