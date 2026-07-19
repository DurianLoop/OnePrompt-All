# MoNuSAC Freeze Notice for AdaPromptSet Phase 2

Date: 2026-07-19
Applies to: all Phase 2 GlaS work under `D:\SAM\adapromptset`

MoNuSAC is frozen at `PIVOT_ORACLE_ANALYSIS`.

## Frozen Items

- `configs/actions/prompt_set_bank_v2.json` and `.yaml` must not be changed for MoNuSAC.
- Observed MoNuSAC train/validation/test episode manifests must not be resampled for Phase 2.
- The Gate F threshold remains Dice oracle gap `0.010`; it must not be lowered.
- The current candidate generator must not be changed because of MoNuSAC validation behavior.
- MoNuSAC adaptive policy training remains prohibited.
- The existing MoNuSAC oracle-analysis PDF must not be rewritten as a positive policy paper.

## Authoritative MoNuSAC Result

| Item | Value |
| --- | --- |
| Status | `PIVOT_ORACLE_ANALYSIS` |
| Gate F | `FAIL_ORACLE_GAP_SMALL` |
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | `0.37123847211635835` |
| Mean oracle Dice | `0.376249802278007` |
| Dice oracle gap | `0.0050113301616486194` |
| Mean best fixed AJI | `0.22834095079987268` |
| Mean oracle AJI | `0.23216115704701398` |
| AJI oracle gap | `0.0038202062471413` |

## Allowed Read-Only Work

- Fix report export bugs.
- Add diagnostic-only prompt statistics exporters.
- Add diagnostic-only reference sensitivity exporters.
- Fix clean LaTeX build behavior.
- Add provenance, hashes, and read-only analysis of existing MoNuSAC artifacts.

## Prohibited Feedback From GlaS to MoNuSAC

Phase 2 GlaS results must not be used to reinterpret, retune, or reopen MoNuSAC. GlaS may motivate a new domain-specific branch only under its own predeclared gates.
