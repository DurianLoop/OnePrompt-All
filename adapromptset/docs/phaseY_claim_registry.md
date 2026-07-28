# Phase Y — Claim Registry

Date: 2026-07-21

Directive section: 16, 17.

## 1. Allowed Claim Tiers

| Tier | Source | Label | Allowed in Paper |
|---|---|---|---|
| A | Automatic deployable method on holdout | `DEPLOYABLE_RESULT` | main result table |
| B | Automatic deployable method on validation | `VALIDATION_RESULT` | supplementary |
| C | Finite-bank oracle | `FINITE_BANK_ORACLE` | upper bound, diagnostic |
| D | Cross-reference oracle | `CROSS_REFERENCE_ORACLE` | upper bound, diagnostic |
| E | GT-guided search oracle | `GT_GUIDED_SEARCH_ORACLE` | privileged upper bound, diagnostic |
| F | Phase X 10-episode evidence | `EXPLORATORY_PHASEX_EVIDENCE` | appendix only |

## 2. Strict Table Labels

Every paper table must label each row with one of:

```text
automatic deployable method
finite-bank oracle
cross-reference oracle
GT-guided search oracle
privileged upper bound
```

No mixing of deployable and oracle rows without explicit labels.

## 3. Pre-Registered Claims

| ID | Claim | Conditions | Status |
|---|---|---|---|
| C1 | Stable action heterogeneity exists on fresh queries | Y1 pass | PENDING |
| C2 | Search oracle labels are reference-stable | Y2 pass | PENDING |
| C3 | Bank-action policy beats strongest fixed | Y3 pass + holdout confirms | PENDING |
| C4 | Point-subset policy beats strongest fixed | Y4 pass + holdout confirms | PENDING |
| C5 | Oracle exists but is not predictable | Y3 fail AND Y4 fail (or Y2 blocks Y4) | PENDING |

Only ONE of C3, C4, C5 may appear in the main result. C1 and C2 are
diagnostic and appear in the analysis section.

## 4. Forbidden Claims

- "Search Oracle 0.7256 is a deployable result" — it is privileged.
- "Phase X 10-episode gap generalizes" — without Y1 confirmation.
- "Policy improves over strongest fixed" — without Y3/Y4 + holdout.
- Any claim using formal test before test-freeze protocol.
- Any claim that lowers a frozen gate threshold.
- Any claim that mixes GT-guided search outputs with automatic methods.

## 5. Negative Result Handling

If C5 triggers (oracle exists but not predictable), the paper reports an
honest negative empirical study. This is a valid submission outcome per the
research charter. No post-hoc claim expansion from negative evidence.

## 6. Holdout Claim Lock

The holdout claim is locked to the SINGLE pre-registered model (best on
validation). If the holdout run fails to confirm validation:

- Report the negative honestly.
- Do NOT re-tune on holdout.
- Do NOT access test.

## 7. Phase X Historical Anchors (Frozen, Appendix Only)

| Anchor | Value | Label |
|---|---:|---|
| Top20 + N1 fixed | 0.5411 | EXPLORATORY_PHASEX_EVIDENCE |
| A_large bank oracle | 0.6119 | EXPLORATORY_PHASEX_EVIDENCE |
| Search Oracle | 0.7256 | EXPLORATORY_PHASEX_EVIDENCE |
| Phase X episode count | 10 | EXPLORATORY_PHASEX_EVIDENCE |

These are preserved for historical continuity but MUST NOT be presented as
confirmatory policy success.
