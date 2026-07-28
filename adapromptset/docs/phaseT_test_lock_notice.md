# Phase T Formal Test Lock Notice

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §21 (Formal Test Gate)

## Lock Declaration

The formal test remains **LOCKED** for the entire Phase T program. The
following datasets / splits are LOCKED_UNUSED:

| Dataset | Split | Cases | Status |
|---|---|---:|---|
| MoNuSAC | test | 101 | LOCKED_UNUSED |
| GlaS | testA | 60 | LOCKED_UNUSED |
| GlaS | testB | 20 | LOCKED_UNUSED |
| MoNuSeg | test | 14 | LOCKED_UNUSED |
| DDR | test | 225 | LOCKED_UNUSED |

## Lock Enforcement

1. No Phase T script may read, list, or open any file under the test
   directories of these datasets.
2. No Phase T report, table, or figure may contain test-set metrics.
3. No Phase T claim may reference test-set performance.
4. The lock applies regardless of WP1 / WP2 / WP3 / WP4 Gate outcomes
   (pass or fail).
5. The lock survives the entire Phase T program; it is NOT lifted by any
   automated state transition, including `PHASET_PAPER_CANDIDATE_READY`.
6. Even if WP1 + WP2 + WP4 all pass and `PHASET_PAPER_CANDIDATE_READY` is
   written, the formal test remains locked.

## Unlock Conditions (Future, Beyond Phase T)

The formal test may ONLY be unlocked by a SEPARATE, future Phase U protocol
that satisfies ALL of the following:

1. **WP1 + WP2 + WP4 all pass** — MoNuSAC holdout confirmed, ≥2 extra
   datasets directional support, full paper draft complete, mock review
   with no fatal unresolved issues.
2. **User approval** — explicit user request to proceed to formal test.
   Phase T does NOT self-approve formal test access.
3. **Phase U freeze protocol** — a separately preregistered Phase U
   protocol that:
   - Is preregistered in its own commit.
   - Specifies the exact frozen method checkpoint, config, and thresholds.
   - Produces `docs/phaseU_formal_test_freeze.md`,
     `configs/frozen/phaseU_final_benchmark.yaml`,
     `reports/phaseU_test_hashes.json`,
     `reports/phaseU_statistical_plan.md`,
     `reports/phaseU_qualitative_plan.md`.
   - Runs each test set exactly once.
   - Is hash-verified before the test run begins.

Phase T does NOT unlock the formal test. Phase T does NOT prepare the
Phase U protocol. Phase T only records whether a frozen method reached a
state where a future Phase U formal-test protocol would be warranted.

## No Exceptions

There are NO exceptions to this lock. In particular:

- A WP1 holdout pass does NOT unlock the formal test.
- A WP3 M1 / M2 pass does NOT unlock the formal test.
- A `PHASET_PAPER_CANDIDATE_READY` state does NOT unlock the formal test.
- A user request to "just check" the formal test does NOT constitute the
  Phase U freeze protocol and is NOT honored.
- A negative holdout result does NOT unlock the formal test for "rescue".
- Exceeding the 195 GPU-hour cap does NOT unlock the formal test.

## Audit

Any Phase T artifact that mentions "test" or "testA" or "testB" in a
non-Lock-Notice context will be flagged by the claim-boundary audit before
final commit. The audit covers:

- `docs/phaseT_*.md`
- `reports/phaseT_*.{md,json,csv}`
- `paper/phaseT/*.md`
- Terminal replies

Exceptions are documented in `docs/phaseT_claim_registry.md` and
`docs/phaseT_dataset_registry.md` (where "test" appears as a role label
with LOCKED status).
