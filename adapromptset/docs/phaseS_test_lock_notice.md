# Phase S Formal Test Lock Notice

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Lock Declaration

The following datasets are LOCKED for the entire Phase S program:

| Dataset | Split | Cases | Status |
|---|---|---:|---|
| MoNuSAC | test | 101 | LOCKED_UNUSED |
| GlaS | testA | 80 | LOCKED_UNUSED |
| GlaS | testB | 80 | LOCKED_UNUSED |
| DDR | test | 225 | LOCKED_UNUSED |

## Lock Enforcement

1. No Phase S script may read, list, or open any file under the test
   directories of these datasets.
2. No Phase S report, table, or figure may contain test-set metrics.
3. No Phase S claim may reference test-set performance.
4. The lock applies regardless of Gate outcomes (pass or fail).
5. The lock survives the entire Phase S program; it is not lifted by any
   automated state transition.

## Unlock Conditions (Future, Beyond Phase S)

The formal test may only be unlocked by a SEPARATE, future protocol that:
1. Is preregistered in its own commit.
2. Specifies the exact frozen method checkpoint, config, and thresholds.
3. Is approved by the user.
4. Runs each test set exactly once.

Phase S does NOT unlock the formal test. Phase S does NOT prepare the formal
test protocol. Phase S only records whether a frozen method reached a state
where a future formal-test protocol would be warranted.

## Audit

Any Phase S artifact that mentions "test" or "testA" or "testB" in a
non-Locked-Notice context will be flagged by the claim-boundary audit before
final commit.
