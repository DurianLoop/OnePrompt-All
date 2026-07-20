# Stage 4 Plan: Multi-Reference Episodic Utility Policy

## Motivation

Stage 3 showed that a policy fitted from 30 queries against one reference does
not generalize reliably.  The next experiment changes the amount and structure
of policy supervision, not the held-out test set.

## Dataset and split

Use the local MoNuSAC processed dataset:

- 209 train cases and 101 official test cases (audited locally);
- deterministic train-only partition into a reference set, policy-training
  query set, and policy-validation query set;
- test cases are never used for model fitting, early stopping, architecture
  selection, or prompt-rule selection.

## Multi-reference episodes

For each training query, generate matching maps and action utilities against
multiple distinct references sampled from the training partition.  The policy
therefore observes variation in both the query and the reference-derived
matching evidence.  Validation uses distinct reference/query episodes.

At test time, report both every fixed rule and the policy under the same
prespecified reference protocol.  Reference IDs and query IDs are stored for
every row.

## Staged execution

1. Add a deterministic multi-reference episode builder with disjointness tests.
2. Run a bounded train-only feasibility pass (three references, 48 policy
   training queries, 12 validation queries) to verify label diversity and
   policy ranking stability.
3. If the feasibility pass gives a stable validation gain over `top5_n3`, scale
   to the full MoNuSAC train set and perform the 101-case held-out report.

The bounded feasibility pass prevents an hours-long CPU experiment from being
committed before the method has evidence of learnable, reference-dependent
action variation.
