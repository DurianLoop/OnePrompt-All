# Research Program Decision Tree

Date: 2026-07-19

```text
Phase 4 read-only audit
  |
  +-- PHASE4_AUDIT_SUPPORTS_METHOD_RESEARCH
  |     -> Score A/B/C
  |     -> likely Route A or C primary
  |
  +-- PHASE4_AUDIT_SUPPORTS_BENCHMARK_TRACK
  |     -> Route B primary
  |
  +-- PHASE4_AUDIT_SUPPORTS_DIRECT_SEGMENTATION_TRACK
  |     -> Route C primary
  |
  +-- PHASE4_AUDIT_SUPPORTS_PROMPTABILITY_TRACK
  |     -> Route A primary
  |
  +-- PHASE4_AUDIT_INSUFFICIENT_OR_INCONSISTENT
        -> correction addendum or research-program exhaustion review
```

After track selection:

```text
Primary track gate pass
  -> validation expansion
  -> model/dataset extension
  -> formal test freeze
  -> formal test
  -> ablations/statistics
  -> submission paper and mock review

Primary track gate fail
  -> pivot to registered fallback

Fallback fail
  -> evaluate deferred third track from existing evidence
  -> otherwise RESEARCH_PROGRAM_EXHAUSTED_NOT_SUBMISSION_READY
```

No branch may skip from validation evidence directly to `SUBMISSION_READY`.
