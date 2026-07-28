# GlaS Region Prompt Claim Registry

Date: 2026-07-19

## Current Allowed Claims

After interpretation audit only:

- Phase 2D confirms failure of the P3 grouped sparse-point prompt path.
- C2 should be discussed as an alternative-reference control, not a true negative control.
- Privileged GT box prompting shows the SAM backbone has a region-prompt upper bound on GlaS.
- Phase 3 may study automatic reference-conditioned region proposals.

## Claims Not Allowed Yet

- automatic region prompting works;
- SAM refinement improves raw matching masks;
- adaptive prompt-type routing works;
- deterministic region prompting is sufficient;
- formal testA/testB performance;
- cross-segmenter generalization;
- universal, plug-and-play, or model-agnostic claims.

## Future Claim Boundaries

If only R1 passes:

`matching evidence can generate non-random region proposals on GlaS strict validation`

If R2 passes:

`automatic reference-conditioned region prompts provide a viable interface to frozen SAM-B under the tested validation protocol`

If deterministic region is near oracle:

`a deterministic matching-derived region prompt is sufficient under the tested setting`

If router and formal test pass:

`query-conditioned prompt-representation selection improves over the strongest fixed region prompt under matched inference cost`
