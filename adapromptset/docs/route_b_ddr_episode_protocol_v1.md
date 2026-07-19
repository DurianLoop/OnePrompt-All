# Route B DDR Reference/Query Episode Protocol v1

Date: 2026-07-19

Route: `B -- Reference Promptability Benchmark`

Depends on:

- `docs/route_b_ddr_dev_protocol_v1.md`
- `docs/route_b_benchmark_metric_control_protocol_v1.md`
- `manifests/phase4plus/ddr_route_b_train_positive_queries.csv`
- `manifests/phase4plus/ddr_route_b_valid_positive_queries.csv`

## Purpose

This protocol freezes deterministic DDR reference/query episode construction for Route B development evidence. It does not run matcher or segmenter inference.

## Episode Rule

For each positive query row:

- match only references with the same lesion intent;
- select `references_per_query=3`;
- use deterministic hash ranking with seed `route-b-ddr-episode-v1`;
- for `train` queries, select references from DDR train and exclude the same image id;
- for `valid` queries, select references from DDR train only;
- never use DDR test as query or reference source.

## Statistical Boundary

The independent unit remains the query image/case. Reference rows are repeated support views and must not be treated as independent samples. Image-intent rows are repeated task views of the same image.

## Gate B-DDR-EPISODE

Status is `PASS_DDR_ROUTE_B_EPISODES` only if:

1. train and valid positive-query manifests exist and passed the manifest gate;
2. every positive query receives the requested number of same-intent references;
3. valid references come only from train;
4. train references exclude the query image;
5. no test manifest or DDR test file is read;
6. output episode CSVs and report hashes are written.

Passing this gate only creates development input manifests. It still does not authorize formal test access, submission-ready claims, or method-positive claims.
