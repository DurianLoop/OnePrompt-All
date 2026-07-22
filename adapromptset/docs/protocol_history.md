# Protocol History

## 2026-07-17 Phase 0/1 Initialization

- `AdaPromptSet_Codex_Response.md` is the highest-priority execution decision file.
- MVP scope is locked to MoNuSAC main development, MoNuSeg sanity/historical, and GlaS first extension.
- MVP matcher/segmenter scope is DINOv2-L + SAM-B.
- IDRID is read-only/non-tunable; DDR, CPM-17, SAM-L/H, SAM2, SAM3, and network downloads are deferred.
- Phase 1 audit uses source manifests plus role-level leakage control. Source-train cases with patients appearing in official test are marked `excluded_overlap`.
- Gate A role-level audit passed for MoNuSAC, MoNuSeg, and GlaS using `scripts/audit_data.py`.
- Role counts: MoNuSAC train=200, excluded_overlap=9, test=101; MoNuSeg train=37, test=14; GlaS train=8, excluded_overlap=77, test=80.
- Gate B episode audit passed for MoNuSAC, MoNuSeg, and GlaS using `scripts/build_episodes.py`.
- Episode counts: MoNuSAC train=810, val=190, test=505; MoNuSeg train=29, val=8, test=14; GlaS train=7, val=1, test=80.
- Phase 3 candidate score-map kernel was added with synthetic smoke only. `reports/gate_c_status.md` marks the kernel smoke as PASS and formal matcher Gate C as `PENDING_REAL_MATCHER`.
- Matcher runtime audit records DINOv2-L and SAM-B checkpoint hashes plus explicit external import paths `D:\SAM\code` and `D:\SAM\Matcher`; `dinov2` and `segment_anything` are importable after path configuration.
- Single MoNuSAC validation episode DINOv2-L forward smoke passed on GPU. It wrote `reports/real_matcher_smoke/score_maps.pt` and `candidates.csv`; diagnostic point precision was 0.2 and component coverage was 0.0043, so formal Gate C remains pending a multi-episode candidate quality pilot.
- Gate C candidate pilot was run on 5 MoNuSAC validation episodes. An initial pixel-random baseline was rejected as unfair because method candidates live on DINOv2 feature-grid centers; after switching the predeclared random baseline to the same grid-center support, mean precision was 0.32 vs random 0.20, delta 0.12. Negative blank fraction was 0.0 and coordinate issues were 0, so `reports/gate_c_status.md` marks formal Gate C PASS for the current MVP candidate kernel. Coverage remains low in absolute terms and must be interpreted cautiously in oracle/action experiments.
- SAM-B segmenter smoke ran on one MoNuSAC validation episode. The API path, positive/negative labels, calls-vs-groups accounting, and repeated-prompt determinism passed. A root-cause check showed that the query semantic mask contains 233 connected components; the privileged point's component has 243 pixels, while the full semantic union has 57502 pixels. SAM-B's one-point mask scored Dice=0.9382 and IoU=0.8836 against the point component, but only Dice=0.0084 against the full semantic union. `reports/gate_d_status.md` therefore marks formal Gate D PASS for wrapper correctness, with semantic-union Dice retained as a fusion/utility limitation rather than a segmenter API failure.
- Shared MVP fusion was added in `src/adapromptset/fusion/masks.py`: area filtering, score-ordered mask IoU NMS, semantic union, and a locked `configs/segmenter/fusion_mvp.json` config with hash `5F15975FBE2BBF0E16B2A06DAF0A6F67F0BF7E50824F97CE6B3B66C0A743A8F4`. Dice, IoU, and AJI now have known-case unit tests. Instance overlap arbitration and runtime timing remain open Phase 4 items.
- MVP action bank generation was added through `scripts/build_action_bank.py`. It builds `prompt_set_bank_v1` from the unified candidate pilot CSV using positives-then-negatives candidate indexing. The current manifest has 10 actions, 9 valid actions, and bank hash `586342F7077C4CAD769436F2BD191762C8ABCD3805F97D6ED371BFB21010DDF7`; `A_top10_one_group` is intentionally invalid under the current 5-positive candidate pilot. Gate E is therefore `PASS_MVP_SUBSET` rather than full PASS, because KMeans, local-peak, SynPo-style, GF-SAM-style, and safe-anchor families remain unimplemented.
- Action utility execution was added in `src/adapromptset/utility/action_execution.py` and `scripts/run_action_utility_smoke.py`. A single MoNuSAC validation episode smoke ran all 9 valid MVP actions through SAM-B and shared fusion. It wrote `reports/action_utility_smoke/action_utility_smoke.csv`; best smoke action was `B_top5_pos1neg` with Dice=0.4027, IoU=0.2521, max budget calls=5, decoder_calls=11, cache_hits=4, and GPU-hours estimate=0.000561. This only proves the action-utility path works; Gate F remains pending the required 10-20 query oracle-gap pilot.
- The candidate pilot was expanded to 10 validation episodes and remained Gate C PASS: mean point precision=0.52 vs random=0.16, precision delta=0.36, mean component coverage=0.0520, and mean negative blank fraction=0.08. The 10-query MVP subset oracle-gap pilot then ran 90 utility records over 9 valid actions. Gate F failed for small oracle gap: mean oracle Dice=0.3759, best fixed `B_top5_pos3neg` mean Dice=0.3712, gap=0.0047. All oracle actions were 1-call actions, so budget 1/3/5 matched-budget gaps were also 0.0047. Policy training is paused; the next allowed step is the predeclared Rescue-F1 action-bank expansion on train/validation.
- Rescue-F1 v2 action-bank generation was added through `scripts/build_action_bank.py --bank-version v2`. It preserves v1 actions and adds `R_split5_2groups`, `R_split5_2groups_localneg`, and `R_top5_independent_localneg`; the v2 manifest has 13 actions, 12 valid actions, and bank hash `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`. Gate E is now `PASS_RESCUE_SUBSET`, still not full action-bank PASS.
- The 10-query Rescue-F1 v2 oracle-gap pilot ran 120 utility records over 12 valid actions and still failed Gate F under the authoritative 0.010 Dice threshold: mean oracle Dice=0.3762, best fixed `B_top5_pos3neg` mean Dice=0.3712, gap=0.0050. The refreshed action utility CSV now also records AJI; mean oracle AJI=0.232161, mean best fixed AJI=0.228341, and AJI oracle gap=0.003820, also below the checklist's 0.015 auxiliary threshold. Only one query chose a rescue action (`R_split5_2groups_localneg`), top oracle action fraction stayed 0.5, and matched-budget gaps stayed below threshold. `reports/gate_f_status.md` now records `PIVOT_F_REQUIRED`; MoNuSAC policy training must not start unless a separately predeclared new-domain pilot, such as GlaS/DDR, demonstrates real action heterogeneity.
- Phase 5 decision reporting was added through `scripts/write_decision_checkpoint.py`. It reads the v1/v2 Gate F artifacts, writes `reports/decision_checkpoint/phase5_decision.{json,md}`, and exports `reports/decision_checkpoint/oracle_action_heatmap.csv`. The current unique Phase 5 status is `PIVOT_ORACLE_ANALYSIS`, with allowed next actions limited to fixed-rule/action-oracle analysis or a predeclared new-domain pilot.
- Frozen-module optimizer protection was added through `src/adapromptset/training/optimizer_guard.py` and `scripts/audit_optimizer_guard.py`. The guard fails if any optimizer param group contains matcher or segmenter parameters, independent of `requires_grad`; the current smoke report is `reports/optimizer_guard/optimizer_guard.json` with status `PASS`, `policy_training=not_started`, and file SHA256 `409D0A9BB08C54D0A3952646B05C10EEA1C540C7E3D624081775BDB2BB652D57`.
- Run metadata and config hashing were added through `src/adapromptset/reproducibility/metadata.py` and `scripts/write_run_metadata.py`. The current Phase 0 reproducibility record is `reports/run_metadata/phase0_reproducibility/run_metadata.json`, with config hash `1DD5A11E2C6A897F0EDA671F0E10A77B0F5C3FB9F706341F0EFDB98383954F60` and file SHA256 `1BAEB43D97C91E697B2E7F132DB777AED6EB4B4D79A167BF0CDAF75A302D0A27`. It records the MoNuSAC manifest and Phase 5 decision artifact hashes, compact git status, Python/Torch/CUDA/GPU metadata, and the `PIVOT_ORACLE_ANALYSIS` / policy-prohibited status.
- Fixed-rule/action-oracle pivot analysis was added through `scripts/analyze_fixed_oracle_pivot.py`. It writes `reports/fixed_oracle_analysis/fixed_oracle_summary.json`, `fixed_oracle_report.md`, `episode_regret.csv`, `budget_pareto.csv`, and `action_summary.csv`. The current result does not promote a deterministic method: best fixed remains `B_top5_pos3neg`, mean Dice oracle gap is 0.005011, mean AJI oracle gap is 0.003820, near-oracle episode fraction is 0.6, top oracle action fraction is 0.5, and the recommended claim is `oracle_analysis_only`.
- Claim-safe oracle-analysis paper assets were added through `scripts/export_oracle_analysis_tables.py`. The exporter reads the real fixed-oracle and GlaS smoke artifacts, writes `paper/generated/oracle_analysis_macros.tex`, `paper/tables/table_oracle_pivot.tex`, `paper/tables/table_action_summary.tex`, and `reports/oracle_analysis_paper_assets.md`, and intentionally exports `oracle_analysis_only` / policy-training-prohibited status rather than any adaptive-policy success claim.
- Claim-free paper skeleton generation was added through `scripts/build_paper.py`. Draft mode refreshes the oracle-analysis assets and writes `paper/main.tex`, `paper/sections/*.tex`, and `paper/references.bib`; submission mode runs a guard that rejects simulated placeholders before allowing a submission build. The skeleton deliberately states the `PIVOT_ORACLE_ANALYSIS` status and does not claim policy improvement, transfer, efficiency, or leaderboard-level results. A first PDF compile attempt stalled while loading the nonessential `hyperref` package, so the skeleton was simplified to a smaller LaTeX dependency set and the generated oracle-analysis tables were changed to two-column `table*` floats with compact display labels. The initial oracle-only skeleton wrote `paper/main.pdf` with SHA256 `D422B8FC545EE85E18D5BBB07AAF1CD7DF6ACAEB5BD7C03810DE915297FE8378`; final 8-page paper acceptance remains open.
- A predeclared GlaS new-domain smoke was added after the MoNuSAC Gate F pivot. Dataset-aware path routing and independent GlaS pilot report paths were added so extension evidence does not overwrite MoNuSAC Gate F artifacts. `reports/candidate_pilot_glas/candidate_pilot_summary.json` passed on the single strict GlaS validation episode (mean point precision 1.0 vs random 0.4, component coverage 0.375). `reports/oracle_gap_pilot_glas/oracle_gap_pilot_summary.json` is `PENDING_INSUFFICIENT_EPISODES`: episode_count=1, gate_min_episodes=10, best fixed `R_top5_independent_localneg`, mean oracle Dice=0.7180, mean best fixed Dice=0.7180, and oracle gap=0.0. This is only a new-domain smoke; it does not justify policy training or a GlaS Gate F claim.
- Statistical-unit auditing was added through `src/adapromptset/statistics/units.py` and `scripts/audit_statistical_units.py`. The report writes `reports/statistical_units/statistical_units.{json,md}` with status `PASS`. MoNuSAC train/val/test use 5 reference episodes per query and therefore must not treat reference rows as independent samples; recommended reporting units are patient clusters for MoNuSAC (train 810 episodes / 162 queries / 35 patients; val 190 / 38 / 9; test 505 / 101 / 25) and query-level for the current MoNuSeg/GlaS smoke manifests.
- Reference exposure auditing was added through `src/adapromptset/scripts/audit_reference_exposure.py` and `scripts/audit_reference_exposure.py`. The report writes `reports/reference_exposure_audit.{json,md}` by checking each current episode manifest against its `case_assignments.csv`. Current status is `PASS`: MoNuSAC 1505 episodes, MoNuSeg 51 episodes, and GlaS 88 episodes all use train-role support/reference rows, with no support case, support patient, support image path, or support mask path pointing to the test role. This closes the current `reference pool train-only` and `reference-test leakage = 0` checklist items for the existing manifests only; future regenerated manifests must rerun the audit.
- Data profile auditing was added through `src/adapromptset/scripts/audit_data_profile.py` and `scripts/audit_data_profile.py`. The report writes `reports/data_profile_audit.{json,md}` by opening each manifest image/mask pair and summarizing image dimensions, foreground mask area, mask area fraction, and instance/label count. The first real run exposed that manifest paths are relative to each dataset root's parent directory, matching `audit_data.py`; the resolver now supports both root-relative and parent-relative paths. Current status is `PASS` over 526 rows: GlaS 165 cases, MoNuSAC 310 cases, and MoNuSeg 51 cases, with no profiling issues. This closes the current image-size/mask-area/instance-distribution checklist item for the existing manifests.
- DDR protocol auditing was added through `src/adapromptset/scripts/audit_ddr_protocol.py` and `scripts/audit_ddr_protocol.py`. The report writes `reports/ddr_protocol_audit/ddr_protocol_audit.{json,md}` from the local DDR lesion segmentation tree. Current status is `PASS_PROTOCOL_DEFINED`: train 383, valid 149, and test 225 images each have EX/HE/MA/SE labels, with total_missing_label_count=0. Stage 1 is `binary_union_of_EX_HE_MA_SE` for infrastructure only; Stage 2 is a four-intent EX/HE/MA/SE positive-query protocol with per-intent and macro metrics. Formal DDR policy training remains prohibited until a separately predeclared gate allows it.
- Coordinate resize/pad round-trip utilities were added through `src/adapromptset/geometry/resize_pad.py`. The utility records original size, target size, uniform scale, resized size, and letterbox padding, and maps `(x,y)` points between original and padded coordinates with inverse pixel clamping. `tests/test_resize_pad_transform.py` covers wide/tall images, padding offsets, inverse round-trip, batch mapping, and invalid dimensions. This closes the current coordinate resize/pad inverse-mapping test item only; feature caching remains a separate open item.
- Matching score-map global features were added through `score_map_global_features` in `src/adapromptset/matching/evidence.py`. The function computes valid-cell count, score mean/std, top score and grid index, second score, top margin, normalized softmax entropy, high-score threshold, and high-score area fraction, with optional valid-mask exclusion. `tests/test_matching_evidence.py` covers entropy/margin/high-score-area behavior and valid-mask handling. The real matcher smoke export now writes these diagnostics for `s_fg`, `s_bg`, and `s_ctr` under `score_map_features` in `reports/real_matcher_smoke/real_matcher_smoke.json`; the current artifact status is `PASS` with formal Gate C still `PENDING_FULL_CANDIDATE_PILOT`. This supports the oracle/representation-analysis route without reopening MoNuSAC policy training.
- Query-level paired patient-cluster bootstrap and paired sign-flip permutation support were added through `src/adapromptset/statistics/bootstrap.py` and `scripts/run_bootstrap.py`. The current v2 fixed-oracle pilot report writes `reports/bootstrap/bootstrap_summary.{json,md}` and `query_level_comparison.csv`. Its status is `PENDING_INSUFFICIENT_CLUSTERS`: the 10 reference episodes aggregate to 2 queries but only 1 patient cluster, observed oracle-vs-best-fixed mean delta is 0.005011, exact paired p-value is 0.5, and no formal CI or significance claim should be made from this pilot.
- Standardized gate-report export was added through `src/adapromptset/reporting/gate_reports.py` and `scripts/export_gate_reports.py`. It writes `reports/gates/gate_<letter>_<name>.{json,md}` for Gates A-I, with file hashes, command provenance, diagnostics, rule application, allowed next actions, prohibited next actions, and deviations. The current export records Gate A/B/C/D as PASS, Gate E as a rescue subset pass rather than a complete action-bank pass, Gate F as `FAIL_ORACLE_GAP_SMALL`, Gate G as `NOT_RUN_AFTER_GATE_F_PIVOT`, and Gate H/I as prohibited or blocked by the Gate F pivot. This consolidates evidence only; it does not reopen MoNuSAC policy training.
- Representation-pivot diagnostics were added through `src/adapromptset/scripts/analyze_representation_pivot.py` and `scripts/analyze_representation_pivot.py`. The report writes `reports/representation_analysis/representation_summary.json`, `episode_representation.csv`, and `representation_report.md` by joining the existing v2 oracle-gap utility matrix with candidate-pilot diagnostics. Current status is `PIVOT_REPRESENTATION_ANALYSIS`: mean regret to oracle is 0.005011, mean top-2 Dice margin is 0.020441, near-tie episode fraction is 0.4, oracle differs from best fixed in 0.5 of episodes, mean candidate point precision is 0.52, and mean component coverage is 0.0520. The resulting claim remains `oracle_analysis_only_insufficient_separation`; MoNuSAC policy training, post-hoc MoNuSAC action redesign, and test-time action search remain prohibited.
- Gate F failure diagnostics were added through `src/adapromptset/scripts/audit_gate_f_failure_diagnostics.py` and `scripts/audit_gate_f_failure_diagnostics.py`. The report writes `reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.{json,md}` from the existing candidate pilot and v2 utility matrix. Current diagnostics are coverage=`LOW_COMPONENT_COVERAGE` with max-pool mean component coverage 0.052005, proposal separation=`ACTIONS_SEPARATED` with mean top-2 Dice margin 0.020441 and near-tie episode fraction 0.4, and SAM sensitivity=`NO_HIGHER_CALL_GAIN` with high-call delta -0.020570. This closes the post-Gate-F diagnostic checks without reopening MoNuSAC policy training or post-hoc action-bank redesign.
- Representation-pivot paper assets were added to `scripts/export_oracle_analysis_tables.py` and the claim-free paper skeleton. The exporter now reads `reports/representation_analysis/representation_summary.json`, writes `paper/tables/table_representation_pivot.tex`, adds `RepresentationAnalysis*` macros to `paper/generated/oracle_analysis_macros.tex`, and records the representation claim plus prohibited actions in `reports/oracle_analysis_paper_assets.md`. `scripts/build_paper.py --mode submission --compile` now includes the representation table in `sections/05_results.tex` and produces `paper/main.pdf` without overfull boxes, undefined references, or simulated-placeholder hits; the representation-stage PDF was 95591 bytes with SHA256 `D1809296B349DC56983E841CC0587CFEF15CE7E39460277C261DE625BFE427B6`. This is still a pivot diagnostic asset, not an adaptive-policy success claim.
- Scientific claim auditing was strengthened in `src/adapromptset/scripts/audit_scientific_claims.py` and `scripts/audit_scientific_claims.py`. In addition to prohibited-token scans and Gate-F evidence checks, it now requires explicit positive boundary statements that oracle rows are analysis upper bounds rather than deployable outputs, background reference evidence is not explicit SAM negative prompting, multi-reference episodes are repeated measurements rather than independent samples, no in-domain retraining or transfer claim is made, and failure results/limitations are disclosed. The refreshed `reports/scientific_claim_audit.{json,md}` is `PASS` with zero violations under the current oracle-analysis-only skeleton.
- Final project handoff reporting was added through `src/adapromptset/scripts/export_final_report.py` and `scripts/export_final_report.py`. The report writes `reports/final_project_report.json` and `reports/final_project_report.md` by reading existing Phase 5, Gate F, fixed-oracle, representation, GlaS smoke, bootstrap, run-metadata, compute-ledger, and PDF artifacts. The current report status is `PIVOT_ORACLE_ANALYSIS`, claim boundary is `oracle_analysis_only`, Gate F is `FAIL_ORACLE_GAP_SMALL`, Dice oracle gap is 0.005011, AJI oracle gap is 0.003820, GlaS remains `PENDING_INSUFFICIENT_EPISODES`, bootstrap remains `PENDING_INSUFFICIENT_CLUSTERS`, and prohibited next actions include MoNuSAC policy training, post-hoc MoNuSAC action redesign, and test-time action search. This is a handoff/reproducibility report, not final project completion or paper submission readiness.
- Gate-P scientific claim auditing was added through `src/adapromptset/scripts/audit_scientific_claims.py` and `scripts/audit_scientific_claims.py`. The audit reads the final report, Gate F report, fixed-oracle summary, and representation summary, then scans reader-facing README/report/paper text for prohibited claim tokens and verifies the `oracle_analysis_only` boundary. The first real audit caught two negated occurrences of a leaderboard-performance phrase in the final report and limitations section; the generators were changed to avoid that token entirely. The current `reports/scientific_claim_audit.{json,md}` status is `PASS`, with Gate F pivot, policy-training prohibition, GlaS insufficient-episode status, and insufficient-cluster bootstrap status all verified. This is a machine claim-boundary audit, not Final Sign-off.
- After the Gate-P audit wording cleanup, `scripts/build_paper.py --mode submission --compile` regenerated `paper/main.pdf` as a 2-page claim-free oracle-analysis skeleton, 95612 bytes, SHA256 `4C6A283FD8BF00C94FF7AD9A899067CCE29C3C82BD20926C14FDFA8478207AFD`. `reports/final_project_report.json` now records that hash and includes `scripts\audit_scientific_claims.py` as the final reproduction command.
- Phase 0 scope/structure auditing was added through `src/adapromptset/scripts/audit_phase0_scope_structure.py` and `scripts/audit_phase0_scope_structure.py`. The report writes `reports/phase0_scope_structure/phase0_scope_structure.{json,md}` and records status `PASS_SCOPE_STRUCTURE`: 26 required paths exist, no structure paths are missing, RQ count is 3, hypothesis count is 5, GF-SAM/SynPo/AlignSAM/Point Prompt Defender boundaries are present, and sequential RL, continuous coordinate generation, and online SAM-response refinement are marked non-mainline. Toolchain status is now `PASS` after adding Ruff lint/format configuration to `pyproject.toml` and the environment spec; this closes the Phase 0 pyproject/environment/format/lint/pytest checklist item.
- Draft/submission paper mode separation was added to `src/adapromptset/scripts/build_paper.py`. Draft mode now writes `\simulatedtrue` plus a visible `DRAFT WATERMARK`; submission mode writes `\simulatedfalse`, omits the watermark, and keeps `guard_submission_placeholders` as the blocker for `SIMULATED PLACEHOLDER` / `simulatedtrue`. The refreshed `paper/main.tex` is submission-safe with `\simulatedfalse`, and `scripts\build_paper.py --mode submission --compile` regenerated the current two-page oracle-analysis skeleton PDF.
- Related-work boundary assets were added through `src/adapromptset/reporting/related_work.py`, `src/adapromptset/scripts/export_related_work.py`, and `scripts/export_related_work.py`. The export writes `reports/related_work_boundaries/related_work_boundaries.{json,md}` plus `paper/tables/table_related_work.tex`, covering GF-SAM, SynPo, AlignSAM, and Point Prompt Defender without claiming AdaPromptSet is the first learned prompting method. The paper skeleton now inputs the generated table from `sections/02_related.tex`; the table is formatted with short display labels and compact spacing so the regenerated PDF has no Overfull, undefined-reference, citation, or rerun warnings.
- Paper-section boundary auditing was added through `src/adapromptset/scripts/audit_paper_sections.py` and `scripts/audit_paper_sections.py`. The report writes `reports/paper_section_audit.{json,md}` and currently records status `PASS` with zero violations. It verifies the current oracle-analysis paper skeleton has a claim-free abstract, related-work boundary wording for GF-SAM/SynPo/AlignSAM/Point Prompt Defender, no end-to-end differentiable method claim, patient/query statistical-unit wording, Dice plus cost descriptors in results, limitations for action-bank upper bound/reference quality/frozen matcher irrecoverability/utility generation cost, and an oracle-analysis-only conclusion. `reports/final_project_report.{json,md}` now records this audit and includes `scripts\audit_paper_sections.py --project-root D:\SAM\adapromptset` in the reproduction command list. This closes current section-boundary text checks only; it is not a full 8-page final-paper sign-off.
- AJI oracle-gap reporting was threaded through `src/adapromptset/utility/action_execution.py`, `src/adapromptset/scripts/run_oracle_gap_pilot.py`, `src/adapromptset/scripts/analyze_fixed_oracle_pivot.py`, `src/adapromptset/reporting/gate_reports.py`, `src/adapromptset/scripts/export_oracle_analysis_tables.py`, and `src/adapromptset/scripts/export_final_report.py`. The refreshed Gate F JSON, fixed-oracle summary, oracle-analysis paper assets, and final report all record AJI gap=0.003820 alongside Dice gap=0.005011. `scripts/analyze_representation_pivot.py` and `scripts/run_bootstrap.py` wrappers now insert `src` into `sys.path`, matching the other reproduction wrappers. Full unittest discovery now runs 132 tests successfully; LaTeX warning scans for undefined references/citations/overfull/rerun and reader-facing prohibited-claim scans both return no matches.
- New-domain pilot readiness auditing was added through `src/adapromptset/scripts/audit_new_domain_pilot_readiness.py` and `scripts/audit_new_domain_pilot_readiness.py`. The report writes `reports/new_domain_pilot_readiness.{json,md}` and is read-only: it does not train policy, reopen MoNuSAC policy, or redesign the MoNuSAC action bank. Current status is `READY_WITH_BLOCKED_FORMAL_DATASETS`: GlaS is `BLOCKED_INSUFFICIENT_EPISODES` with 1 strict validation episode versus a 10-episode Gate F minimum; DDR is `READY_FOR_INFRASTRUCTURE_ONLY` using the already audited Stage-1 union / Stage-2 four-intent protocol; IDRID is `READ_ONLY_NON_TUNABLE`; CPM-17 is `DEFERRED_MISSING_OR_INCOMPLETE` with the local `cpm15` subset insufficient for formal evaluation. The final report now includes this audit and its reproduction command.
- Oracle-analysis figure export was added through `src/adapromptset/scripts/export_oracle_analysis_figures.py` and `scripts/export_oracle_analysis_figures.py`. It reads the real fixed-oracle `budget_pareto.csv` and `action_summary.csv`, writes `paper/figures/oracle_pareto.png`, `paper/figures/oracle_action_wins.png`, and `reports/oracle_analysis_figures.{json,md}` with source and figure SHA256 values. The claim-free paper skeleton now includes both figures via `graphicx`; the refreshed `paper/main.pdf` is still a 2-page oracle-analysis skeleton, now 224396 bytes with SHA256 `1B34DE77B0B7314D27A782D7C6C1BBFA66E88FD208DC6819451AF38EF59582DB`. Full unittest discovery now runs 135 tests successfully, and the LaTeX warning scan remains clean.
- Method overview Figure 1 export was added through `src/adapromptset/scripts/export_method_overview_figure.py` and `scripts/export_method_overview_figure.py`. It writes claim-safe TikZ to `paper/figures/method_overview.tex` plus `reports/method_overview_figure.{json,md}`, explicitly separating frozen DINOv2-L matcher, frozen SAM-B segmenter, train/validation-only utility labels, Gate-F-dependent learned utility predictor, and the test-time safe path. The claim-free paper skeleton now includes the Figure 1 input from `sections/03_method.tex`, and `reports/paper_section_audit.json` verifies `method_figure_frozen_learned_train_only=true`. The refreshed `paper/main.pdf` is a 3-page oracle-analysis skeleton, 247474 bytes with SHA256 `661FA24807E7C60E3DAD5B9B4502591B50EBF79D2C9618662CB5814F46B86EA9`; this is still not final 8-page paper acceptance. Full unittest discovery now runs 136 tests successfully, targeted Ruff passes, and reader-facing prohibited-claim plus LaTeX warning scans remain clean.
- Action-bank Figure 2 export was added through `src/adapromptset/scripts/export_action_bank_figure.py` and `scripts/export_action_bank_figure.py`. It reads the real `configs/actions/prompt_set_bank_v2.json`, writes claim-safe TikZ to `paper/figures/action_bank.tex`, and records `reports/action_bank_figure.{json,md}` with status `PASS_ACTION_BANK_FIGURE`, source bank hash `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`, and figure SHA256 `290AB62437347460BCB8F2D5BB85C21A4539B245180E67955C0A18FE3155F862`. The figure shows the current v2 MVP action-bank subset and explicitly marks KMeans, density, and GF-SAM-style graph families as deferred/missing rather than implemented. The paper skeleton now includes Figure 2 from `sections/03_method.tex`, `reports/paper_section_audit.json` verifies `method_action_bank_figure_current_subset=true`, and the refreshed `paper/main.pdf` is a 4-page oracle-analysis skeleton, 291717 bytes with SHA256 `26BFB8F8F5FB30113DC40EDFAC7462AFCCC107D8D4E57C891D9D4075CF8DDBDC`. Full unittest discovery now runs 137 tests successfully, targeted Ruff passes, scientific-claim audit remains `PASS`, and reader-facing prohibited-claim plus LaTeX warning scans remain clean.
- Oracle/privileged table-caption boundaries were tightened in `scripts/export_oracle_analysis_tables.py` and audited in `scripts/audit_paper_sections.py`. The generated oracle pivot and action summary captions now state that oracle columns/wins are analysis upper bounds or post-hoc labels, not deployable method outputs/selections; all current analysis tables state that no privileged-input row is included. `reports/paper_section_audit.json` records `oracle_privileged_caption_boundaries=true`, `reports/final_project_report.md` records Oracle/privileged captions=`PASS`, and the refreshed `paper/main.pdf` remains a 4-page claim-free oracle-analysis skeleton, 292020 bytes with SHA256 `A332ED507B81D886D27A3B4E5D9674440EF505AB9526425BEE5FD775EF06C34B`.
- PDF metadata/title/author auditing was added through `src/adapromptset/reporting/pdf_metadata.py`, `src/adapromptset/scripts/audit_pdf_metadata.py`, and `scripts/audit_pdf_metadata.py`. The paper generator now writes pdfTeX native `\pdfinfo` metadata without reintroducing `hyperref`; `pdfinfo paper/main.pdf` reports Title=`AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation`, Author=`Internal MVP Draft`, Subject=`Oracle-analysis-only AdaPromptSet MVP artifact after Gate-F pivot`, and the expected keywords. `reports/pdf_metadata_audit.json` records status `PASS`, `reports/final_project_report.md` records PDF metadata=`PASS`, and the refreshed `paper/main.pdf` remains a 4-page claim-free oracle-analysis skeleton, 292317 bytes with SHA256 `B6E8E628BA04CA64E8A1E6E45BE3A2C749E88C9030A832C8A77EE0EC365CB9C1`.

## 2026-07-19 Phase 2D GlaS Gate D Confirmation Preregistration

- Phase 2D was registered after Phase 2 closed at `STOP_GLAS_SEGMENTER_FAILURE`. It does not reopen MoNuSAC and does not delete or overwrite the original GlaS 6-case Gate D reports.
- The original SAM-B/SAM-L 6-case Gate D result is preserved as `EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`.
- The amendment rationale in `docs/glas_gate_d_amendment_rationale.md` records why an all-case-must-win rule is too strong for a 6-case execution/viability gate and why exactly one 30-query confirmation is allowed before any Gate E/F reopening.
- The confirmation protocol in `docs/glas_gate_d_confirmation_protocol_v1.md` freezes the 30-query official-train strict-validation pool, P3 primary prompt family, C1/C2 matched controls, P1/P2/P4/P5 secondary diagnostics, catastrophic failure definition, query-level paired comparison, and SAM-B/SAM-L selection rule.
- The configuration in `configs/segmenter/glas_gate_d_confirmation_v1.yaml` freezes `P3_component_grouped_centroids` as primary and sets C2 to `first_frozen_episode_support_mask`, so C2 uses the existing utility/reference support manifest and never uses query GT.
- The protocol snapshot, input hashes, and compute plan are stored under `reports/phase2_glas/gate_d_confirmation/`. SAM-B/SAM-L confirmation inference has not yet run and must wait for the preregistration commit `phase2d: preregister glas gate d confirmation`.
- After preregistration commit `f5c2de3`, `scripts/run_glas_gate_d_confirmation.py --project-root D:\SAM\adapromptset --device cuda:0` ran the full 30-query confirmation. Both SAM-B and SAM-L failed the preregistered P3 primary criterion: SAM-B margin `-0.157937`, win rate `0.433333`, catastrophic rate `0.666667`; SAM-L margin `-0.096909`, win rate `0.466667`, catastrophic rate `0.400000`. Coordinate and cost audits passed, and testA/testB access remained `false`.
- The final Phase 2D state is `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`. Gate E/F/G/H/I remain closed/prohibited for GlaS, and no second Gate D criterion revision is allowed.

## 2026-07-19 Phase 3 GlaS Region Prompt Preregistration

- Branch `phase3-glas-region-prompts` was created from Phase 2D artifact commit `93b319e`.
- A CPU-only interpretation audit was added through `src/adapromptset/scripts/audit_phase3_interpretation.py` and `scripts/audit_phase3_interpretation.py`. It writes `reports/phase3_interpretation_audit/*` without new SAM inference and relabels C2 scientifically as `alternative_reference_control`.
- The historical Phase 2D operational state remains `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`, while the Phase 3 interpretation state is `PIVOT_GLAS_REGION_PROMPT_REQUIRED`.
- A 40/15 split inside the 55-image GlaS utility pool was added through `src/adapromptset/scripts/build_glas_phase3_region_splits.py`. It writes `manifests/phase3/glas_region_calibration_40.csv`, `glas_region_selection_15.csv`, and `glas_region_references.csv`; testA/testB access remains false.
- Region bank v1 was frozen in `configs/actions/glas_region_bank_v1.{yaml,json}` with R00-R09 automatic actions and U01-U03 privileged upper bounds. The strict validation pool cannot select thresholds or add actions.
- Phase 3 protocol, claim registry, exposure audit, gate status, compute ledger, protocol snapshot, and input hashes were created before any new SAM region-prompt inference.
- After preregistration commit `471f529`, frozen DINOv2-L/local-peaks candidate evidence was generated for the 40/15 internal pool under `reports/phase3_glas/region_candidate_calibration/`.
- Gate R1 was evaluated under `reports/phase3_glas/gate_r1_region_proposal/` without SAM inference. All R1 families failed the preregistered thresholds; the final Phase 3 state is `STOP_GLAS_REGION_PROPOSAL_FAILURE`.
- Because R1 failed, SAM-B R2, R3 oracle analysis, router training, and testA/testB evaluation remain blocked/prohibited.

## 2026-07-19 Phase 4+ Research Program Preregistration

- Branch `phase4plus-submission-program` was created from Phase 3 final commit `9f33382`.
- Phase 4+ is not a continuation of same-validation GlaS method repair. It is a research-program layer for submission-readiness, with Phase 4 limited to read-only evidence audit before any new matcher/segmenter inference.
- Frozen states are preserved: MoNuSAC `PIVOT_ORACLE_ANALYSIS`, GlaS grouped sparse point `STOP_GLAS_GROUPED_POINT_PROMPT_PATH_CONFIRMED`, GlaS region scientific state `STOP_GLAS_REGION_LOCALIZATION_ADVANTAGE_NOT_ESTABLISHED`, and GlaS testA/testB `LOCKED_UNUSED`.
- Routes A/B/C are registered for evidence-based selection after Phase 4 audit: Promptability-Aware Routing, Reference Promptability Benchmark, and Direct Reference Segmentation.
- Phase 4 read-only audit produced `PHASE4_AUDIT_SUPPORTS_BENCHMARK_TRACK`. It found that R03's high absolute box IoU is confounded by large automatic box scale and high size-matched random-box control performance; R00 is high precision but low recall. Route B was selected as primary, Route A as fallback, and Route C as deferred.

## 2026-07-19 Route B DDR Development Manifest Gate

- `docs/route_b_ddr_dev_protocol_v1.md` registered DDR train/valid as the first fresh-data development entry for Route B and kept DDR test locked.
- `scripts/build_route_b_ddr_manifests.py --project-root D:\SAM\adapromptset --splits train valid` generated all-intent and positive-query manifests under `manifests/phase4plus/`.
- The report `reports/route_b_ddr_manifest/ddr_route_b_manifest_report.json` records `PASS_DDR_ROUTE_B_DEV_MANIFEST`, train 383 images / 964 positive image-intent queries, valid 149 images / 401 positive image-intent queries, and no issues.
- The builder rejects any requested split outside `train`/`valid`; DDR test files were not read and no matcher/segmenter inference was run.

## 2026-07-19 Route B Metric/Control Freeze

- `docs/route_b_benchmark_metric_control_protocol_v1.md` and `configs/experiments/route_b_promptability_benchmark_v1.json` froze the Route B row families, prompt representations, controls, metrics, statistical units, and query-level output schema.
- `scripts/audit_route_b_benchmark_freeze.py --project-root D:\SAM\adapromptset` wrote `reports/route_b_benchmark_freeze/route_b_benchmark_freeze_audit.{json,md}` with status `PASS_ROUTE_B_B1_PREREQ_FREEZE`.
- The freeze authorizes only development evidence generation on legal train/valid data. It does not authorize formal test access, a method-positive claim, or a submission-ready claim.

## 2026-07-19 Route B DDR Episode Manifest Gate

- `docs/route_b_ddr_episode_protocol_v1.md` froze same-intent reference/query episode construction for DDR Route B development.
- `scripts/build_route_b_ddr_episodes.py --project-root D:\SAM\adapromptset --references-per-query 3` generated `manifests/phase4plus/ddr_route_b_train_reference_query_episodes.csv` and `manifests/phase4plus/ddr_route_b_valid_reference_query_episodes.csv`.
- Status is `PASS_DDR_ROUTE_B_EPISODES`: train has 964 covered positive queries / 2892 reference rows; valid has 401 covered positive queries / 1203 reference rows.
- Valid references come only from train; train references exclude the query image; DDR test files were not read.

## 2026-07-19 Route B DDR Matcher Development Smoke

- `scripts/run_route_b_ddr_matcher_smoke.py --project-root D:\SAM\adapromptset --device auto` ran one DINOv2-L matcher smoke on the first DDR valid episode row.
- Query: `DDR:valid:007-2846-100:MA`; reference: `DDR:train:007-4668-200:MA`.
- Output: `reports/route_b_ddr_development_smoke/route_b_ddr_matcher_smoke.json` and `route_b_ddr_matcher_candidates.csv`.
- Status: `PASS_ROUTE_B_DDR_MATCHER_SMOKE` for pipeline execution; point precision and component coverage were both 0.0.
- Interpretation: this is development-smoke/failure-taxonomy evidence only. It is not a method-positive result and does not authorize DDR test access or a benchmark-ready claim.

## 2026-07-19 Route B DDR Stratified Matcher Development Evidence

- `scripts/run_route_b_ddr_stratified_matcher.py --project-root D:\SAM\adapromptset --queries-per-intent 2 --device auto` ran DINOv2-L matching on 16 DDR train/valid query-intent units: two per split and lesion intent.
- Output: `reports/route_b_ddr_stratified_matcher/stratified_matcher_summary.json`, `stratified_matcher_summary.md`, `selected_episodes.csv`, and `query_metrics.csv`.
- Status: `PASS_ROUTE_B_DDR_STRATIFIED_MATCHER_SMOKE` for legal execution; mean point precision `0.0125`, mean component coverage `0.0025`, zero precision fraction `0.9375`.
- Interpretation: the candidate-only DINOv2 matching route is mostly nonlocalizing on this DDR development slice. Route B should now emphasize failure taxonomy, matched controls, and benchmark diagnostics rather than expanding this exact smoke as a method-positive path.

## 2026-07-19 Route B DDR Failure Taxonomy

- `scripts/analyze_route_b_ddr_failure_taxonomy.py --project-root D:\SAM\adapromptset` produced CPU-only control-aware taxonomy from the 16-row stratified matcher evidence.
- Output: `reports/route_b_ddr_failure_taxonomy/failure_taxonomy_summary.json`, `failure_taxonomy_summary.md`, and `failure_taxonomy.csv`.
- Status: `PASS_ROUTE_B_DDR_FAILURE_TAXONOMY`; mean DINOv2 point precision `0.0125`, mean random point precision `0.0`, mean DINOv2 component coverage `0.0025`, mean random component coverage `0.0`.
- Failure flags: small sparse target `16/16`, diffuse matching map `16/16`, no candidate hit `15/16`, no component covered `15/16`, random not beaten `15/16`.
- Interpretation: this strengthens the benchmark/failure-taxonomy direction and closes candidate-only DINOv2 matching as a method-positive DDR route unless a separately registered representation/control redesign is justified.


## 2026-07-20 Phase X Action-Space Adequacy / Search-Oracle Audit Preregistration

- Branch `phaseX-action-space-oracle-audit` created for Phase X.
- Phase X is a development-stage action-space audit on MoNuSAC train/validation only. It does not reopen Phase 4+ Route B work and does not access formal test.
- Historical Oracle claim renamed to `FINITE_BANK_ORACLE_GAP_SMALL`. Historical anchors: v2 bank hash `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`, best fixed `B_top5_pos3neg`, mean best fixed Dice 0.371238, mean finite-bank oracle Dice 0.376250, gap 0.005011; both ends sit in a low-absolute-performance regime (~0.37 Dice), so the small gap cannot establish action-space saturation.
- Phase X preregistration docs created: `docs/phaseX_goal_read_receipt.md`, `docs/action_space_problem_statement.md`, `docs/historical_oracle_correction.md`, `docs/dense_prompt_scaling_protocol.md`, `docs/search_oracle_protocol.md`, `docs/action_space_saturation_protocol.md`, `docs/formal_test_lock_notice.md`.
- Phase X reports scaffolding created: `reports/phaseX_gate_status.md`, `reports/phaseX_compute_ledger.csv`, `reports/phaseX_protocol_snapshot.json`, `reports/phaseX_input_hashes.json`.
- Historical bank audit created under `reports/phaseX_historical_bank_audit/`: `action_bank_exact_definitions.{csv,md}`, `fixed_action_absolute_metrics.csv`, `fixed_action_rankings.csv`, `old_bank_oracle_query_metrics.csv`, `old_bank_oracle_summary.md`, `cost_distribution.csv`, `historical_claim_correction.md`, `audit_summary.json`.
- Frozen grids: K+ in {1, 3, 5, 10, 20, 30, 50, 75, 100}; K- in {0, 1, 3, 5, 10, 20, 30, 50}; groupings = {G_ONE_GROUP, G_ONE_POINT_PER_GROUP, G_FIXED_CHUNK_5, G_FIXED_CHUNK_10, G_FIXED_CHUNK_20, G_KMEANS_3, G_KMEANS_5, G_KMEANS_10, G_KMEANS_20, G_SPATIAL_GRID, G_GRAPH, G_CANDIDATE_COMPONENT}; call budgets = {1, 3, 5, 10, 20}; point budgets = {5, 10, 20, 50, 100, 150}; nested banks A_small/A_medium/A_large with limits ({'max_positive': 5, 'max_negative': 4, 'max_calls': 3}, {'max_positive': 20, 'max_negative': 10, 'max_calls': 5}, {'max_positive': 100, 'max_negative': 30, 'max_calls': 20}); search beam_widths = {4, 8}, depths = {10, 20, 50}, call budgets = {1, 3, 5, 10, 20}, point budgets = {10, 20, 50, 100}; adaptive-value Dice threshold = 0.01; min query fraction = 0.25; GPU-hour cap = 132.0.
- No new SAM/matcher inference has been run. Preregistration commit must precede candidate coverage scaling, dense prompt scaling, search oracle, and any policy reopening decision.


## 2026-07-21 Phase Y Fresh-Query Policy Feasibility Preregistration

- Branch `phaseY-policy-feasibility` created from Phase X latest commit `ae883df`.
- Phase Y is a fresh-query confirmation, oracle stability, and deployable policy feasibility study on MoNuSAC train/validation only. It does NOT access formal test and does NOT reopen GlaS/DDR work.
- Phase X exploratory evidence preserved but labeled `EXPLORATORY_PHASEX_EVIDENCE`: Top20+N1 fixed=0.5411, A_large bank oracle=0.6119, Search Oracle=0.7256, 10 episodes. These are excluded from confirmatory statistics.
- MoNuSAC data audit completed (`reports/phaseY_data_audit/`): 310 total cases (209 train, 101 test LOCKED). Phase X exposed 12 cases across 10 patients. Critical findings: (1) Phase X references train_004 and train_005 are both from patient TCGA-55-1594 (single-patient reference pool); (2) Phase X queries train_111 and train_113 are both from patient TCGA-UZ-A9PO (patient-level leakage risk). Phase Y splits at patient level to eliminate both issues.
- Fresh pool: 36 patients, 161 cases. Patient-level split (seed=20260721): Policy Train=80 cases (18 patients), Policy Validation=41 cases (10 patients), Confirmatory Holdout=40 cases (8 patients). Patient-level disjoint verified.
- Reference Sets A and B built: 3+3 non-overlapping references per query, patient-disjoint from query, sourced from train fold only. 483 rows each (161 queries × 3). Reference audit PASS.
- Full audit bank frozen (`configs/actions/phaseY_full_single_call_bank.yaml`): 210 actions = 3 pos_selectors × 10 K+ × (1 N_NONE@K0 + 2 neg_selectors × 3 K>0) × G_ONE_GROUP × 1 call. Compact policy bank is placeholder pending Pareto pruning on train-only (post-preregistration, section 7.2).
- 11 preregistration docs created under `docs/phaseY_*.md`: goal_read_receipt, scientific_questions, data_split_protocol, reference_protocol, single_call_action_bank, oracle_stability_protocol, bank_policy_protocol, point_subset_protocol, statistics_protocol, claim_registry, test_lock_notice.
- 5 structured reports created: `reports/phaseY_gate_status.md`, `reports/phaseY_compute_ledger.csv`, `reports/phaseY_protocol_snapshot.json`, `reports/phaseY_input_hashes.json`, `reports/phaseY_reference_audit.json`.
- Main question: "Can a query-conditioned, single-call prompt-configuration policy outperform the strongest fixed dense prompt under a strictly matched deployment budget?"
- 4 Gates pre-registered: Y1 (stable heterogeneity, gap>=0.015, optimism<=60%), Y2 (search label stability, cross-ref>=0.03, optimism<=50%), Y3 (bank policy, Delta>=0.010, recovery>=30%), Y4 (point-subset, Delta>=0.015 or >=0.0075 over bank).
- Model ladders: M0-M5 (bank-action) and P0-P5 (point-subset). PPO/DQN forbidden.
- Compute cap: 155 GPU-hours total.
- No new SAM/matcher inference has been run. Preregistration commit must precede Gate Y1 utility matrix, search oracle Y2, policy training Y3/Y4, and holdout evaluation.

## 2026-07-21 Phase Y Gate Y1 — Fresh-Query Action Heterogeneity (FAILED)

- Y1 validation utility matrix completed: 41 validation queries × 2 reference sets (A, B) × 3 references = 246 pairs, 210 actions per pair, 51,660 action evaluations total. Runtime: ~3 hours on GPU (42.4s/pair average). DINOv2-L features cached per case_id, SAM-B single-call one-group configuration.
- Gate Y1 decision: `PHASEY_GATE_Y1_ORACLE_GAP_SMALL` — 2 of 6 conditions passed.
- Strongest fixed action: `Y_P_TOPK_K50_N0_N_NONE_G1_C1` (K+=50, K-=0, P_TOPK, G_ONE_GROUP, 1 call), mean Dice = 0.4338 on Reference Set A.
- Raw oracle A: 0.5007 (gap = 0.0669). Raw oracle B: 0.4836.
- Cross-reference oracle A→B: 0.4258 (BELOW fixed 0.4338). Cross-reference oracle B→A: 0.4539.
- Stable oracle = mean(cross A→B, cross B→A) = 0.4399. Stable oracle gap = 0.0061 (threshold: ≥0.015) → FAIL.
- Oracle optimism A = 0.0749 (optimism ratio = 1.12, threshold: ≤0.60) → FAIL. Optimism exceeds raw gap, meaning cross-reference Dice is below fixed Dice.
- Bootstrap 95% CI on stable oracle gap: [-0.057, 0.070] (threshold: lower >0) → FAIL.
- diff_from_fixed_rate = 0% (threshold: ≥30%) → FAIL. No query's stable best-action family differs from strongest fixed.
- median Spearman ρ = 0.745 (threshold: >0.30) → PASS. Action rankings are correlated across references.
- All actions use 1 SAM call → PASS.
- Key finding: The Phase X raw oracle gap of 0.0708 (from 10 exploratory queries) does NOT persist under fresh-query cross-reference evaluation. The stable oracle gap is only 0.0061, well below the 0.015 threshold. The raw gap was almost entirely Winner's Curse / selection bias. When the oracle action selected on Reference Set A is evaluated on Reference Set B, it performs WORSE than the strongest fixed action (cross A→B = 0.4258 < fixed = 0.4338).
- Routing per section 19.1: Bank Policy (Y3) and Holdout are PROHIBITED. Search Oracle stability audit (Y2) is ALLOWED for diagnostic purposes. If Y2 also fails, final state = `PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY`.

## 2026-07-21 Phase Y Gate Y2 — Search Oracle Stability (FAILED)

- Y2 Search Oracle Stability audit completed: 41 validation queries, GT-guided greedy add/remove positives + add negatives over dual reference sets A/B with cross-evaluation.
- Runtime: 29357.5s (~8.15 GPU-hours), 148757 total SAM calls (search budget, not deployed calls). Mean ~12 min/query.
- Gate Y2 decision: `PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT` — 1 of 5 conditions passed.
- Raw search Dice (mean over A, B): 0.5947. Raw search gap over fixed = 0.1609.
- Cross-reference search Dice (A→B, B→A averaged): 0.4105 (BELOW fixed 0.4338!) → cond1 FAIL (need ≥ fixed + 0.03 = 0.4638).
- Search optimism = 0.1843, optimism ratio = 1.1449 (threshold: ≤0.50) → cond3 FAIL. Optimism exceeds the entire raw gap, meaning raw search Dice is almost entirely reference-overfit artefact.
- Paired bootstrap 95% CI on (cross - fixed): [-0.089, 0.040] (threshold: lower >0) → cond2 FAIL.
- Median |K_A - K_B| = 6.0 (≤8) → cond4 PASS. K is stable across references even though Dice is not.
- A-search prompt beats strongest fixed on B: 26.8% of queries (threshold: ≥70%) → cond5 FAIL.
- Mean point overlap (Jaccard) = 0.332, mean rank overlap = 0.558, mean spatial overlap (IoU) = 0.898.
- Key finding: The Phase X search oracle Dice of 0.7256 (from 10 exploratory queries) does NOT persist under cross-reference evaluation. Raw search Dice on the full 41-query validation set is only 0.5947, and cross-reference Dice collapses to 0.4105 — below the strongest fixed action (0.4338). The search oracle overfits to the reference set's candidate pool and cannot generalize to a different reference.
- Final routing per section 19.1: Y1 FAIL + Y2 FAIL → final state = `PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY`. Bank Policy (Y3), Point-Subset (Y4), and Confirmatory Holdout remain PROHIBITED. Formal test remains LOCKED.
- Allowed claim per section 21 (情况 A): "The large raw oracle gap observed in the exploratory 10-query audit did not persist under fresh-query and cross-reference evaluation."
- Total Phase Y GPU usage: 11.28 GPU-hours (Y1: 3.13h + Y2: 8.15h), well under the 155 GPU-hour cap.


## 2026-07-22 Phase R — Preregistration Complete

Directive: `AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md`
Branch: `phaseR-reference-stable-feasibility`

### Phase R0 Historical Audit
- 8 audit files produced under `reports/phaseR_historical_audit/`
- Mean best-worst Dice gap: 0.1874
- Mean reference std: 0.0687
- 82.9% queries with gap >= 0.05
- Reference instability strongly confirmed

### Reference Pool Audit
- Pool size: 80 cases, 18 patients
- Min available refs: 70 (train), 80 (validation)
- Status: PHASER_REFERENCE_POOL_SUFFICIENT

### Manifests
- 5 train resampling seeds (2400 rows total)
- 1 validation fixed A/B (246 rows)
- All verified: A/B non-overlap, patient-disjoint, seed diversity

### Frozen Configs
- Prompt configs: P20-N1, P30-N0, P50-N0
- Consensus methods: B0-B7
- beta grid: [0.25, 0.5, 1.0]
- tau grid: [0.50, 0.67, 0.83]
- Statistics: 4 Gates with frozen thresholds

### Preregistration Documents
- 9 docs/phaseR_*.md files
- 3 reports/phaseR_* files (md, csv, json)
- 4 configs/frozen/phaseR_*.yaml files
- Input hashes generated

### Compute
- Used: 0.00 GPU-hours (CPU-only so far)
- Budget: 85 GPU-hours

### Next
- Gate R1: Reference Instability Replication (train=80, 5 seeds)


## 2026-07-22 Phase R Gate R1 — Reference Instability Replication (PASSED)

- R1 inference completed: 80 train queries × 5 resampling seeds × 6 references × 3 prompt configs = 7200 SAM calls.
- Runtime: 501.7s on RTX 5090 (0.139 GPU-hours, well within the 25 GPU-hour R1 cap).
- Gate R1 decision: `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED` — all 5 conditions passed.
- Replication statistics:
  - % queries with best-worst Dice gap ≥ 0.05: 0.8875 (need ≥ 0.30) — PASS
  - Mean reference std: 0.0737 (need ≥ 0.02) — PASS
  - Direction consistent across 5 seeds: YES — PASS
  - Sensitive config: P20-N1 (K+=20, K-=1, N_BOTTOMK) — PASS
  - Not a code/coordinate/leakage error: YES — PASS
- The instability observed in the Phase R0 historical audit (mean best-worst gap=0.1874, 82.9% queries gap≥0.05) replicates on fresh resampled references with 5 seeds. The mean reference std (0.0737) is 3.7× the cond2 threshold.
- Output artifacts under `reports/phaseR_gateR1/`: `per_reference_metrics.csv` (7200 rows), `per_query_instability.csv` (1200 rows), `per_query_aggregate.csv` (240 rows), `gateR1_summary.{json,md}`, `run_metadata.json`.
- Cumulative GPU usage: 0.139 / 85 GPU-hours.
- Per directive section 23 step 14: proceed to deterministic consensus (B1-B6) on train=80 to select β/τ, then Gate R2 on validation=41.

## 2026-07-22 Phase R Gate R2 — Deterministic Consensus Feasibility (FAILED)

- R2 inference completed: 80 train queries × 5 seeds × 3 configs × 10 method variants (B1-B6 with β∈{0.25,0.5,1.0}, τ∈{0.50,0.67,0.83}) plus 41 val queries × 3 configs × multiple methods = 3138 SAM calls total (1539 new train + 1599 val; train reused R1 per-reference Dice for B1).
- Runtime: 271.7s on RTX 5090 (0.0755 GPU-hours).
- Best config and method selected on train: P30-N0 + B2_mean (β=0.25, τ=0.83 frozen for val).
- Validation results (frozen β/τ):
  - Mean consensus Dice (B2_mean, P30-N0): 0.4756
  - Mean B1 random single Dice: 0.4297
  - Δ Dice: +0.0460
  - Paired bootstrap 95% CI: [0.0125, 0.0835]
  - Variance(single)=0.0543, Variance(consensus)=0.0571
  - Variance reduction: -0.0506 (variance INCREASED by 5.06%)
  - Catastrophic failure rate (Dice<0.1): consensus 0.0732 vs B1 0.1220 (improved)
  - A/B prediction overlap: consensus 0.6075 vs single 0.5942 (improved)
  - Consistent direction across configs: 3/3
- Gate R2 decision: `PHASER_GATE_R2_CONSENSUS_NO_GAIN` — 7 of 8 conditions passed, 1 failed.
  - Cond1 Δ Dice ≥ 0.010: PASS (0.0460)
  - Cond2 bootstrap CI lower > 0: PASS (0.0125)
  - Cond3 variance reduction ≥ 25%: **FAIL** (-0.0506)
  - Cond4 catastrophic rate not increase: PASS (0.0732 vs 0.1220)
  - Cond5 A/B overlap improve: PASS (0.6075 vs 0.5942)
  - Cond6 single SAM call: PASS (by construction)
  - Cond7 no query GT used: PASS (by construction)
  - Cond8 ≥ 2 of 3 configs consistent: PASS (3/3)
- Interpretation: Deterministic consensus aggregation improves mean Dice, catastrophic-failure rate, and A/B prediction overlap, but it does NOT reduce per-query variance. The central R2 hypothesis (consensus reduces reference-induced variance) is therefore not supported. The Dice gain appears to come from averaging the contrast maps, which also smooths the failure mode rather than eliminates it.
- Per directive §12 (R3): "只有 Gate R2 通过才训练" — R3 reliability estimator training is BLOCKED.
- Per directive §13 (R4): "只有 R2 或 R3 通过" — R4 holdout is BLOCKED.
- Per directive §14 (GlaS): "只有 MoNuSAC R2 通过后运行 GlaS" — GlaS secondary diagnostic is BLOCKED.
- Per directive §22: formal test remains LOCKED and UNUSED.
- Output artifacts under `reports/phaseR_gateR2/`: `train_consensus_metrics.csv`, `train_selection_summary.json`, `val_consensus_metrics.csv`, `val_per_reference_metrics.csv`, `val_cross_set_stability.csv`, `gateR2_summary.{json,md}`, `run_metadata.json`.
- Cumulative GPU usage: 0.2153 / 85 GPU-hours (R1: 0.1394 + R2: 0.0755).
- Per directive section 23 step 21: proceed to final report generation (18 tables, 20 figures) and Phase R final state.

## 2026-07-22 Phase S Preregistration — Multi-Track Reference-Robust Research

- Branch `phaseS-multitrack-reference-robust-research` created from Phase R commit c35d5e9.
- Phase S directive `research.md` defines a 4-track research program (A/B/C/D) with 210 GPU-hour cap, 20 tables, 24 figures, and strict preregistration.
- Created 11 control documents under `docs/`: goal read receipt, research charter, track registry, hypothesis registry, data roles, exposure map, metric definitions, model registry, claim registry, publication readiness, test lock notice.
- Created 7 report documents under `reports/`: program status (.md + .json), compute ledger, decision tree, risk register, protocol snapshot, input hashes.
- Starting scientific state: `Reference instability confirmed + deterministic consensus utility gain confirmed + stability reduction unresolved + learned reliability scientifically motivated` (Phase R is NOT a failed direction).
- Data roles frozen: MoNuSAC train=80 / val=41 / holdout=40 / test=101 locked; GlaS 55/30; DDR 383/149; Phase X 10 queries excluded from primary stats.
- Track A must run first (metric correction + stability benchmark, 20 GPU-hour cap). Track B (reliability, 35h) only if A passes. Track C (set models, 45h) only if B plateaus. Track D (direct segmentation, 50h) only if B/C fail but matching evidence has value.
- Per directive §24: do not stop all research because one Gate fails; do not claim publishable on small positive results.
- Formal test locked for entire Phase S program.
- No new SAM inference in this commit; preregistration only.
