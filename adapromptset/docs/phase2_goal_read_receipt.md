# AdaPromptSet Phase 2 Read Receipt

Date: 2026-07-19
Branch: `phase2-glas-strict`
Git commit at protocol start: `016daa85a785ed9c6c272326b93977f0f70cbdfa`
Workspace: `D:\SAM`
Project root: `D:\SAM\adapromptset`
Protocol version: `adapromptset.phase2.glas_strict.v2`

## Objective Read

The Phase 2 objective is `AdaPromptSet-v2: GlaS Strict Diagnostic and Policy Gate Re-evaluation`.

This is not a restart from zero and not a mechanical completion pass over all old unchecked boxes. The previous MoNuSAC result is frozen as `PIVOT_ORACLE_ANALYSIS`. Phase 2 asks whether GlaS gland segmentation contains enough query-level prompt-set heterogeneity to justify a later cost-aware adaptive action selector.

## Required Documents Read

| Path | SHA256 |
| --- | --- |
| `AdaPromptSet_Codex_Response.md` | `B9C5F6D4EAA3BB5481D062A943218718954415FDF6E3F78CD0EBA1E97818482E` |
| `open_questions_and_gate_extensions.md` | `4F0997D5C1B5F11C381E3C55C05D5D8E507217300AA61A2512249967029F2EFF` |
| `execution.md` | `006353509203DC77DEC6D918B3D714C1AA5C902966FB090AE0233E40E2BA94DE` |
| `checklist.md` | `AA0589B87995E25C883ACB826587DEF3B4AD51791F127832BB30D5BA66F456BF` |
| `docs/goal_progress_nodes_and_skipped_assets_20260719.md` | `14254FD9D68F8B79BB69BAB1F48FD2DB3A916AB7AE4FEB6126BD39E9D077D8FD` |
| `reports/decision_checkpoint/phase5_decision.json` | `B6EC74EA3AF9984EA37F17EC38C35DDD67758F94D88E1FE9A1764143768FBE90` |
| `reports/gates/gate_f_oracle_gap.json` | `D70A77F86A34A526834A6DFB35AB88220E6C6D667E1900A275966864705FDEA2` |
| `reports/final_project_report.json` | `4F7408860F959C3F7E01E296726306D50BDBB53E450CFFF2C7BC0920989E9CB0` |
| `reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.json` | `F046ED0A3F3F730473A5D4579EC635A0FE038D7C38872ED62E65FCFCED438513` |
| `reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.md` | `8F48E876D89F1074AA70F4FCDBD0536AAC9FE1010CD22F6E5274329318571F9B` |
| `reports/representation_analysis/representation_summary.json` | `0F4F941602F08653211A036563F61C3118E690304E06A17A15B2C48D122D4E1C` |
| `reports/representation_analysis/representation_report.md` | `DE219468D82B203FEF720EEB2739BE42A17EE9CAAC3EA147D30DEE9C71508310` |
| `reports/representation_analysis/episode_representation.csv` | `40F61334A498FBE052A7A0A07E2DD0B8CDE2C5A0384BA062E0D82486D33BD8EC` |
| `reports/new_domain_pilot_readiness.json` | `B4B3F98D2802A933FA41D2CECFB63333140247F2A98AC3D8F0B0A3178133EB40` |

## Frozen MoNuSAC Facts

| Item | Value |
| --- | --- |
| Phase status | `PIVOT_ORACLE_ANALYSIS` |
| Gate F status | `FAIL_ORACLE_GAP_SMALL` |
| MoNuSAC v2 episodes | `10` |
| MoNuSAC v2 utility records | `120` |
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | `0.37123847211635835` |
| Mean oracle Dice | `0.376249802278007` |
| Dice oracle gap | `0.0050113301616486194` |
| Gate F minimum Dice gap | `0.010` |
| Mean best fixed AJI | `0.22834095079987268` |
| Mean oracle AJI | `0.23216115704701398` |
| AJI oracle gap | `0.0038202062471413` |
| Current MoNuSAC claim | `oracle_analysis_only` |

## Phase 2 Local Environment

| Item | Value |
| --- | --- |
| Python | `3.11.15` |
| PyTorch | `2.11.0+cu128` |
| CUDA available | `true` |
| CUDA runtime | `12.8` |
| GPU | `NVIDIA GeForce RTX 5090 D v2` |
| Matcher | `DINOv2-L`, frozen |
| Segmenter | `SAM-B`, frozen |
| DINOv2-L checkpoint observed | `D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth` |
| SAM-B checkpoint observed | `D:\SAM\assets\checkpoints\sam_vit_b.pth` |

## GlaS Data Facts at Phase 2 Start

| Item | Value |
| --- | --- |
| Local root | `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` |
| Official train image-mask pairs | `85` |
| Official testA image-mask pairs | `60` |
| Official testB image-mask pairs | `20` |
| Grade metadata | `Grade.csv` present |
| GlaS grade counts | `benign=74`, `malignant=91` across all 165 rows |
| Official train patients | `15` |
| Official test patients | `12` |
| Train/test patient-ID overlap | `1,2,3,4,5,7,9,10,11,13,15` |

## Phase 2 Allowed Changes

- Create Phase 2 protocol, audit, manifest, report, paper_phase2, and cache artifacts.
- Build a new GlaS official-train-only 55/30 diagnostic split.
- Use only official train images for policy/action utility training and strict validation before any formal test.
- Preserve official testA/testB from action, threshold, architecture, or paper-structure selection.
- Add prompt-statistics and reference-sensitivity exporters if they are diagnostic-only.
- Fix clean LaTeX build issues without altering scientific claims.

## Phase 2 Prohibited Changes

- Do not modify `prompt_set_bank_v2` for MoNuSAC.
- Do not resample observed MoNuSAC episodes.
- Do not lower the MoNuSAC Gate F threshold.
- Do not train a MoNuSAC adaptive policy.
- Do not use GlaS testA/testB for action-bank design, threshold tuning, model selection, or paper claim selection.
- Do not train any GlaS policy before Gate F returns `CONTINUE_GLAS_POLICY`.
- Do not claim universal, model-agnostic, or broad transfer behavior.

## Phase 2 Manifest Built

The Phase 2 GlaS split builder generated the following artifacts after protocol setup:

| Artifact | SHA256 |
| --- | --- |
| `manifests/phase2/glas/case_assignments.csv` | `0E3B35444993DE858DA59F92E1F00DB126A5E7A85E2FC77252C347FF3E322A34` |
| `manifests/phase2/glas/episodes_train.csv` | `D7184FBECC670F55038BCC2599C08B6CF800845AB2EAD430116C0F017D34E343` |
| `manifests/phase2/glas/episodes_val.csv` | `18291E9E9A1CFACCCC4395B2333B57034D65C3DF8AC5ED5AA1DEFF74581AF3EA` |
| `manifests/phase2/glas/locked_test_cases.csv` | `77D1D210B8A187873D4E1D93135228AE52510DD6D4338E462AA644DF9E71E260` |
| `manifests/phase2/glas/protocol_metadata.json` | `28AC00279B7F76F26B43EC74262B5652DA438BCC8C8C7C6B8698623B280477BF` |
| `reports/phase2_glas/split/split_report.json` | `7E3FA79B01F085715A586186432BB3C61C32BAA164371BC91A76967EDBD9EC3E` |

Manifest checks:

- phase2 roles: `utility_train=55`, `strict_val=30`, `locked_testA=60`, `locked_testB=20`;
- episode counts: `train=165`, `val=90`;
- query counts: `train=55`, `val=30`;
- independent leakage check found `BadCount=0` for support-not-train, self-reference, and test-query leakage;
- no testA/testB episode file was generated.

## Gate C Candidate Diagnostics

Gate C was run after the Phase 2 split was corrected to use GlaS-root-relative paths.

### Frozen Candidate Generator

| Artifact | SHA256 |
| --- | --- |
| `reports/phase2_glas/candidates/candidate_pilot_summary.json` | `C350FCA9FAFDBB75DE25C98BAAE7894EB5FC2571E5D0CFBE048CC8D97EF718BF` |
| `reports/phase2_glas/candidates/gate_c_summary.json` | `194DEB4A47609766E245F7FD7C70F64F45D3C2046AA8786570998B456793C7AD` |
| `reports/phase2_glas/candidates/gate_c_summary.md` | `7A80E166861520E9D09EC1CA20CC653178BCD92EE67F42FE1EA410AB2F0945EB` |
| `paper_phase2/figures/candidate_diagnostics/precision_vs_pool.png` | `4072FAFE6BC07DE02A734FF37F3A91CC79C47FCA846B4ED5F9B2E8F16584A831` |
| `paper_phase2/figures/candidate_diagnostics/coverage_vs_pool.png` | `8895F76801D10384A954D764C6466A42EDC0E4444BC425DEE036B9A2522C0C42` |
| `paper_phase2/figures/candidate_diagnostics/negative_blank_vs_pool.png` | `7AF0B94D124F6A1750A3FA4F0C3E0C0EEC1051C7A8C50266EED571F72DA99A24` |

Gate C query-level result:

- validation query images: `30`;
- reference episodes: `90`;
- mean point precision: `0.971111`;
- mean random precision: `0.548889`;
- precision delta: `0.422222`;
- precision ratio: `1.769231`;
- mean component coverage: `0.418820`;
- required component coverage: `0.600000`;
- mean negative blank fraction: `0.135556`;
- coordinate validation: `PASS`;
- status: `FAIL`;
- decision: `REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`.

### Predeclared Candidate Redesign v1

The allowed redesign was pre-registered in `docs/glas_candidate_redesign_v1.md`.

| Artifact | SHA256 |
| --- | --- |
| `reports/phase2_glas/candidates_redesign_v1/candidate_pilot_summary.json` | `2028D52F3919919142648385FEF1DA1848D005938951A0F6E9434AA188872109` |
| `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json` | `7A33C513EBD96984D4720D282F712B2B025D711BADB4377E81EF5FFC5093F13C` |
| `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.md` | `757176C1C606D768B6A3D6B48DF7BD66D4BA188AC88A9990A58B101F2836494D` |
| `paper_phase2/figures/candidate_diagnostics_redesign_v1/precision_vs_pool.png` | `240AA32C8563E8583DE5EEC3608A6B02710BD5142F766AC81EBE3F7E7D362D4E` |
| `paper_phase2/figures/candidate_diagnostics_redesign_v1/coverage_vs_pool.png` | `FAF60FBAAF42E5D770F237BBBA02D376659C30F53019532E23FF8F044DAB0F04` |
| `paper_phase2/figures/candidate_diagnostics_redesign_v1/negative_blank_vs_pool.png` | `EFD93262CA62FFCED03D09593FBEF0014687F2AC65019180FAF771AAF9AB2447` |

Redesign v1 query-level result:

- positive layout: `local_peaks`;
- positive diagnostic pool sizes: `5,10,20`;
- validation query images: `30`;
- reference episodes: `90`;
- mean point precision: `0.971667`;
- mean random precision: `0.537778`;
- precision delta: `0.433889`;
- precision ratio: `1.806818`;
- mean component coverage: `0.630221`;
- required component coverage: `0.600000`;
- mean negative blank fraction: `0.135556`;
- coordinate validation: `PASS`;
- status: `PASS`;
- decision: `CONTINUE_TO_GATE_D`.

## Current Decision

## Gate D Segmenter Smoke

SAM-B Gate D smoke ran on 6 selected strict-validation cases. It produced target-related privileged masks but failed the per-case privileged-control margin rule.

| Artifact | SHA256 |
| --- | --- |
| `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json` | `069DD4097C4BF3F620B0E5CC9A5B698718EA4F04586BA5BB94CA2B1B966F150C` |
| `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_case_metrics.csv` | `E20C29A13455022EE36EEC497099365FA2A6D8CFE3AB07619D32B69A0F0325EF` |
| `docs/glas_gate_d_segmenter_alternative_v1.md` | `DDB7D657EBDBC04621021B12096EAB79A6AAD3E7FC6D829CA88D34A0262990D3` |
| `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json` | `8F76D4BC3FC5D56B8D17428D55FB8805103BBE869DE24A67B7944D75D8FA40AB` |
| `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_case_metrics.csv` | `2209AF0786DC961E97DF3F44DBC0CE134AAC49E91C236D2D84DA2A2BACEDE646` |

Gate D result:

- SAM-B cases: `6`;
- SAM-B mean privileged best Dice: `0.733908`;
- SAM-B mean control best Dice: `0.638274`;
- SAM-B status: `FAIL`;
- single allowed alternative: `SAM-L`;
- SAM-L cases: `6`;
- SAM-L mean privileged best Dice: `0.762905`;
- SAM-L mean control best Dice: `0.412671`;
- SAM-L status: `FAIL`;
- shared issue: privileged prompts did not clearly exceed random/mismatched controls on at least one selected case.

## Current Decision

The current Phase 2 state is `STOP_GLAS_SEGMENTER_FAILURE`.

The frozen candidate generator had strong positive precision enrichment but insufficient component coverage on GlaS strict validation. The single predeclared `local_peaks` redesign passed Gate C. Gate D then failed for SAM-B, and the only predeclared alternative SAM-L also failed. Action-bank execution, oracle-gap analysis, utility generation, policy training, and formal test are stopped/prohibited for this Phase 2 branch. GlaS testA/testB remain locked and unused.

## Phase 2D Gate D Confirmation Preregistration

Phase 2D is a preregistered confirmation of whether the original 6-case all-case Gate D criterion is too strong for a segmenter execution/viability gate. It does not erase the original Gate D result. The original SAM-B/SAM-L 6-case reports remain preserved as:

`EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`

Phase 2D is allowed to use only the 30 strict-validation GlaS query images from official train, the frozen utility/reference episodes, the frozen redesigned candidate outputs, SAM-B, and SAM-L. GlaS testA/testB remain locked and must not be read.

Preregistration artifacts created before any new SAM confirmation inference:

| Artifact | SHA256 |
| --- | --- |
| `docs/glas_gate_d_confirmation_protocol_v1.md` | `4D5C95F31C756147F066C7C9506B50292636293E73D78B47BE24FB60031FFECA` |
| `docs/glas_gate_d_amendment_rationale.md` | `CB8252DDD40CC53BCF4B559971BC46C348BFC913796596628948279304495542` |
| `configs/segmenter/glas_gate_d_confirmation_v1.yaml` | `3D0D7074394D0CAACF61B93F11B48A0B00DAD5D73F26B5E23746C4F21EA4A145` |
| `reports/phase2_glas/gate_d_confirmation/protocol_snapshot.json` | `C5EFA51FBFF98E46B4A14F1313FB7C60A935C1125630848870FC94F0169882B6` |
| `reports/phase2_glas/gate_d_confirmation/input_hashes.json` | `9FED471C54AF43630BBEC59032134A655B99C9AF442885D3421FEF0E39C697E9` |
| `reports/phase2_glas/gate_d_confirmation/compute_plan.json` | `1FEE1C526FF70361EB5FDA59F7191780B0AFB7DC5CB279456C806586232DF014` |
| `src/adapromptset/scripts/run_glas_gate_d_confirmation.py` | `C2520D144419495F5CCD9C621A60D59685156173B7AC0CF7882B284F697F264C` |
| `scripts/run_glas_gate_d_confirmation.py` | `5FC8FB83373417E66E50DE4E37C231EDE4559B6DF87707663AD2136A65B68753` |
| `tests/test_run_glas_gate_d_confirmation.py` | `3C7C91EE36959AC4B825250FBF554965A77BE6B8B8635AFEE95B8E86CFD66898` |

Phase 2D primary protocol:

- primary family: `P3_component_grouped_centroids`;
- primary matched control: `mean(C1_random_coordinate, C2_mismatched_reference)`;
- C2 mapping: `first_frozen_episode_support_mask`;
- statistical unit: query image;
- query count: `30`;
- pass thresholds: P3 mean paired Dice margin `>= 0.05`, win rate `>= 0.70`, catastrophic failure rate `<= 0.10`, positive bootstrap point estimate, coordinate/cost audits passing.

Phase 2D confirmation result after preregistration commit `f5c2de3`:

| Metric | SAM-B | SAM-L |
| --- | ---: | ---: |
| P3 mean paired Dice margin | `-0.157937` | `-0.096909` |
| P3 win rate | `0.433333` | `0.466667` |
| P3 catastrophic failure rate | `0.666667` | `0.400000` |
| Coordinate audit | `PASS` | `PASS` |
| Cost audit | `PASS` | `PASS` |
| Status | `FAIL` | `FAIL` |

Final Phase 2D state:

`STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

GlaS testA/testB remain locked and unused. Gate E/F/G/H/I remain closed/prohibited.
