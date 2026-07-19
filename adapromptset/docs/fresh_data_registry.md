# Fresh Data Registry

Date: 2026-07-19

| Dataset | Local path | Current role | Freshness for Phase 4+ | Formal-test rule |
| --- | --- | --- | --- | --- |
| MoNuSAC | `D:\SAM\code\datasets\MoNuSAC\processed` | historical/dev boundary | Not fresh for new adaptive-policy claims; read-only or development boundary only | No new formal claim from existing observed MVP splits. |
| GlaS 55/30 official-train protocol | `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` | prior Phase 2/3 development and strict validation | Not fresh for selecting new GlaS method; may be read-only audit evidence | testA/testB remain locked until final method freeze. |
| GlaS testA/testB | same source tree; locked manifests under Phase 2 | untouched formal pathology test and morphology-shift test | Fresh and locked | Access only after `docs/test_freeze_glas.md`, frozen config, hashes, and final method. |
| DDR lesion segmentation | `D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation` | candidate fresh domain | train/valid may support new development; test remains fresh | Formal DDR test requires `docs/test_freeze_ddr.md` and four-intent protocol freeze. |
| MoNuSeg | `D:\SAM\code\datasets\MonuSeg\official` | secondary pathology cross-check | Partly historical; official split identity must be re-audited before formal use | Not sole formal test. |
| IDRID | `D:\SAM\code\datasets\IDRID` | read-only boundary | Not fresh unless an unobserved subset can be proven | No tuning; no new formal claim by default. |
| CPM-17 | `D:\SAM\external\research_data\raw\CPM17` | deferred incomplete data | Not ready | Requires user approval for repair/download and license/source confirmation. |

No network download or new model acquisition is allowed by this registry.

## 2026-07-19 Route B DDR Development Manifest

- Protocol: `docs/route_b_ddr_dev_protocol_v1.md`
- Report: `reports/route_b_ddr_manifest/ddr_route_b_manifest_report.json`
- Status: `PASS_DDR_ROUTE_B_DEV_MANIFEST`
- Generated splits: `train`, `valid`
- DDR test status: `LOCKED_UNUSED`; no test files were read.
- Positive-query counts: train 964, valid 401.
