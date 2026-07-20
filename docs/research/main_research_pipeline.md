# Main Research Pipeline：MonuSeg Reference-Guided Prompting

日期：2026-07-09

## 0. 当前共识

现在不要急着加学习模块。更合理的顺序是：

```text
先筛 segmenter，再筛 feature backbone，再做 prompt policy，最后再训练 selector / policy。
```

原因很简单：

1. 如果 segmenter 本身不适合 dense point prompt，后面训练 selector 也只是补救。
2. 如果 feature backbone 找点不准，SAM 系列模型再强也没有用。
3. DINOv2 只是当前 matcher 的一个选择，不应该被默认钉死。
4. 论文里最终可以展示最好的几个模型，但筛选过程必须可复现，不能变成测试集挑结果。

这份文档是新的主路线，不依赖旧文档。后续优先看它。

## 1. 研究问题重新定义

我们的任务不是证明某一个 SAM 版本一定最好，而是研究：

```text
给定一张带人工 mask 的 reference 图，如何把 reference mask 中的目标信息转化为 query 图上的有效 prompt，从而提升 SAM-family / MedSAM-family 模型在 MonuSeg all-nuclei segmentation 上的表现。
```

完整链路：

```text
support image + support mask
        ↓
feature backbone 提取 patch features
        ↓
foreground/background matching 得到 query score map
        ↓
point / box / mask prompt policy
        ↓
SAM / SAM2 / SAM3 / MedSAM segmenter
        ↓
candidate masks
        ↓
评价或 selector 选择最终 mask
```

当前阶段先做四件事：

```text
0. 当前数据集到底有多大，是否是官方完整集？
1. 哪些 segmenter 有潜力？
2. 哪些 feature backbone 找点更准？
3. 哪些 prompt policy 在 MonuSeg 上更合理？
```

暂时不优先训练。

## 2. 数据集规模确认

### 2.1 当前实验实际使用的数据

当前主实验使用的是：

```text
runs/research_direction/matcher_medical_expanded_50/medical_matcher_pilot_manifest.csv
```

这个 manifest 一共有：

| Dataset | support-query pairs |
| --- | ---: |
| IDRID | 30 |
| MonuSeg | 20 |
| Total | 50 |

其中 MonuSeg 当前只用了 20 个 query case，并且 support 固定为：

```text
support_case_id = MonuSeg_01_1
```

query 来自：

```text
code/datasets/MonuSeg/kmms/images/*.png
code/datasets/MonuSeg/kmms/masks/*.png
```

因此必须明确：

```text
当前已有 SAM/SAM3 结果不是官方完整 MonuSeg 数据集上的完整实验，
而是本地构造的 20-case MonuSeg pilot/expanded manifest。
```

### 2.2 本地 MonuSeg 文件规模

本地 `code/datasets/MonuSeg` 目前包含多个目录：

| Local folder | images | masks | 说明 |
| --- | ---: | ---: | --- |
| `kmms/images` | 50 png + 32 tif | 82 png masks | 当前 manifest 主要从这里取前 20 个 png query |
| `kmms_training/kmms_training/images` | 24 tif | 24 png masks | 本地训练子集 |
| `kmms_test/kmms_test/images` | 50 png + 8 tif | 58 png masks | 本地测试/patch 子集 |

这个本地结构看起来包含原图和 patch/派生图混合版本。仅凭当前目录结构，不能直接宣称它就是官方完整 MoNuSeg 原始训练/测试集。

下一步必须做一个数据审计：

```text
1. 建立 all_monu_valid_pairs.csv，枚举所有 image-mask 可配对样本。
2. 标记每个样本来自 original tif 还是 patch png。
3. 标记 train/test/local split。
4. 和官方 MoNuSeg 文件列表核对。
5. 论文中明确报告：使用 official full set / local patch set / curated subset。
```

### 2.3 大规模优先原则

你希望一开始用较大规模，然后后续再缩小。这里可以这样执行：

```text
第一步不再只用 5-case screening。
先构建本地所有可配对 MonuSeg 样本的 full-local manifest。
如果 full-local case 数在 100 左右以内，model/backbone screening 直接用 full-local。
如果超过 100，再按 organ/source 分层抽样，但必须保留完整 manifest 作为附录。
```

也就是说，新的顺序变成：

```text
Dataset audit -> full-local MonuSeg manifest -> broad model/backbone screening -> validation 收缩 -> full experiment
```

## 3. 已有实验事实

### 3.1 SAM ViT-B

已有完整实验：

| 项目 | 数值 |
| --- | ---: |
| cases | 20 |
| candidates | 340 |
| best K | 20 |
| best lambda | 0.25 |
| best layout | local maxima |
| best fixed-strategy Dice | 0.3861 |
| candidate oracle Dice | 0.4464 |

解释：

```text
在 SAM ViT-B 上，reference-guided dense matched points 是有潜力的。
```

### 3.2 SAM3

已有完整重测：

| 项目 | 数值 |
| --- | ---: |
| cases | 20 |
| candidates | 1460 |
| best K | 10 |
| best lambda | 0.75 |
| best layout | DBSCAN medoid |
| best fixed-strategy Dice | 0.0742 |
| candidate oracle Dice | 0.1287 |

解释：

```text
同样的 dense positive point policy 不能直接迁移到 SAM3 interactive point prompt。
```

这不说明 SAM3 弱，只说明：

```text
SAM3 的优势场景可能更接近 concept/text/exemplar segmentation，而不是把很多 positive points 一次性合成为 all-nuclei mask。
```

## 4. 总体执行顺序

建议按照下面顺序推进。

```text
Phase 0: Dataset audit and full-local manifest
Phase 1: Large-scale Segmenter zoo screening
Phase 2: Large-scale Feature backbone zoo screening
Phase 3: Prompt policy controlled experiments
Phase 4: Baseline / auxiliary / ablation formal experiments
Phase 5: Optional learning modules
Phase 6: Paper tables and report
```

最重要的是：Phase 0、Phase 1 和 Phase 2 先完成，暂时不要急着训练。

## 5. Phase 1：Segmenter Model Zoo

### 4.1 本地已有模型

| Model key | Family | Local checkpoint | Prompt type | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `sam_vit_b` | SAM | `assets/checkpoints/sam_vit_b.pth` | point / box | 已跑 | 当前最重要 baseline |
| `sam_vit_h` | SAM | `assets/checkpoints/sam_vit_h_4b8939.pth` | point / box | 待跑 | SAM 更强版本，本地已有 |
| `sam2p1_hiera_tiny` | SAM2.1 | `assets/checkpoints/sam2.1_hiera_tiny.pt` | point / box | 待跑 | 轻量 SAM2，不代表最强 SAM2 |
| `sam3_modelscope` | SAM3 | `assets/checkpoints/sam3_modelscope/sam3.pt` | point / text / box | 已跑 point | 本地 ModelScope SAM3 权重 |
| `medsam_vit_b` | MedSAM | `assets/checkpoints/medsam_vit_b.pth` | box | 待跑 | MedSAM 更适合 box prompt |
| `lite_medsam` | Lite-MedSAM | `assets/checkpoints/medsam_download/lite_medsam.pth` | box/medical prompt | 待筛 | 轻量医学 baseline |
| `medical_adapter_microscope` | Medical adapter | `assets/checkpoints/medical_adapter_zoo/White Blood Cell_MicroScope_sam_1024.pth` | 待确认 | 待筛 | 可能和 MonuSeg 更接近 |
| `medical_adapter_fundus` | Medical adapter | `assets/checkpoints/medical_adapter_zoo/OpticDisc_Fundus_SAM_1024.pth` | 待确认 | 低优先级 | 更适合眼底，不是 MonuSeg |

### 5.2 必须补充下载并测试

这些模型都进入第一轮 broad screening。后续是否进入论文主表，由 validation screening 决定。

| Model | 目的 | 优先级 | 备注 |
| --- | --- | --- | --- |
| SAM ViT-L | 补齐 SAM B/L/H 三档 | P0 | 避免只比较 B 和 H |
| SAM2.1 Hiera Small | 比 tiny 更强但较省资源 | P0 | 必测 |
| SAM2.1 Hiera Base+ | 中等强度 SAM2 baseline | P0 | 必测 |
| SAM2.1 Hiera Large | SAM2 强 baseline | P0 | 必测，即使慢也要先测一轮 |
| SAM3 official HF copy | 和 ModelScope 镜像交叉确认 | P0 | 必测；可能需要 HF 授权 |
| BioMedCLIP | 医学 VLM，可作为 feature backbone，不是 segmenter | P0 | 放入 feature zoo |
| UNI / CONCH pathology models | 病理 foundation model | P0 | 放入 feature zoo；可能需要授权 |
| Medical-SAM-Adapter | 医学适配 SAM | P0 | 必测 |
| SAM-Med2D | 医学 2D SAM baseline | P0 | 必测 |

### 5.3 Model zoo manifest

建议新建：

```text
docs/research/model_zoo_manifest.csv
```

字段：

| 字段 | 含义 |
| --- | --- |
| model_key | 例如 `sam_vit_l` |
| family | SAM / SAM2 / SAM3 / MedSAM / adapter |
| checkpoint | 本地权重路径 |
| config | SAM2/SAM3 需要的配置 |
| prompt_types | point / box / text / mask / exemplar |
| source | official / mirror / third-party |
| paper | 对应论文或 repo |
| status | missing / downloaded / runnable / failed |
| notes | 是否最强配置、是否镜像、是否授权 |

### 5.4 Segmenter broad screening protocol

第一轮不再只做 5-case 小筛。新的原则：

```text
优先使用 full-local MonuSeg manifest。
如果样本数过大，再做分层抽样，但抽样规则必须写入 manifest。
```

为了控制计算量，第一轮 broad screening 使用较小 grid：

```text
cases = full-local MonuSeg valid pairs if feasible
K = {1, 5, 10, 20, 50}
lambda = {0, 0.25, 0.75, 1.5}
layout = {top-k, local maxima}
```

每个模型输出：

| Model | Prompt type | mean Dice | oracle Dice | runtime/case | memory | 是否进入 full |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SAM ViT-B | point | 已有/待筛 | 已有/待筛 | 待填 | 待填 | 待定 |
| SAM ViT-L | point | 待填 | 待填 | 待填 | 待填 | 待定 |
| SAM ViT-H | point | 待填 | 待填 | 待填 | 待填 | 待定 |
| SAM2.1 Tiny | point | 待填 | 待填 | 待填 | 待填 | 待定 |
| SAM2.1 Small | point | 待填 | 待填 | 待填 | 待填 | 待定 |
| SAM3 | point | 已有/待筛 | 已有/待筛 | 待填 | 待填 | 待定 |
| MedSAM | box | 待填 | 待填 | 待填 | 待填 | 待定 |

筛选规则：

```text
每个 family 选 1-2 个代表模型进入 full experiment。
选择依据必须来自 validation screening，而不是 test set。
主文放代表模型，附录保留 model zoo 全表。
```

这样可以“挑出效果好的模型”，但不是 cherry-pick。

## 6. Phase 2：Feature Backbone Zoo

### 5.1 为什么 DINOv2 也要重新筛

目前 matcher 用的是 DINOv2 ViT-L feature。

但 DINOv2 不是唯一合理选择。MonuSeg 是病理/显微图像，可能存在三类更合适的 feature：

1. 更小或更大的 DINOv2 版本。
2. biomedical vision-language feature，例如 BioMedCLIP。
3. pathology foundation model，例如 UNI、CONCH、Virchow 等。

所以 feature backbone 必须单独筛选。

### 6.2 五个必测 feature backbone

第一轮至少测五个 backbone：

| Backbone key | 类型 | 可能优势 | 风险/成本 | 优先级 |
| --- | --- | --- | --- | --- |
| `dinov2_vitl14` | general SSL | 当前已用，强 baseline | 非医学域 | P0 |
| `dinov2_vitb14` | general SSL | 性能/速度均衡 | 非医学域 | P0 |
| `biomedclip` | biomedical VLM | 医学图文预训练，域更近 | patch localization 未必强 | P0 |
| `uni_pathology` | pathology FM | 病理图像域更近 | 权重/接口/patch 尺度需处理 | P0 |
| `conch_pathology` | pathology VLM | 病理视觉语言特征 | 可能授权/下载麻烦 | P0 |

可选扩展：

| Backbone key | 用途 | 优先级 |
| --- | --- | --- |
| `dinov2_vits14` | 轻量对照 | P1 |
| `dinov2_vitg14` | 大模型上限 | P1 |
| `virchow_pathology` | 病理大模型上限 | P2 |
| `retfound` | IDRID/眼底方向 | P2 |

### 6.3 Feature screening 不调用 SAM

第一轮不要把 feature screening 和 SAM 推理绑在一起，否则变量太多。

先只评价 feature/matching 本身：

```text
support image + support mask
        ↓
feature backbone
        ↓
query score map
        ↓
top-k matched points
        ↓
和 query GT 比较
```

指标：

| 指标 | 说明 |
| --- | --- |
| point precision@K | top-k 点有多少落在 GT mask 内 |
| component recall@K | top-k 点覆盖多少 GT nuclei connected components |
| score-map AP | score map 对 downsampled GT 的 average precision |
| score-map AUC | 前景 patch 是否排在背景 patch 前面 |
| fg-bg separation | GT 前景 patch 分数均值 - 背景 patch 分数均值 |
| top-k diversity | top-k 点是否都堆在一起 |

建议 K：

```text
K = {10, 20, 50, 100}
```

输出表：

| Backbone | precision@20 | component recall@50 | score AP | fg-bg separation | speed | 进入下一轮 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DINOv2-S | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |
| DINOv2-B | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |
| DINOv2-L | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |
| BioMedCLIP | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |
| UNI | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |

### 6.4 Feature + Segmenter 组合筛选

第二轮只选前 2-3 个 feature backbone，和前 2-3 个 segmenter 组合。

例如：

```text
features = {DINOv2-L, BioMedCLIP, UNI}
segmenters = {SAM ViT-B, SAM ViT-H, MedSAM/SAM2 best}
```

组合表：

| Feature | Segmenter | Prompt | Dice | Oracle Dice | candidates/case | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| DINOv2-L | SAM ViT-B | point | 已有 | 已有 | 已有 | 当前 baseline |
| DINOv2-L | SAM ViT-H | point | 待填 | 待填 | 待填 | 更强 SAM |
| BioMedCLIP | SAM ViT-B | point | 待填 | 待填 | 待填 | 医学 VLM feature |
| UNI | SAM ViT-B | point | 待填 | 待填 | 待填 | 病理 feature |
| best feature | MedSAM | box | 待填 | 待填 | 待填 | box policy |

## 7. Phase 3：Prompt Policy Controlled Experiments

这一步仍然不训练，只做受控变量实验。

### 6.1 K sensitivity

```text
K = {1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100}
```

目的：

```text
证明 MonuSeg 需要 task-specific point budget，而不是点越多越好。
```

### 6.2 lambda sensitivity

```text
lambda = {0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}
```

目的：

```text
证明 support mask 外的 background feature 是否能抑制 false positive。
```

### 6.3 layout sensitivity

```text
top-k
diverse top-k
local maxima
boundary mix
DBSCAN medoid
heatmap connected components
```

目的：

```text
证明点的位置和布局比“随便取 top-k”更重要。
```

## 8. Phase 4：论文实验结构

论文实验分四类。

### 7.1 主实验：Baseline 对比

主实验回答：

```text
我们的方法是否比合理 baseline 更好？
```

Baseline：

| Baseline | 说明 |
| --- | --- |
| center point | 固定单点 |
| random point | 随机点 |
| grid points | 固定网格点 |
| fg-only matcher | 只用 foreground prototype |
| fg-bg matcher | 使用 foreground-background contrast |
| GT box oracle | 使用 query GT box，只作上限 |
| random GT foreground point oracle | 使用 query GT 采点，只作上限 |
| nnU-Net | fully supervised upper bound |

主表：

| Method | Feature | Segmenter | Reference mask | Train? | candidates/case | Dice | IoU | Obj recall | Merge |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| center point | none | SAM | no | no | 1 | 待填 | 待填 | 待填 | 待填 |
| fg-only matcher | DINOv2-L | SAM | yes | no | 1 | 待填 | 待填 | 待填 | 待填 |
| fg-bg matcher | DINOv2-L | SAM | yes | no | 1 | 待填 | 待填 | 待填 | 待填 |
| best screened pipeline | best | best | yes | no/yes | 1-3 | 待填 | 待填 | 待填 | 待填 |
| nnU-Net | none | none | no | full | 0 | 待填 | 待填 | 待填 | 待填 |

### 7.2 辅助证明实验

辅助实验解释为什么有效：

1. feature backbone 是否真的找到 GT 前景点。
2. foreground-background matching 是否比 foreground-only 好。
3. dense points 是否适合 MonuSeg。
4. 不同 segmenter 对同一 prompt 的响应差异。

### 7.3 消融实验

消融拆掉自己的模块：

| Variant | Feature | bg proto | K policy | layout | selector | Dice |
| --- | --- | --- | --- | --- | --- | ---: |
| full | best | yes | best/learned | best/learned | yes | 待填 |
| w/o bg | best | no | same | same | same | 待填 |
| fixed K | best | yes | fixed | same | same | 待填 |
| fixed layout | best | yes | same | top-k | same | 待填 |
| no selector | best | yes | same | same | no | 待填 |

### 7.4 泛化/展示实验

展示不同模型家族：

```text
SAM / SAM2 / SAM3 / MedSAM / medical adapter
```

注意分两张表：

1. 同 prompt 对照：分析模型响应。
2. 各模型合理 prompt：展示各模型实际能力。

## 9. Phase 5：什么时候再加学习模块

只有当下面两件事完成后，再考虑学习：

```text
1. segmenter screening 得到 2-3 个高潜力模型。
2. feature screening 得到 2-3 个高潜力 backbone。
```

学习模块优先级：

1. proposal selector。
2. K/lambda/layout policy。
3. score re-weighting。

暂时不要微调 SAM/SAM2/SAM3 主干。

## 10. 训练/验证/测试原则

如果只做 20 cases：

```text
outer 5-fold: test
inner split: validation/model selection
```

如果扩到 50 cases：

```text
5-fold:
40 train/validation
10 test
```

关键原则：

```text
模型选择、backbone 选择、K/lambda 选择都只能在 validation 上做。
test set 只用于最终报告。
```

如果主文只展示几个最好的模型，必须说明：

```text
它们是 validation screening 选出来的。
完整 screening 结果放附录。
```

## 11. 近期可执行清单

### T0：数据集审计

新建：

```text
docs/research/dataset_audit_monu_idrid.md
runs/research_direction/dataset_audit/all_monu_valid_pairs.csv
runs/research_direction/dataset_audit/all_idrid_valid_pairs.csv
```

必须先回答：

```text
当前使用的是 official full dataset、local full dataset、还是 curated subset？
```

### T1：建立 model/backbone manifest

新建：

```text
docs/research/model_zoo_manifest.csv
docs/research/feature_backbone_zoo_manifest.csv
```

### T2：补下载模型

优先：

```text
SAM ViT-L
SAM2.1 Hiera Small
SAM2.1 Hiera Base+
SAM2.1 Hiera Large
```

再考虑：

```text
SAM3 official HF copy
BioMedCLIP
UNI / CONCH pathology models
Medical-SAM-Adapter / SAM-Med2D
```

### T3：segmenter broad screening

优先 full-local MonuSeg，先大规模测，再收缩。

输出：

```text
runs/research_direction/segmenter_zoo_screening/
docs/reports/segmenter_zoo_screening/index.html
```

### T4：feature broad screening

先不调用 SAM，只评价 top-k matched points 和 score map。

输出：

```text
runs/research_direction/feature_backbone_zoo_screening/
docs/reports/feature_backbone_zoo_screening/index.html
```

### T5：组合筛选

选择：

```text
top 2-3 feature backbones × top 2-3 segmenters
```

输出组合表。

### T6：正式 baseline/辅助/消融

根据筛选结果跑完整 20/50 cases。

## 12. 参考来源

这些来源用于确定 model/backbone zoo，不代表我们已经下载或完整复现了所有模型。

| 方向 | 来源 |
| --- | --- |
| SAM | Segment Anything, arXiv 2304.02643 |
| SAM2 | SAM 2: Segment Anything in Images and Videos, arXiv 2408.00714 |
| SAM3 | SAM 3: Segment Anything with Concepts, arXiv 2511.16719 / facebookresearch/sam3 |
| DINOv2 | DINOv2: Learning Robust Visual Features without Supervision, arXiv 2304.07193 |
| BioMedCLIP | BiomedCLIP, arXiv 2303.00915 |
| MedSAM / biomedical SAM2 | MedSAM / Biomedical SAM2 papers and repositories |

## 13. 2026-07-09 当前确认结果与执行状态

### 13.1 数据集规模结论

当前已经跑出来的 SAM / SAM3 结果，使用的不是官方完整数据集，而是本地构造的 50 条 support-query manifest：

```text
runs/research_direction/matcher_medical_expanded_50/medical_matcher_pilot_manifest.csv
```

其中：

| 数据域 | 当前实验 pair 数 | 说明 |
| --- | ---: | --- |
| MonuSeg | 20 | 当前主线 dense prompt / matcher 实验实际使用的 MonuSeg query case |
| IDRID | 30 | 医学 matcher 扩展 manifest 中的眼底 pair |
| 合计 | 50 | 不是 official full dataset protocol |

已经生成本地数据审计表：

```text
runs/research_direction/dataset_audit/all_monu_valid_pairs.csv
runs/research_direction/dataset_audit/all_idrid_valid_pairs.csv
```

审计结果：

| 数据域 | 本地可配对 image-mask 数 | 来源拆分 | 结论 |
| --- | ---: | --- | --- |
| MonuSeg | 88 | `kmms/images`: 82；`kmms_test/kmms_test/images`: 6 | 比当前 20-case 实验大，但混合 png patch 和 tif，不能直接等同官方完整协议 |
| IDRID | 81 | training: 54；test: 27 | 本地 fundus image 和 combined mask 配对完整 |

所以目前应区分三件事：

```text
已有实验规模：MonuSeg 20 + IDRID 30 = 50 pairs
本地可用规模：MonuSeg 88 + IDRID 81 = 169 image-mask pairs
官方完整数据集：尚未完成官方文件列表/协议核对，不能直接宣称已经完整使用
```

本地 MonuSeg 文件夹里确实有更多数据，但目前看起来混合了 original tif、png patch、training/test 子目录和派生样本。仅凭当前目录结构，不能直接宣称“已经使用官方完整 MoNuSeg”。下一步如果要写论文，必须在方法或附录里明确我们使用的是 official full set、local full set、还是 curated patch set。

### 13.2 当前路线调整

按照你的要求，路线调整为：

```text
先用尽可能大的本地可用规模做 broad screening
-> 再根据结果缩小模型、backbone 和 prompt policy
-> 最后再考虑是否加入 learning / selector
```

也就是说，现阶段先不把变量收得过窄，而是先把下面两类“底座能力”测清楚：

1. 哪些 SAM-family / MedSAM-family segmenter 对 MonuSeg dense nuclei prompt 有潜力。
2. 哪些 feature backbone 能更稳定地把 reference mask 信息迁移到 query 图上。

### 13.3 已下载完成的 P0 权重

2026-07-09 已补齐并核验文件大小：

| 模型/Backbone | 本地路径 | 文件大小 | 当前状态 |
| --- | --- | ---: | --- |
| SAM ViT-L | `assets/checkpoints/sam_vit_l_0b3195.pth` | 1,249,524,607 bytes | 已下载，待接入筛选 |
| SAM2.1 Hiera Small | `assets/checkpoints/sam2.1_hiera_small.pt` | 184,416,285 bytes | 已下载，待接入筛选 |
| SAM2.1 Hiera Base+ | `assets/checkpoints/sam2.1_hiera_base_plus.pt` | 323,606,802 bytes | 已下载，待接入筛选 |
| SAM2.1 Hiera Large | `assets/checkpoints/sam2.1_hiera_large.pt` | 898,083,611 bytes | 已下载，必须进入第一轮大模型筛选 |
| DINOv2 ViT-B/14 | `assets/checkpoints/dinov2_vitb14_pretrain.pth` | 346,378,731 bytes | 已下载，待做 feature screening |

此前本地已经存在：

| 模型/Backbone | 本地路径 | 当前状态 |
| --- | --- | --- |
| SAM ViT-B | `assets/checkpoints/sam_vit_b.pth` | 已跑过 MonuSeg dense prompt baseline |
| SAM ViT-H | `assets/checkpoints/sam_vit_h_4b8939.pth` | 已下载，待筛选 |
| SAM2.1 Hiera Tiny | `assets/checkpoints/sam2.1_hiera_tiny.pt` | 已下载，待筛选 |
| SAM3 ModelScope copy | `assets/checkpoints/sam3_modelscope/sam3.pt` | 已跑过 SAM3 point-grid retry |
| DINOv2 ViT-L/14 | `assets/checkpoints/dinov2_vitl14_pretrain.pth` | 当前 matcher 主 backbone |
| MedSAM ViT-B | `assets/checkpoints/medsam_vit_b.pth` | 已下载，待筛选 |
| Medical-SAM-Adapter microscope adapter | `assets/checkpoints/medical_adapter_zoo/White Blood Cell_MicroScope_sam_1024.pth` | 已下载，和 MonuSeg 更相关 |
| BioMedCLIP | `assets/checkpoints/biomedclip/open_clip_pytorch_model.bin` | 已下载，待做 feature backbone 适配 |
| SAM-Med2D | `assets/checkpoints/sam_med2d_b.pth` | 已下载，待做接口适配和筛选 |

### 13.4 需要授权或特殊下载的模型

| 模型/Backbone | 当前判断 | 下一步 |
| --- | --- | --- |
| SAM3 official HF copy `facebook/sam3` | 已尝试，HF 返回 Access denied/requires approval | 有 HF token 且账号通过授权后下载到 `assets/checkpoints/sam3_hf`，再和 ModelScope copy 对照 |
| BioMedCLIP | 已下载到 `assets/checkpoints/biomedclip` | 下一步安装/适配 `open_clip`，设计 patch feature 提取 |
| UNI pathology | 已尝试，HF 返回 Access denied/requires approval | 需要申请/登录后下载 |
| CONCH pathology | 已尝试，HF 返回 Access denied/requires approval | 需要申请/登录后下载 |
| SAM-Med2D | 已从 README 的 Google Drive 权重链接下载 | 已保存为 `assets/checkpoints/sam_med2d_b.pth`，下一步做接口适配 |

### 13.5 下一步立即执行顺序

1. 已完成数据审计 CSV：MonuSeg 本地可配对 88，IDRID 本地可配对 81。
2. 下载 BioMedCLIP；UNI / CONCH / SAM3 HF 先尝试，若 gated 就记录为“需要授权”。
3. 接入 SAM2.1 Hiera Large / Base+ / Small 到现有 point prompt screening runner。
4. 在 full-local MonuSeg 上先跑 segmenter zoo broad screening。
5. 单独跑 feature backbone zoo screening，不调用 SAM，只看 top-k point 是否落在 GT nuclei 上。
