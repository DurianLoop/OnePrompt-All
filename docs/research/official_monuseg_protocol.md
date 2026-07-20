# 官方完整 MoNuSeg protocol 补全说明

日期：2026-07-09

## 0. 当前补全结果

官方 MoNuSeg 数据已经下载并整理到本地。原始下载文件保留，XML annotation 也已经转换成 binary all-nuclei mask，后续可以直接用于 reference-guided matching、SAM-family prompt segmentation、baseline 对比和 broad screening。

### 0.1 官方来源

官方页面：

```text
https://monuseg.grand-challenge.org/Data/
```

本次下载的官方文件：

| 文件 | 官方说明 | 本地位置 |
| --- | --- | --- |
| training data | MoNuSeg 2018 training data | `code/datasets/MonuSeg/official/downloads/monuseg_2018_training_data.zip` |
| testing data | MoNuSeg 2018 testing data | `code/datasets/MonuSeg/official/downloads/monuseg_2018_testing_data.zip` |
| training organ information | training image organ metadata | `code/datasets/MonuSeg/official/downloads/training_data_organ_information` |
| MATLAB mask code | XML annotation 转 binary/n-ary mask 的官方参考代码 | `code/datasets/MonuSeg/official/downloads/HE_to_binary_nary_masks.m` |

### 0.2 当前本地目录

```text
code/datasets/MonuSeg/official/
├── downloads/          # 官方原始下载文件
├── raw/                # 解压后的官方原始 tif+xml
├── train/
│   ├── images/         # 复制出的训练 tif
│   └── masks/          # 由 XML 转出的 binary nuclei mask
└── test/
    ├── images/         # 复制出的测试 tif
    └── masks/          # 由 XML 转出的 binary nuclei mask
```

可复现脚本：

```text
scripts/prepare_official_monuseg.py
```

统一 manifest：

```text
runs/research_direction/dataset_audit/monuseg_official_manifest.csv
```

manifest 字段包括：

```text
dataset, split, sample_id, image_path, annotation_path, mask_path,
width, height, nuclei_region_count, mask_area_ratio, source_zip, status
```

### 0.3 已验证数量

本次下载和转换后的验证结果：

| split | image | XML annotation | binary mask | manifest rows |
| --- | ---: | ---: | ---: | ---: |
| train | 37 | 37 | 37 | 37 |
| test | 14 | 14 | 14 | 14 |
| total | 51 | 51 | 51 | 51 |

mask 验证：

```text
empty masks = 0
nuclei_region_count min = 249
nuclei_region_count max = 1862
```

### 0.4 关于 30/14 和 37/14 的差异

官方网页文字描述里常见的 challenge protocol 是：

```text
30 training images + 14 test images = 44 images
```

但是当前从官方页面链接下载的 training zip 里，忽略 `__MACOSX` 和 `._*` 这类 macOS 隐藏资源文件后，实际存在：

```text
37 real training tif+xml pairs
14 real testing tif+xml pairs
```

因此当前最严谨的说法是：

```text
official Grand Challenge download package observed locally = 37 train + 14 test
```

我没有把多出来的 7 个 training pair 删除。后续有两种合理用法：

1. broad screening 阶段：使用当前官方包完整的 37/14，提高筛选覆盖面。
2. 严格 challenge protocol 阶段：再根据官方论文/organ information 固定 30-image train subset，并明确报告使用的是 30/14。

这个差异后续写报告或论文时必须主动说明，否则容易被认为数据规模写错。

## 1. 为什么要补官方完整 protocol

当前我们已经跑过的 MonuSeg 实验主要是：

```text
20 个 MonuSeg query case
support 固定为 MonuSeg_01_1
输入主要来自 code/datasets/MonuSeg/kmms/images/*.png
```

这只能证明 reference-guided prompting 在一个本地小规模 patch 子集上有潜力，不能支撑“训练模型”或者“论文主表”的结论。

如果后续要做训练、模型选择、baseline 对比和论文实验，必须把 MoNuSeg 数据划分分清楚：

```text
official full protocol
local original-like tif protocol
local patch-expanded protocol
```

三者不能混在一起，否则会出现两个问题：

1. 训练/测试泄漏：同一张原始病理图切出来的 patch 不能同时出现在 train 和 test。
2. 规模误报：不能把 patch 数量当作 official case 数量。

## 2. 官方 MoNuSeg challenge 常用 protocol

MoNuSeg 是 Multi-Organ Nuclei Segmentation 数据集，核心任务是 H&E 病理图像中的细胞核分割。

常用官方 challenge protocol 应按原始图像级别组织，而不是按 patch 组织：

| 项目 | protocol |
| --- | --- |
| 数据类型 | H&E stained histopathology images |
| 图像级别 | 原始大图 crop，常见为约 1000 x 1000 像素 |
| 标注 | 每个细胞核的人工实例标注，可转换为 binary all-nuclei mask |
| 常用官方规模 | training 30 images，test 14 images，总计 44 images |
| 数据来源 | TCGA 多器官病理图像 |
| 训练原则 | 只能用 training images 训练/调参 |
| 测试原则 | test images 只用于最终报告，不能参与模型选择 |

我们当前项目的主任务是 all-nuclei segmentation，所以可以先把 instance annotation 合并为 binary mask：

```text
instance masks -> binary nuclei mask
```

如果后续论文要报告 instance-level 指标，再从 binary pipeline 扩展到 instance split / watershed / connected component 评估。

## 3. 当前本地 MonuSeg 数据不是完整官方 protocol

当前本地审计结果：

```text
runs/research_direction/dataset_audit/all_monu_valid_pairs.csv
```

本地可配对 MonuSeg image-mask pair：

| 类型 | 数量 | 说明 |
| --- | ---: | --- |
| png patch pair | 50 | 256 x 256 patch，适合快速筛选，但不能代表 official image-level protocol |
| original-like tif pair in `kmms` | 32 | TCGA tif 原图级 pair，比 patch 更接近 official protocol |
| `kmms_training` tif pair | 24 | 本地已有 train-like split |
| `kmms_test` tif pair | 8 | 本地已有 test-like split |

因此当前本地更准确地说是：

```text
local KMMS split = 24 train + 8 test = 32 original-like tif images
local patch set = 50 png patches
```

它不是完整 official challenge protocol：

```text
official challenge target = 30 train + 14 test = 44 images
current local original-like = 24 train + 8 test = 32 images
gap = 缺少或尚未核对 12 张 official image-level samples
```

## 4. 训练时应该怎么用这些数据

### 4.1 论文主实验优先级

优先级应该是：

```text
P0: official full MoNuSeg 30/14 protocol
P1: local original-like 24/8 protocol
P2: local patch-expanded protocol
```

如果 official full protocol 补齐失败，论文中必须诚实写成 local KMMS protocol，不能写 official full MoNuSeg。

### 4.2 patch 可以用，但不能污染 test

如果要通过切 patch 扩大训练规模，必须以原始 image-level split 为边界：

```text
train images -> train patches
val images -> val patches
test images -> test patches
```

绝对不能：

```text
同一张 TCGA tif 切出来的一部分 patch 进 train，另一部分 patch 进 test
```

这样会让模型看到同一张病理图的颜色、纹理和组织结构，测试结果会虚高。

### 4.3 推荐训练/验证/测试划分

如果补齐 official 44 images：

| split | 用途 | 数量 |
| --- | --- | ---: |
| train | 训练 adapter / prompt policy / selector | 30 official train images 内部再划分 |
| val | 选择 K、lambda、backbone、segmenter | 从 train 中切出，按 image-level |
| test | 最终主表 | 14 official test images |

如果暂时只有本地 24/8：

| split | 用途 | 数量 |
| --- | --- | ---: |
| train | 训练/筛选 | 24 local train images |
| val | 从 24 train 内部按 image-level 切分 |
| test | 最终 local test | 8 local test images |

patch 增广只在 split 内部进行。

## 5. 接下来要补齐的产物

### 5.1 official 文件列表核对

需要生成：

```text
runs/research_direction/dataset_audit/monuseg_official_file_audit.csv
```

字段：

| 字段 | 含义 |
| --- | --- |
| sample_id | TCGA image id |
| expected_split | official train / official test |
| local_found | 本地是否存在 |
| image_path | 本地图像路径 |
| mask_path | 本地 mask 路径 |
| width / height | 图像尺寸 |
| mask_area_ratio | mask 面积比例 |
| status | ok / missing_image / missing_mask / duplicate / patch_only |

### 5.2 official 数据补齐

需要做两件事：

1. 找到 official MoNuSeg train/test 文件包。
2. 下载或手动放到：

```text
code/datasets/MonuSeg/official/train/images
code/datasets/MonuSeg/official/train/masks
code/datasets/MonuSeg/official/test/images
code/datasets/MonuSeg/official/test/masks
```

然后再生成统一 manifest：

```text
runs/research_direction/dataset_audit/monuseg_official_44_manifest.csv
```

## 6. 当前结论

现在不能说我们已经使用了 official full MoNuSeg。

当前能说的是：

```text
我们已经完成本地 MonuSeg 审计。
当前已有实验只覆盖 20 个 MonuSeg patch/query case。
本地存在 32 个 original-like TCGA tif pair 和 50 个 png patch pair。
官方完整 protocol 目标应补到 30 train + 14 test = 44 image-level samples。
训练模型时必须按 image-level split，再在 split 内部切 patch 扩大训练量。
```

这一步补齐后，后续训练模型才有更稳的规模基础。

## 7. 参考入口

- MoNuSeg challenge data page: https://monuseg.grand-challenge.org/Data/
- MoNuSeg / generalized nuclear segmentation paper entry: https://ieeexplore.ieee.org/document/7872382
