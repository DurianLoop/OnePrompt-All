# Dataset Audit：MonuSeg / IDRID 当前规模确认

日期：2026-07-09

## 0. 2026-07-09 重新审计后的直接结论

已经生成两个审计 CSV：

```text
runs/research_direction/dataset_audit/all_monu_valid_pairs.csv
runs/research_direction/dataset_audit/all_idrid_valid_pairs.csv
```

当前本地可配对规模：

| 数据域 | 本地可配对 image-mask 数 | 拆分 |
| --- | ---: | --- |
| MonuSeg | 88 | `kmms/images`: 82；`kmms_test/kmms_test/images`: 6 |
| IDRID | 81 | training: 54；test: 27 |

当前已经跑过的实验规模仍然是：

| 数据域 | 当前实验 pair 数 |
| --- | ---: |
| MonuSeg | 20 |
| IDRID | 30 |
| Total | 50 |

因此结论必须写得很清楚：

```text
当前已有实验不是官方完整数据集实验；
本地现在能审计到 MonuSeg 88 + IDRID 81 个可配对样本；
MonuSeg 的 88 个样本混合了 png patch 和 tif/original-like 文件，仍需和官方 MoNuSeg 文件列表核对后，才能声称 official full dataset。
```

## 1. 当前实验实际使用规模

当前 dense prompt / matcher medical 实验使用：

```text
runs/research_direction/matcher_medical_expanded_50/medical_matcher_pilot_manifest.csv
```

manifest 规模：

| Dataset | support-query pairs |
| --- | ---: |
| IDRID | 30 |
| MonuSeg | 20 |
| Total | 50 |

结论：

```text
当前实验不是官方完整数据集上的完整实验，而是本地构造的 pilot/expanded manifest。
```

## 2. 当前 MonuSeg 使用方式

当前 MonuSeg 20 个 query case 都来自：

```text
code/datasets/MonuSeg/kmms/images/*.png
code/datasets/MonuSeg/kmms/masks/*.png
```

并且 support 固定为：

```text
support_case_id = MonuSeg_01_1
```

这说明当前实验验证的是：

```text
single support mask -> multiple local MonuSeg query patches
```

不是完整 official train/test protocol。

## 3. 本地 MonuSeg 文件规模

当前本地目录统计：

| Local folder | images | masks | 备注 |
| --- | ---: | ---: | --- |
| `code/datasets/MonuSeg/kmms/images` | 50 png + 32 tif | 82 png masks | 原图和 patch/派生图混合 |
| `code/datasets/MonuSeg/kmms_training/kmms_training/images` | 24 tif | 24 png masks | 本地 training 子集 |
| `code/datasets/MonuSeg/kmms_test/kmms_test/images` | 50 png + 8 tif | 58 png masks | 本地 test/patch 子集 |

注意：

```text
当前本地文件夹可能包含 official MoNuSeg 原图、训练/测试子集和 patch 化派生样本。
在完成文件名/来源核对前，不能宣称它就是官方完整 MoNuSeg 数据集。
```

## 4. 下一步需要生成的审计产物

建议生成：

```text
runs/research_direction/dataset_audit/all_monu_valid_pairs.csv
runs/research_direction/dataset_audit/all_idrid_valid_pairs.csv
```

字段：

| 字段 | 含义 |
| --- | --- |
| dataset | MonuSeg / IDRID |
| sample_id | 样本 ID |
| image_path | 图像路径 |
| mask_path | mask 路径 |
| source_folder | 来源文件夹 |
| extension | tif / png / jpg |
| width | 图像宽 |
| height | 图像高 |
| split_guess | train / test / local / unknown |
| is_patch | 是否 patch/派生图 |
| mask_area_ratio | mask 面积比例 |

只有完成这个审计后，才能决定：

```text
1. 使用 official full dataset；
2. 使用 local full dataset；
3. 使用 curated subset；
4. 使用 patch-level expanded dataset。
```
