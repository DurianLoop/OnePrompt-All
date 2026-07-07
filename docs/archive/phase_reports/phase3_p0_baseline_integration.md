# Phase3 P0 Baseline 接入审计

本报告记录 P0 强基线从“口头计划”推进到“本地可审计状态”的结果。

- 已有代码仓库：4 / 5
- 已有代码和权重：3 / 5

| Baseline | 状态 | 仓库 | 权重 | 下一步 |
| --- | --- | ---: | ---: | --- |
| SAM-Med2D | code_ready_weight_missing | 1 | 0 | Download official sam-med2d_b.pth from the README Google Drive/Baidu link, then place it at assets/checkpoints/sam-med2d_b.pth. |
| Med-SA Adapter Zoo OpticDisc | evaluated_in_main_table | 1 | 1 | 已接入 `phase2_lite.py` 并写入 `runs/phase3_full_with_medsa/metrics_with_medsa.csv`。 |
| Med-SA Adapter Zoo Microscope | evaluated_in_main_table | 1 | 1 | 已接入 `phase2_lite.py` 并写入 `runs/phase3_full_with_medsa/metrics_with_medsa.csv`。 |
| nnU-Net v2 | preprocessed_training_ready | 1 | 1 | 已完成 2D planning/preprocessing；下一步运行 `scripts/nnunet_train_phase3_fold0.ps1`。 |
| Medical SAM2 / MedSAM2 | not_cloned_project_page_identified | 0 | 0 | Resolve the exact official repo/checkpoint from the Medical-SAM2 project page before cloning; then add video/volume protocol. |

## 对当前论文路线的影响

- `SAM-Med2D` 是最该补的 promptable medical SAM 强基线，但官方权重还没落到本地。
- `Med-SA Adapter Zoo` 两个 adapter 已经进入主表；当前更适合作为 domain-near sanity baseline，而不是最终强医学基线。
- `nnU-Net` 已经有代码、Phase3 raw dataset 和预处理缓存，但它是监督训练 baseline，必须训练后才能进入最终对比，不能直接和 promptable SAM 混为一谈。
- `MedSAM2 / Medical SAM2` 仍需要确认精确官方仓库和权重入口，适合支撑跨切片/视频 memory claim。

## 下一步最短路径

1. 运行 nnU-Net fold0 训练，得到监督式强参考。
2. 手动或脚本下载 `sam-med2d_b.pth` 后写 SAM-Med2D runner。
3. 训练完成后把 nnU-Net 20-case test predictions 汇总到 Phase3 主表的 supervised reference 区。
