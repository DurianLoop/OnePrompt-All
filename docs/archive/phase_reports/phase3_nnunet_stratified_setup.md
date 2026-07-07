# Phase3 nnU-Net 分层 Split 设置

## 为什么重做

旧的 `Dataset901_OCSAMPhase3` 按 manifest 顺序做 80/20 split。由于 manifest 前 50 个是 IDRID，后 50 个是 MonuSeg，最终 held-out test 全部来自 MonuSeg，不能作为跨数据集强基线。

新的 `Dataset902_OCSAMPhase3Stratified` 使用 dataset-wise stratified split：

| Dataset | Train | Test |
| --- | ---: | ---: |
| IDRID | 40 | 10 |
| MonuSeg | 40 | 10 |
| Total | 80 | 20 |

当前状态：

- Raw dataset 已生成：`runs/nnunet/nnUNet_raw/Dataset902_OCSAMPhase3Stratified`
- nnU-Net 2D planning/preprocessing 已完成：`runs/nnunet/nnUNet_preprocessed/Dataset902_OCSAMPhase3Stratified`
- 验收脚本已通过：`code/benchmarks/check_nnunet_stratified_phase3.py`

## 已准备的入口

生成 raw dataset：

```powershell
cd D:\SAM
.\scripts\prepare_nnunet_phase3_stratified.ps1
```

预处理：

```powershell
cd D:\SAM
.\scripts\nnunet_preprocess_phase3_stratified.ps1
```

这一步当前已经完成；除非你修改 split 或 raw dataset，否则不用重复运行。

训练 250 epoch：

```powershell
cd D:\SAM
.\scripts\nnunet_train_phase3_stratified_fold0_250epochs.ps1
```

预测 test split，使用 `checkpoint_best.pth`：

```powershell
cd D:\SAM
.\scripts\nnunet_predict_phase3_stratified_fold0_250epochs.ps1
```

## 注意

这会保留旧的 `Dataset901_OCSAMPhase3`，不覆盖已有 250-epoch 结果。后续论文主表应优先使用 `Dataset902_OCSAMPhase3Stratified`。
