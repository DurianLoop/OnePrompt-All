# Phase3 nnU-Net 训练准备记录

## 当前状态

Phase3 的 `IDRID + MonuSeg` 100-case 数据已经转换为 nnU-Net v2 可读的 2D 数据集：

- Dataset ID: `901`
- Dataset name: `Dataset901_OCSAMPhase3`
- Raw dataset: `runs/nnunet/nnUNet_raw/Dataset901_OCSAMPhase3`
- Preprocessed dataset: `runs/nnunet/nnUNet_preprocessed/Dataset901_OCSAMPhase3`
- Split: `train=80`, `test=20`
- Image format: single-channel grayscale PNG
- Label format: binary foreground/background PNG
- Reader/writer: `NaturalImage2DIO`

这次把原来的 RGB PNG 改成了灰度 PNG。原因是 nnU-Net 的 `NaturalImage2DIO` 会把 RGB PNG 读成 3 个通道，而我们的 `dataset.json` 只声明了一个输入通道，导致预处理阶段出现 `normalization_schemes[c]` 越界。改成灰度后，fingerprint、planning 和 2D preprocessing 已经跑通。

## 已完成的命令

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe code\benchmarks\prepare_nnunet_phase3_dataset.py
```

```powershell
$env:nnUNet_raw='D:\SAM\runs\nnunet\nnUNet_raw'
$env:nnUNet_preprocessed='D:\SAM\runs\nnunet\nnUNet_preprocessed'
$env:nnUNet_results='D:\SAM\runs\nnunet\nnUNet_results'
D:\SAM\conda_envs\sam_gpu\Scripts\nnUNetv2_plan_and_preprocess.exe -d 901 -c 2d -npfp 1 -np 1 --clean
```

## 下一步训练

从 `D:\SAM` 运行：

```powershell
.\scripts\nnunet_train_phase3_fold0.ps1
```

默认脚本不再使用 nnU-Net 的 `1000 epochs` 正式 trainer，而是先使用内置的 `nnUNetTrainer_100epochs` 做 sanity baseline。原因是当前只有 100 个 case，直接跑满 1000 epochs 容易把验证集波动误当成真实性能。

训练脚本额外设置了：

- `nnUNet_n_proc_DA=0`：Windows 上使用单线程 data augmenter，避免 multiprocessing `WinError 5`。
- `nnUNet_def_n_proc=1`：后处理、导出等默认单进程。
- `MPLCONFIGDIR=runs/nnunet/matplotlib_cache`：避免 matplotlib 往用户目录写 cache 时权限失败。

训练完成后预测 20 个 test case：

```powershell
.\scripts\nnunet_predict_phase3_fold0.ps1
```

预测脚本强制使用：

- `-tr nnUNetTrainer_100epochs`
- `-chk checkpoint_best.pth`

这样评估的是验证 EMA pseudo Dice 最好的权重，而不是训练结束时的 `checkpoint_final.pth`。

如果 100 epochs 的验证曲线还在明显上升，再把脚本里的 trainer 从 `nnUNetTrainer_100epochs` 改成 `nnUNetTrainer_250epochs`，作为正式小数据 baseline。暂时不建议直接使用默认 `nnUNetTrainer` 的 1000 epochs。

也可以直接使用已经准备好的 250-epoch 脚本：

```powershell
.\scripts\nnunet_train_phase3_fold0_250epochs.ps1
.\scripts\nnunet_predict_phase3_fold0_250epochs.ps1
```

## 论文定位

nnU-Net 不应该和 SAM/MedSAM/SAM2 混成同一种 promptable 方法。它在这里的角色是监督式强基线或近似上界，用来回答一个关键审稿问题：

如果我们的方法只是 prompt 工程，是否还能在同样数据上接近或超过任务专用监督模型？

因此，后续主表建议分成两块：

- Promptable models: SAM ViT-B, SAM ViT-H, MedSAM, SAM2 Tiny, Med-SA adapters, OCSAM-X variants.
- Supervised references: nnU-Net 2D, 后续可加 U-Net/TransUNet/SAM-Med2D fine-tuned variant.

如果 OCSAM-X 无法超过 nnU-Net，也仍然可以成立为交互式 promptable segmentation 方法，但 claim 要写成交互效率、prompt 鲁棒性和少标注/零训练设置，而不是监督分割 SOTA。
