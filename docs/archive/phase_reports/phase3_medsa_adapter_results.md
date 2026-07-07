# Phase3 Med-SA Adapter 主表结果

本报告把已下载的两个 Medical-SAM-Adapter Zoo 权重真正跑进了 Phase3 100-case 主表。

- `medsa_opticdisc`: `OpticDisc_Fundus_SAM_1024.pth`
- `medsa_microscope`: `White Blood Cell_MicroScope_sam_1024.pth`
- 评估 prompt: `gt_box`, `auto_box`, `auto_medical_box_v1`, `auto_medical_point_v1`

## 全模型均值

| Model | Rows | Mean Dice |
| --- | ---: | ---: |
| medsa_microscope | 400 | 0.325 |
| medsa_opticdisc | 400 | 0.2873 |
| medsam_vit_b | 1200 | 0.326 |
| sam2_tiny | 1900 | 0.2968 |
| sam_vit_b | 1900 | 0.2973 |
| sam_vit_h | 1900 | 0.3154 |

## Med-SA 细分结果

| Model | Dataset | Prompt | Rows | Mean Dice |
| --- | --- | --- | ---: | ---: |
| medsa_microscope | IDRID | auto_box | 50 | 0.4202 |
| medsa_microscope | IDRID | auto_medical_box_v1 | 50 | 0.2618 |
| medsa_microscope | IDRID | auto_medical_point_v1 | 50 | 0.2639 |
| medsa_microscope | IDRID | gt_box | 50 | 0.2618 |
| medsa_microscope | MonuSeg | auto_box | 50 | 0.3445 |
| medsa_microscope | MonuSeg | auto_medical_box_v1 | 50 | 0.3446 |
| medsa_microscope | MonuSeg | auto_medical_point_v1 | 50 | 0.3562 |
| medsa_microscope | MonuSeg | gt_box | 50 | 0.347 |
| medsa_opticdisc | IDRID | auto_box | 50 | 0.4239 |
| medsa_opticdisc | IDRID | auto_medical_box_v1 | 50 | 0.2496 |
| medsa_opticdisc | IDRID | auto_medical_point_v1 | 50 | 0.2274 |
| medsa_opticdisc | IDRID | gt_box | 50 | 0.1011 |
| medsa_opticdisc | MonuSeg | auto_box | 50 | 0.3223 |
| medsa_opticdisc | MonuSeg | auto_medical_box_v1 | 50 | 0.3223 |
| medsa_opticdisc | MonuSeg | auto_medical_point_v1 | 50 | 0.3252 |
| medsa_opticdisc | MonuSeg | gt_box | 50 | 0.3267 |

## 判断

两个 adapter 已经完成工程接入；但它们来自 OpticDisc/WBC 任务，和 IDRID lesion、MonuSeg nuclei 并非完全同分布，所以当前只能作为 domain-near Med-SA baseline，而不是最终强结论。
