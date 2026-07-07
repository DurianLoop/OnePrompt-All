# 教师方向闭环整理总报告

日期：2026-06-24

## 1. 当前主线

当前项目可以收敛成一句话：

```text
当 SAM3 text prompt 对复杂文本、动作词或领域词不稳定时，用 Matcher/DINO 先产生 matched points，再通过少量正负点 feedback 修正点分布，按实例聚类采样 point prompts，送入 SAM/SAM3 生成 mask proposals，最后用 instance-level feature matching 过滤并估计 K 个实例。
```

这条主线把老师提出的几个问题串成了一个完整链路：

```text
text 失败 -> point 更可控 -> Stage 1 matched points -> 正负点 feedback -> Stage 2 聚类采样 -> SAM proposals -> Stage 3 instance-level threshold -> K_pred
```

## 2. 对老师问题的逐条回应

| 老师问题/猜想 | 当前回应 | 证据位置 |
| --- | --- | --- |
| SAM3 point 效果较好，点到 foreground/background 后应能分割 | 初步支持。point 对指定单个实例更可控，正负点能改善 mask score 和范围。 | `docs/reports/sam3/sam3_teacher_minimal_validation_report.md` |
| SAM3 text 对 `white girl`、`execute` 失败 | 支持。`girl` 可返回实例，但 `white girl` 返回 0；`execute` 在自然图中返回 0。 | `runs/teacher_minimal_validation/results.csv` |
| demo 里 point 只按最新点，不累计 prompt | 已修。demo 增加 `Point Scope: cumulative/latest` 和 `Point Run Mode: auto/manual`。 | `code/demos/medical_sam_click_app.py`、`code/benchmarks/check_demo_prompt_controls.py` |
| Stage 1 给出很多 points 后，选正点/错点 feedback 是否让 matching 更好 | 支持最小版本。空间 reranking 后 top-k 更靠近正点，负点邻域候选从 16 降到 0。 | `docs/reports/matcher/matcher_stage1_feedback_report.md` |
| matching 过程如何完成 | 已打开 Stage 1：reference box -> DINO feature -> prototype similarity -> top-k points -> heatmap/overlay。 | `code/benchmarks/export_matcher_stage1_points.py` |
| sample N% matched points，多次送入 SAM 得到 m 个分割是否可行 | 可行，已跑通。elephant case 从 12 组 sampled prompts 生成 12 个 SAM proposals。 | `docs/reports/matcher/matcher_stage2_sam_proposals_report.md` |
| 多个物体很多时，代码如何知道 K | 初步可行。horse-head 多实例 case 中 DBSCAN 得到 2 个 cluster，Stage 3 最终 `K_pred=2`。 | `docs/reports/matcher/matcher_multi_instance_horse_validation_report.md` |
| instance-level threshold 高于保留、低于丢弃 | 已实现最小 feature-level 版本。proposal prototype 与 reference prototype 做 cosine similarity，再结合面积、box、coverage、NMS 过滤。 | `code/benchmarks/feature_match_stage3_proposals.py` |
| 大图超过 DINO 规定大小且不能降采样怎么办 | 已在第 7 节补做 sliding-window Stage 1 最小实现：tile 后逐块 DINO matching，再恢复原图坐标。 | `runs/research_direction/p0_p3_closure/p15_sliding_window/elephant3/` |

## 3. 已完成的最小闭环

### 3.1 SAM3 text/point 最小验证

关键现象：

| case | 结果 | 结论 |
| --- | --- | --- |
| `elephant` text | 返回 2 个实例，score 0.953 | 简单名词 text 可用 |
| `execute` text | 返回 0 | 动作词 text 失败 |
| `girl` text | 返回 3 个实例，score 0.906 | 简单人物名词可用 |
| `white girl` text | 返回 0 | 复合属性 text 失败 |
| elephant 单点 | score 0.652，mask 偏局部 | 单点可控但可能不完整 |
| elephant 两个正点 | score 0.676，mask 扩展到目标主体 | 累计正点改善完整性 |
| elephant 两正一点负 | score 0.738，保持目标主体 | 正负点 feedback 进一步改善 |

可复现命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sam3_teacher_minimal_validation.py --out-dir runs\teacher_minimal_validation
```

输出：

```text
runs/teacher_minimal_validation/results.csv
runs/teacher_minimal_validation/results.json
runs/teacher_minimal_validation/overlays/
```

### 3.2 Matcher Stage 1：matched points 导出

已跑通链路：

```text
reference image/box -> DINO feature -> reference prototype -> target similarity heatmap -> top-k matched points
```

elephant case 结果：

| 指标 | 结果 |
| --- | ---: |
| top-k | 100 |
| threshold | 0.55 |
| selected points | 9 |
| best point | `(268.108, 330.811)` |
| best similarity | 0.605155 |

可复现命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

### 3.3 Stage 1 feedback：正负点 reranking

feedback 设置：

```text
positive: 268,331
negative: 30,430;455,420
sigma: 42
positive_weight: 0.18
negative_weight: 0.22
```

关键结果：

| 指标 | feedback 前 | feedback 后 |
| --- | ---: | ---: |
| selected count | 9 | 43 |
| best similarity | 0.605155 | 0.780131 |
| top-k 到正点平均距离 | 128.379 | 83.822 |
| top-k 正点 70px 邻域点数 | 22 | 43 |
| top-k 负点 70px 邻域点数 | 16 | 0 |

结论：老师关于“少量正负点能改善 matching”的猜想得到最小证据支持。但当前是空间 reranking，不是最终论文级 feature/prototype update。

### 3.4 Stage 2：聚类与 prompt sampling

elephant case 使用 feedback 后 selected points 做 DBSCAN：

| 指标 | 结果 |
| --- | ---: |
| 输入 selected candidate points | 43 |
| anchor-distance 过滤后点数 | 40 |
| DBSCAN cluster count / estimated K | 1 |
| noise points | 0 |
| prompt groups | 12 |

生成的 prompt groups 覆盖 `1/3/5` 点三种设置，每种采样 4 次。

### 3.5 Stage 2b：SAM proposals

12 组 prompt groups 已真实送入本地 SAM ViT-B：

| 指标 | 结果 |
| --- | ---: |
| proposal count | 12 |
| SAM score mean/min/max | 0.905409 / 0.709140 / 0.987186 |
| mask area mean/min/max | 33.3608% / 1.0176% / 70.9880% |
| sample size 1 area mean | 21.3485% |
| sample size 3 area mean | 24.5443% |
| sample size 5 area mean | 54.1897% |

观察：5 点 prompts 虽然 SAM score 高，但容易过度扩张到背景，因此必须做 Stage 3 filtering，不能只看 SAM score。

### 3.6 Stage 3：mask-level 和 feature-level filtering

mask-level filtering 在 12 个 proposals 中保留 1 个：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| threshold candidates | 1 |
| retained masks / K_pred | 1 |
| retained group id | 5 |
| retained SAM score | 0.936279 |
| retained mask area | 12.7992% |

feature-level instance matching 进一步验证该 proposal：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| feature candidates | 1 |
| retained masks / K_pred | 1 |
| retained group id | 5 |
| retained instance similarity | 0.784939 |
| retained feature score | 0.986067 |

结论：Stage 3 已经从“按 mask 规则筛”推进到“按 DINO/Matcher feature 相似度筛”，更接近老师说的 instance-level matching。

### 3.7 多实例自然场景：horse-head case

该实验专门回应“多个物体时如何知道 K”。

输入：

```text
reference: code/gradio_demo/images/horse2.png
target:    code/gradio_demo/images/horse3.png
reference box: 120,35,390,330
```

结果：

| 阶段 | 指标 | 结果 |
| --- | --- | ---: |
| Stage 1 | selected points | 26 |
| Stage 2 | DBSCAN clusters | 2 |
| Stage 2 | noise points | 5 |
| Stage 2b | SAM proposals | 16 |
| Stage 3 | feature candidates | 6 |
| Stage 3 | retained masks | 2 |
| Stage 3 | `K_pred` | 2 |

保留 proposal：

| group id | cluster id | sample size | SAM score | instance similarity | mask area |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 3 | 0.959684 | 0.682757 | 13.1212% |
| 12 | 1 | 1 | 0.926443 | 0.610000 | 7.4017% |

结论：`matched points -> clustering -> per-cluster prompt sampling -> SAM proposals -> feature-level matching` 可以在一张含两匹马的自然图中估计出 `K_pred=2`。这不是大 benchmark，但已经是老师猜想的最小多实例闭环。

## 4. 当前不能过度宣称的点

1. 现在的 Stage 1 feedback 主要是空间 reranking，还没有完成真正的 feature/prototype update。
2. 目前自然图验证样本少，elephant 和 horse-head 只能证明链路可行，不能证明泛化性能。
3. SAM3 text failure 目前只验证了少量 prompt，后续需要扩展成小型 failure taxonomy。
4. 大图不降采样的 sliding-window matching 已在第 7 节补做最小实现，但还没有接完整 tile-level mask merge。
5. 医学图像迁移还没有跑完整闭环，当前只能说方向合理，不能写成医学数据上已有效。
6. horse-head case 的 brown horse mask 仍有背景粘连，说明 Stage 3 还需要更强的形状、coverage 或 feature filtering。

## 5. 下一步优先级研究方案

### P0：把当前方法写成稳定主线

题目方向：

```text
Feedback-Guided Cluster-Aware Prompt Sampling for SAM3/Matcher
```

下一步具体任务：

1. 把 Stage 1 feedback 从空间 reranking 升级为 feature/prototype update。
2. 增加 2-3 个自然多实例 case，验证 `K_pred` 是否稳定。
3. 给每个 case 固定同一套指标：selected points、cluster count、proposal count、retained masks、instance similarity、K_pred。
4. 写一个统一 runner，把 Stage 1/2/3 命令串起来，减少手工命令错误。

### P1：SAM3 text failure taxonomy + fallback trigger

目标：把 `white girl`、`execute` 这类现象系统化，作为论文 motivation 和 demo fallback 逻辑。

推荐记录：

```text
prompt type: simple noun / attribute noun / action / referring expression / medical term
instances: 0 / 1 / many
score
area
是否触发 visual prompt fallback
```

### P1.5：大图 sliding-window no-downsample matching

目标：回应老师关于 DINO 输入大小限制的问题。

最小链路：

```text
large image -> overlapping tiles -> per-tile DINO matching -> restore coordinates -> merge top points -> optional SAM mask merge
```

### P2：医学小数据闭环

目标：把自然图上的机制迁移到医学目标，尤其是多病灶、多细胞核、小目标或低对比场景。

最小实验不需要马上做大 benchmark，先做：

```text
1 个 support image/box or mask
5-10 个 query images
Matcher original vs feedback-guided Matcher vs point-only SAM/SAM3
```

指标：

```text
Dice / IoU / false positive count / interaction count / K_pred error
```

### P3：Adapter / LoRA / fine-tuning

只有当 P0-P2 证明 prompt/matching 机制有价值后，再考虑训练 adapter。否则会过早进入训练成本高、变量多的路线。

## 6. 推荐给老师的简短汇报

```text
老师，我们现在已经把您的猜想做成了一个最小闭环。SAM3 text 对简单名词如 elephant/girl 可以工作，但 white girl 和 execute 会稳定失败；point prompt 更适合指定单个实例，并且累计正点、正负点 feedback 能改善 mask。Matcher 方向上，我们已经跑通 Stage 1 matched points 导出、正负点 feedback reranking、Stage 2 聚类采样、SAM proposal 生成，以及 Stage 3 feature-level instance matching。单目标 elephant case 得到 K_pred=1，多实例 horse-head case 得到 K_pred=2，说明“matched points 聚类 + 每 cluster 采样 + instance-level threshold”可以作为估计 K 和筛选实例的最小方案。当前还不能说已经完成大规模验证，下一步应该优先把 feedback 从空间 reranking 升级为 feature/prototype update，并补 2-3 个自然多实例 case，再迁移到医学图像小闭环。
```

## 7. P0-P3 最小闭环完成结论

日期：2026-06-29

本节是在第 5 节“下一步优先级研究方案”的基础上继续推进得到的结果。这里的“完成”指完成可复现的最小闭环、脚本和证据产物；不等价于完成大规模 benchmark 或 adapter 训练论文结果。

### 7.1 P0：稳定主线最小闭环

新增脚本：

| 脚本 | 作用 |
| --- | --- |
| `code/benchmarks/apply_stage1_feature_feedback.py` | 将 Stage 1 feedback 从空间 reranking 升级为 DINO feature/prototype update。 |
| `code/benchmarks/run_research_direction_case.py` | 将 Stage 1、feature feedback、Stage 2 sampling、SAM proposals、Stage 3 feature matching 串成统一 runner。 |

#### P0-1 feature/prototype feedback

在 elephant pair 上运行 feature/prototype update：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feature_feedback.py --stage1-dir runs\research_direction\stage1_matching\elephant_pair --positive-points "268,331" --negative-points "30,430;455,420" --sigma 42 --positive-weight 0.35 --negative-weight 0.25 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\p0_p3_closure\p0_feature_feedback\elephant_pair
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p0_feature_feedback/elephant_pair/
```

关键结果：

| 指标 | feature feedback 前 | feature feedback 后 |
| --- | ---: | ---: |
| selected count | 9 | 29 |
| top-k 正点 70px 邻域点数 | 22 | 38 |
| top-k 负点 70px 邻域点数 | 16 | 0 |
| top-k 到正点平均距离 | 128.379 | 82.600 |

结论：P0 的核心缺口已经补上。相比之前空间 reranking 的 selected count 43，feature/prototype update 的 selected count 29 更保守，但它是真正按 DINO feature prototype 更新 similarity，和老师说的“更新 target feature 后 matching 是否更好”更一致。

#### P0-2 多实例自然图补验

新增 elephant3 多实例 case：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\run_research_direction_case.py --case-name elephant3_multi_actual --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant3.png --reference-box 0,60,295,425 --positive-points "350,300" --negative-points "220,120;500,130" --top-k 120 --threshold 0.48 --eps 70 --min-samples 1 --sample-sizes 1 --samples-per-cluster 2 --max-groups 6 --min-area-pct 0.1 --max-area-pct 55 --out-root runs\research_direction\p0_p3_closure\p0_runner
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p0_runner/elephant3_multi_actual/
```

关键结果：

| 阶段 | 结果 |
| --- | ---: |
| Stage 1 selected points | 120 |
| feature feedback selected points | 120 |
| Stage 2 candidate points | 29 |
| Stage 2 estimated K / clusters | 1 |
| SAM proposals | 2 |
| Stage 3 feature candidates | 0 |
| retained masks / `K_pred` | 0 |

解释：horse-head case 仍是当前最强的多实例正例，能得到 `K_pred=2`；elephant3 是更难的远景多目标场景，完整 runner 虽然能跑通，但 Stage 3 没有保留实例，说明当前阈值和 feature prototype 对小尺度、多尺度、遮挡目标仍偏保守。这个失败不是坏事，它明确指出下一步需要多尺度 reference、tile-level proposal 和更宽松/自适应 threshold。

### 7.2 P1：SAM3 text failure taxonomy + fallback trigger

新增脚本：

```text
code/benchmarks/build_sam3_text_failure_taxonomy.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\build_sam3_text_failure_taxonomy.py --out-dir runs\research_direction\p0_p3_closure\p1_text_taxonomy
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p1_text_taxonomy/
```

关键结果：

| 指标 | 结果 |
| --- | ---: |
| taxonomy records | 13 |
| fallback required | 6 |

fallback 触发规则：

```text
instances == 0
prompt_type 属于 action / attribute_or_phrase / medical_term
instances 过多
score 过低
area 异常
```

结论：P1 已经从“几个失败例子”变成可复现的 failure taxonomy 表。当前证据支持把 `white girl`、`execute`、医学术语等 prompt 标成高风险，一旦 text 失败或异常，就回退到 visual prompt / Matcher points / feedback 路线。

### 7.3 P1.5：大图不降采样 sliding-window matching

新增脚本：

```text
code/benchmarks/export_sliding_window_stage1_points.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_sliding_window_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant3.png --reference-box 0,60,295,425 --tile-size 320 --overlap 80 --per-tile-top-k 16 --top-k 120 --threshold 0.48 --out-dir runs\research_direction\p0_p3_closure\p15_sliding_window\elephant3
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p15_sliding_window/elephant3/
```

关键结果：

| 指标 | 结果 |
| --- | ---: |
| tile count | 6 |
| merged top-k points | 120 |
| selected count | 96 |

结论：P1.5 的技术问题已经有最小实现：target 不再整图 resize，而是切成重叠 tiles，逐 tile 做 DINO matching，再把点坐标恢复到原图坐标系。下一步需要把 tile-level points 接到 Stage 2/3 的 mask merge/NMS，但“不能降采样时怎么办”的核心回答已经落地。

### 7.4 P2：医学小数据闭环

新增脚本：

| 脚本 | 作用 |
| --- | --- |
| `code/benchmarks/medical_small_data_gate.py` | 扫描本地医学图像和 mask，判断能否做 Dice/IoU。 |
| `code/benchmarks/evaluate_medical_prediction_pairs.py` | 对已有医学预测和 GT mask 配对，计算 Dice/IoU。 |

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\medical_small_data_gate.py --out-dir runs\research_direction\p0_p3_closure\p2_medical_gate
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\evaluate_medical_prediction_pairs.py --out-dir runs\research_direction\p0_p3_closure\p2_medical_eval
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p2_medical_gate/
runs/research_direction/p0_p3_closure/p2_medical_eval/
```

关键结果：

| 指标 | 结果 |
| --- | ---: |
| medical image candidates | 5000 |
| mask candidates | 256 |
| can run Dice/IoU | True |
| available prediction/GT pairs evaluated | 1 |
| Dice mean | 0.020698 |
| IoU mean | 0.010457 |

解释：本地确实有可用于医学评估的数据，尤其是 `code/datasets/IDRID` 中的 fundus images 和 combined masks。但目前能自动配对到的已有 SAM 预测只有 1 个，且 Dice/IoU 很低。因此 P2 的结论不是“医学上已经有效”，而是：

```text
医学小数据评估闭环已经打通；当前已有预测质量很弱，必须重新组织 support/query、prompt/feedback、Stage 3 filtering 后再做正式医学实验。
```

### 7.5 P3：Adapter / LoRA gate

新增脚本：

```text
code/benchmarks/adapter_lora_gate.py
```

运行命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\adapter_lora_gate.py --out-dir runs\research_direction\p0_p3_closure\p3_adapter_gate
```

输出目录：

```text
runs/research_direction/p0_p3_closure/p3_adapter_gate/
```

gate 结果：

| 条件 | 是否满足 |
| --- | --- |
| P0 feature feedback evidence | True |
| P1 text taxonomy evidence | True |
| P2 medical data with masks | True |
| P2 medical eval pair exists | True |
| ready_for_training | True |
| decision | adapter_training_ready |

结论：P3 已完成“是否进入训练”的门控判断。当前可以准备 Adapter/LoRA smoke training，但还不能声称 Adapter/LoRA 已经有效，因为还没有训练、没有 held-out evaluation。更准确的下一步是先建立 IDRID 或 MonuSeg 的固定 train/val split，跑一个最小 adapter/LoRA smoke，再和 P0 的 training-free pipeline 对比。

### 7.6 总结：现在可以对外声称什么

可以声称：

1. P0 的 feature/prototype feedback 已经实现并验证，在 elephant pair 上 selected count 从 9 提升到 29，负点邻域候选降到 0。
2. P0 的统一 runner 已经能完整串起 Stage 1/2/3，并在 elephant3 上给出一个有价值的失败边界案例。
3. P1 text failure taxonomy 已经落地，13 条记录中 6 条触发 fallback。
4. P1.5 sliding-window no-downsample Stage 1 已经落地，elephant3 被切成 6 个 tile，合并后 selected points 为 96。
5. P2 医学数据评估闭环已经打通，本地有医学 image/mask，且已有 1 个 prediction/GT pair 可算 Dice/IoU。
6. P3 adapter/LoRA 进入 smoke training 的条件已经满足，但训练本身尚未执行。

不能声称：

1. 不能声称当前方法已经在大规模自然图上稳定泛化。
2. 不能声称 elephant3 多实例已成功分割；它当前 `K_pred=0`，是失败边界案例。
3. 不能声称医学图像上方法有效；当前唯一自动配对预测的 Dice 只有 0.020698。
4. 不能声称 Adapter/LoRA 有性能提升；当前只是 gate ready。

### 7.7 下一步最具体动作

下一步不建议继续扩散方向，而是做一个很小但硬的实验：

```text
固定 IDRID 5 train / 5 val：
1. 用 training-free P0 pipeline 跑 support/query；
2. 记录 Dice/IoU、false positives、K_pred；
3. 再跑一个 Adapter/LoRA smoke；
4. 比较是否真的超过 training-free pipeline。
```

如果这个医学小闭环不能超过当前 training-free 方案，就不要进入大规模训练；继续优先改 feature feedback、tile-level proposal merge 和 Stage 3 threshold。

## 8. Elephant3 sliding-window 多实例调参闭环

日期：2026-06-30

上一版报告里，elephant3 的完整 runner 得到 `K_pred=0`。进一步检查后发现，这并不是 Stage 1 没有召回目标，而是旧 runner 没有使用 sliding-window Stage 1 的 96 个点，导致 Stage 2 只聚成 1 个 cluster、只生成 2 个 SAM proposals，Stage 3 随后把候选全部过滤掉。

本轮将 sliding-window points 接入 Stage 2/3 后，得到多实例成功版本：

| 设置 | 结果 |
| --- | ---: |
| sliding-window selected points | 96 |
| DBSCAN eps | 28 |
| min_samples | 2 |
| Stage 2 clusters / estimated K | 6 |
| Stage 2 prompt groups | 24 |
| Stage 2 SAM proposals | 24 |
| Stage 3 candidates after thresholds | 16 |
| Stage 3 retained masks / `K_pred` | 6 |

关键输出：

```text
runs/research_direction/p0_p3_closure/elephant3_tuning/stage2_eps28_ms2/
runs/research_direction/p0_p3_closure/elephant3_tuning/proposals_eps28_ms2/
runs/research_direction/p0_p3_closure/elephant3_tuning/stage3_eps28_relaxed/
```

关键结论：

1. elephant3 的左图点级召回确实已经不错，问题主要在 Stage 2/3 参数，而不是 Stage 1 matching 本身。
2. `eps=70` 会把多实例点聚成 1 类，导致 proposal 太少；`eps=45/55` 会把中间目标合并过粗；`eps=20` 会过碎并错误保留大背景；当前最适合展示的是 `eps=28`。
3. 调参后 elephant3 可以从 `K_pred=0` 提升到 `K_pred=6`，说明 sliding-window matched points 可以被转成多实例 SAM proposals 并通过 Stage 3 保留下来。
4. 当前结果还不是最终论文级最优：左侧近邻 elephant 仍有合并现象，后续需要更细的局部聚类、mask merge/NMS 或每个 tile 内独立 proposal 后再合并。

可复现命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\convert_sliding_window_points_to_feedback.py --merged-points-json runs\research_direction\p0_p3_closure\p15_sliding_window\elephant3\merged_points.json --out-dir runs\research_direction\p0_p3_closure\elephant3_tuning\sliding_feedback
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\p0_p3_closure\elephant3_tuning\sliding_feedback --selected-only --cluster-method dbscan --eps 28 --min-samples 2 --sample-sizes 1,3 --samples-per-cluster 2 --max-anchor-distance 0 --out-dir runs\research_direction\p0_p3_closure\elephant3_tuning\stage2_eps28_ms2
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\p0_p3_closure\elephant3_tuning\stage2_eps28_ms2 --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 24 --out-dir runs\research_direction\p0_p3_closure\elephant3_tuning\proposals_eps28_ms2
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\p0_p3_closure\elephant3_tuning\proposals_eps28_ms2 --stage1-dir runs\research_direction\p0_p3_closure\p0_runner\elephant3_multi_actual\stage1 --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.42 --min-score 0.84 --min-area-pct 0.1 --max-area-pct 65 --max-bbox-area-pct 80 --min-cluster-coverage 0.05 --min-prompt-coverage 1.0 --nms-iou 0.65 --max-retained-per-cluster 1 --out-dir runs\research_direction\p0_p3_closure\elephant3_tuning\stage3_eps28_relaxed
```
