# SAM 工作区整理、老师猜想验证与下一步方向

日期：2026-06-24  
工作目录：`D:\SAM`

## 1. 当前工作区整理结论

这个工作区现在不是单一项目，而是一个合并型研究工作台。建议先不删除历史目录，改用“主线入口 + 历史材料 + 实验结果”的方式管理。

| 区域 | 当前定位 | 建议 |
|---|---|---|
| `code/demos/medical_sam_click_app.py` | 当前 FastAPI + Canvas 网页 demo 主入口 | 保留为权威 demo，但后续应拆分 HTML/JS、后端 API、模型 wrapper。 |
| `code/matcher/` | 当前 Matcher 改造主线 | 作为后续 Stage 1/2/3、feedback、K-aware matching 的主要修改位置。 |
| `Matcher/` | 旧的 Matcher 源码树/对照树 | 暂时只读，作为原始实现参考，不再直接改。 |
| `sam3/`、`sam2/` | 外部模型源码 | 视为 vendor 源码，除 Windows 兼容补丁外不做大改。 |
| `assets/checkpoints/` | 本地权重集中区 | 保留。已确认有 `sam3_modelscope/sam3.pt`、SAM ViT-B/H、SAM2、MedSAM、DINOv2 等。 |
| `code/benchmarks/` | 现有 benchmark 与分析脚本 | 保留。新增轻量检查脚本也放这里。 |
| `runs/` | 已有实验输出 | 不纳入源码整理，不删除。后续报告只引用关键 CSV/JSON。 |
| `docs/phase*`、`docs/prompt_selector*` | 历史阶段报告 | 建议保留，但新增一个索引说明哪些是当前主线，哪些是历史草案。 |
| `docs/research/sam3_matcher_teacher_direction_report.md` | 老师方向逐项解释 | 作为这条 SAM3 + Matcher 路线的主参考文档。 |

## 2. 已完成的 demo 行为修正

用户反馈的问题：使用网页 demo 里的 SAM3 point 时，它不像累计前几个 prompt 后再给分割，而像只根据最新一次结果给出。

定位结果：

1. 前端原本每点击一个点就立即调用 `runClick()`。
2. 请求体会带当前累计 `points` 数组，但 SAM3 推理较慢，连续点击时多次异步请求可能互相覆盖。
3. 用户没有办法选择“先累计点，再按 Run 一次性分割”。
4. 这会造成体验上像“只看最新 prompt/最新请求结果”。

已做修改：

1. 在 Prompt Controls 中新增 `Point Run Mode`：
   - `Auto after each click`
   - `Manual accumulate then Run`
2. 新增 `Point Scope`：
   - `Cumulative points`
   - `Latest point only`
3. SAM3 默认设为：
   - `Manual accumulate then Run`
   - `Cumulative points`
4. 其他模型默认回到：
   - `Auto after each click`
   - `Cumulative points`
5. Run 和 Compare 都遵循相同 point scope，避免对照实验使用不同 prompt。

新增检查脚本：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_demo_prompt_controls.py
```

检查结果：已通过。

## 3. 老师猜想的最小实验验证

### 3.1 本轮已经跑通的 smoke 结果

测试图：`code/gradio_demo/images/elephant1.png`  
模型：`assets/checkpoints/sam3_modelscope/sam3.pt`  
设备：`cuda:0`

| 实验 | Prompt | 结果 |
|---|---|---|
| SAM3 text noun | `elephant` | Instances: 2，Scores: 0.95, 0.95，Union area: 54.35% |
| SAM3 text action | `execute` | Instances: 0，Union area: 0.00% |
| SAM3 point | foreground 点 `(256,256)` | score 0.695，mask area 0.30% |
| SAM3 point | foreground 点 `(120,220)` | score 0.875，mask area 4.58%，box `[32,67,134,244]` |
| SAM3 point | foreground `(120,220)` + background `(395,220)` | score 0.910，mask area 4.58%，box `[32,67,134,244]` |

初步结论：

1. `execute` 动作词直接作为 SAM3 text prompt 失败，支持老师的“文本提示会失效”猜想。
2. 名词类 prompt `elephant` 可以稳定找到实例，说明 SAM3 文本更适合短名词概念。
3. point prompt 对点位很敏感。正确点位能提高 score，background 点还能进一步提高 score；但本轮没有做人工可视化精确选点，所以还不能声称 point 已经完整分出整只大象。
4. 下一步需要用网页 demo 的“手动累计 + 累计点”模式，人工选更准确的 foreground/background 点，保存 overlay 和数值。

### 3.2 正式验证实验设计

目标是验证老师的三条猜想：

1. SAM3 point/foreground/background 对自然物体有效。
2. SAM3 text 对动作、关系、复杂属性表达容易失效。
3. Matcher Stage 1 的点集经过 feedback 后，matching 是否改善。

建议实验集：

| 场景 | 图像来源 | Prompt 设计 |
|---|---|---|
| 自然多实例 | `code/gradio_demo/images/elephant*.png`、COCO/LVIS/SA-Co | `elephant`、foreground/background points |
| 文本失效 | RefCOCO/gRefCOCO 或自然图手写描述 | `execute`、`white girl`、`person doing ...` |
| 多目标聚类 | 多个同类实例图 | Stage 1 100 points + K-means/DBSCAN |
| feedback | 同一图的正确点/错误点 | baseline points vs feedback-updated points |

记录指标：

| 指标 | 用途 |
|---|---|
| SAM3 text instances | 判断 text prompt 是否返回目标 |
| SAM3 score | 粗略判断 prompt 质量 |
| mask area / box | 检查是否只分到局部或过大区域 |
| point count | 比较 latest-only 和 cumulative |
| overlay 截图 | 给老师看最直观 |
| Stage 1 precision/coverage | 判断 Matcher matching 点是否更好 |
| final Dice/IoU | 如果有 GT，则做最终量化 |

## 4. 下一步可以尝试的方向

### 方向 A：先把网页 demo 做成可靠实验工具

适合短期给老师看。

要做：

1. 支持保存每次运行的 JSON：模型、prompt、points、scope、score、mask area、box。
2. 支持一键导出 overlay PNG。
3. 支持同一图的 text / latest point / cumulative points / foreground+background 对照。
4. 增加 run id，避免多次点击结果混淆。

优点：最快形成可视化证据。  
风险：偏 demo，论文贡献还需要方法部分支撑。

### 方向 B：实现 Matcher feedback 机制

适合对应老师“Stage 1 给 100 points 后人工选 a/b 个点更新 target feature”的建议。

要做：

1. 在 `code/matcher/Matcher.py::patch_level_matching` 输出 top-k points 和 similarity。
2. 增加 positive/negative feedback 点输入。
3. 用正点拉近 prototype，用负点惩罚相似区域。
4. 比较 feedback 前后的 precision、coverage、最终 mask。

优点：更像方法创新。  
风险：需要数据和 GT 支撑，否则只有可视化不够。

### 方向 C：K-aware 多目标 Matcher

适合解决“多个物体且非常多时 sample 是否可行”。

要做：

1. 将 matched points 聚类为 K 个候选实例。
2. 每个 cluster 内独立 sample point prompts。
3. 每个 cluster 单独做 proposal scoring。
4. 最后跨 cluster 做 mask NMS。

优点：直接回应老师关于 K 个物体和 instance-level matching 的问题。  
风险：聚类估计 K 的稳定性要调。

### 方向 D：论文式主线整合

适合后续写作。

可以命名为：

`Feedback-Guided Cluster-Aware Prompt Sampling for SAM3/Matcher`

主线：

1. SAM3 text 对复杂表达不稳定。
2. 视觉 prompt 可靠但人工点成本高。
3. Matcher 自动挖点，但多目标和 outlier 导致混淆。
4. 用少量人工 feedback 校准 feature matching。
5. 用 cluster-aware sampling 生成实例级 prompts。
6. 用 instance-level matching 过滤 proposals。

## 5. 建议立即执行顺序

1. 用新 demo 设置做 5 张自然图可视化对照，确认 cumulative point 是否解决交互问题。
2. 在 `docs/` 放一份小实验报告，包含 overlay、prompt、score。
3. 开始实现 Matcher feedback，先只做 Stage 1 点集变化，不急着完整跑 SAM proposal。
4. 再实现 K-aware sampling，解决多目标实例问题。
5. 最后把 demo 工具、Matcher 方法和 benchmark 结果统一成一条论文叙事。
