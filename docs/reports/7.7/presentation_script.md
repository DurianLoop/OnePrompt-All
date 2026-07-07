# 7.7 阶段研究汇报讲稿

对应网页：`D:\SAM\docs\reports\7.7\index.html`

建议讲法：先讲清楚“老师的问题是什么”，再讲“我现在如何把问题闭环成一条研究主线”，最后讲“已有结果说明什么、还差什么、下一步做什么”。整场汇报可以控制在 12-18 分钟；如果老师追问细节，再展开后面的“追问回答”。

---

## 0. 开场：一句话概括

老师好，我这次主要汇报从您上次布置的两个方向出发，到现在形成的一条研究主线。

最开始的问题是：SAM3 的 point prompt 效果比较好，但是 text prompt 对一些复杂描述，比如 `white girl`、`execute` 这类 prompt 会失效；所以我们不能只依赖文字，而需要一个机制自动找到 query 图里的目标点。  

现在我把这个问题整理成了一个 Feedback-Learned Matcher 的闭环：用一张 support/reference 图里的人工 mask 定义目标，用 Matcher 在另一张 query 图里找相似点，再把这些点送入 SAM 生成多个 proposal，最后用 feedback 和 learning 去判断哪些 proposal 应该保留。

这次网页里主要展示三部分：

1. 老师截图中每个问题的正面回应。
2. Matcher 三阶段闭环和 P0 到 P4 的推进结果。
3. 医学场景下 20 例、50 例、5-fold 训练测试的定量和定性结果。

---

## 1. 总览页：研究已经从现象验证推进到论文主线

网页最上面这一页我想先给一个整体结论。

目前的工作已经不是单独验证 SAM3 某个 prompt 是否有效，而是把它组织成了一个可复现实验链条：

- 20 例医学实验形成 240 个 SAM proposals。
- 50 例扩展实验形成 600 个 SAM proposals。
- 已经做了 review gallery、人工/伪标签组织、proposal learning、分域训练，以及 50 例 case-level 5-fold。

这里最重要的一句话是：原来的 Stage3 手写规则偏保守，候选池里明明有好 mask，但是 Stage3 经常没有保留；而可解释 learning 能明显改善 proposal keep/reject。

汇报时我会强调：这不是说现在已经可以完整分割所有病灶，而是说“候选生成 + 学习筛选”这条路线已经有了实验证据。

---

## 2. 到目前为止的研究路径

这一页按照时间线梳理整个项目。

第一步是发现问题。SAM3 的 point prompt 很强，只要点落在目标上，SAM 往往能切出合理 foreground；但是 text prompt 不稳定，尤其是属性词、动作词、组合词，比如 `white girl`、`execute` 这类 prompt 很难稳定落到一个清晰实例上。

第二步是引入 Matcher。既然不能只靠 text，那就用 reference 图里的人工 mask 来定义目标特征。比如我在一张眼底图里有人工病灶 mask，那么我就可以用这个 mask 提取 DINO feature prototype，再去另一张眼底图里找相似位置。

第三步是跑通 Stage2 和 Stage3。Stage1 找到 top-k matched points；Stage2 把这些点聚类、采样，组成多组 point prompts；每组 points 送入 SAM 得到一个候选 mask；Stage3 再决定哪些 mask 保留。

第四步是加入 feedback。正点和负点不只是给 SAM 用，也可以反馈给 Matcher：正点增强目标特征，负点提供背景信息，用来抑制错误区域。

第五步是形成 learning 主线。以前 Stage3 靠手写 threshold，现在我们把每个 proposal 变成一行训练样本，学习 keep/reject 策略。

这一页我要传达的是：老师提出的问题已经被拆成了一个可操作、可记录、可学习的 pipeline。

---

## 3. 背景与问题定义

这里我会回答为什么不能只依赖 SAM point 或 text。

SAM point prompt 的问题不是能力不够，而是“点从哪里来”。如果用户已经知道目标在哪里，那点一下就能切；但在真实任务里，我们希望系统自动找出 query 图里可能的目标点。

Text prompt 的问题是语义到视觉边界之间不稳定。比如 `white girl` 涉及属性和人这个类别，`execute` 是动作词，不一定对应明确边界。所以 text prompt 可以作为粗检索信号，但不适合作为这个阶段唯一可靠的控制信号。

因此 Matcher 的位置就很清楚：它不是替代 SAM，而是负责在 SAM 前面自动找点。SAM 负责从点生成 mask，learning 负责从多个 mask 里选出更可信的 proposal。

老师截图里提到的几个问题，对应关系是：

- point 是否比 text 更可靠：我们用 point-driven pipeline 继续推进。
- text prompt 失败后怎么办：用 reference image/mask/box 替代纯文本描述。
- Stage1 大图如何处理：用 sliding-window/tile matching，再把点恢复到原图坐标。
- Stage2+3 如何闭环：点聚类、采样、SAM proposals、instance-level filtering。

---

## 4. 会议问题的正面回应

这一节是直接回应老师会后布置的两个方向。

### 4.1 方向一：feedback 如何纳入 Matcher

我现在实现的是两层 feedback。

第一层是点级 feedback。  
如果用户点了 positive point，这个点可以被理解为“这里更像目标”，它应该增强 target prototype 或提高附近相似度。  
如果用户点了 negative point，这个点不只是 SAM 的背景点，也可以被看作背景特征，后续在 Matcher 相似度里要压低类似区域。

第二层是 proposal 级 feedback。  
用户 accept 或 reject 一个 proposal 后，这个选择会进入 proposal casebook，变成 keep/reject 标签。后续学习器就不是凭空学习，而是学习“哪些 proposal 特征更应该被保留”。

所以我对方向一的回答是：feedback 可以从 prompt 层进入 Matcher，也可以从 proposal 层进入 learning 数据。

### 4.2 方向二：learning 如何纳入 Matcher

当前阶段我没有一开始就上复杂神经网络，而是先做可解释 learning。

学习目标是 proposal 是否保留。输入特征包括：

- `feedback_learning_score`
- `filter_score`
- `sam_score`
- `cluster_coverage`
- `prompt_coverage`
- `mask_area`
- `bbox_area`
- small mask / small bbox 等特征

目前结果显示，简单的 scalar threshold 和 decision stump 已经能显著超过手写 Stage3。  
所以方向二的阶段性结论是：threshold、prompt 组合、proposal keep/reject 本身都可以看成学习问题，而不是只靠人工调参。

---

## 5. Matcher 三阶段闭环与实验设计

这一页主要讲清楚 Stage1、Stage2、Stage3 分别在做什么。

### 5.1 Stage1：Reference 到 Query 的 matching

Stage1 的输入是一张 support/reference 图，以及这张图上的目标标注。  
在医学实验里，这个标注是人工 mask，而不是随便画的大框。

具体过程是：

1. 从 reference mask 内提取 DINO feature。
2. 对这些 feature 做平均或聚合，得到目标 prototype。
3. 在 query 图的 feature map 上逐位置计算 cosine similarity。
4. 相似度最高的位置取出来，作为 top-k matched points。

所以 Stage1 回答的是：“query 图里哪里最像 reference mask 中的目标？”

### 5.2 Stage2：点采样与聚类

Stage2 不会把 100 个点一次性全塞给 SAM。  
原因是 SAM 的 point prompt 数量太多时不一定稳定，而且多个目标可能分布在不同区域。

因此我把 top-k points 做聚类或组合：

- 可以取 top1 单点。
- 可以取 top2 两点组合。
- 可以从 top5 里抽 3 个点。
- 也可以按 DBSCAN cluster 形成不同 prompt groups。

每一个 group 送入 SAM，就得到一个候选 mask。  
所以每个 case 会产生多个 proposals，而不是只产生一个结果。

### 5.3 Stage3：实例级筛选

Stage3 做的是 instance-level filtering。  
也就是说，如果 Stage2 给了 12 个候选 mask，Stage3 要判断哪些留下，哪些丢弃。

原始手写 Stage3 通常根据一些规则判断，比如 mask 面积、bbox 面积、cluster coverage、SAM score 等。  
但是实验发现，手写规则经常误杀好 proposal，尤其是在医学微小病灶和跨领域场景里。

这就是为什么后面要把 Stage3 改成 learning。

### 5.4 为什么 Stage2+3 要分开

Stage2 和 Stage3 分开非常重要。

Stage2 负责尽可能提高候选召回。它可以多生成一些 proposal，只要候选池里有好 mask，后续就有机会选出来。

Stage3 负责提高精度。它不应该过早把好 proposal 丢掉。

如果把二者混在一起，就看不清失败原因。现在分开后可以判断：

- 如果 oracle proposal 也不好，说明 Stage1/Stage2 候选生成失败。
- 如果 oracle proposal 很好，但 Stage3 没选中，说明问题在 proposal filtering。

这也是后续论文可以写得比较清楚的地方：我们不是只报一个最终分割结果，而是把失败位置拆开分析。

---

## 6. P0 到 P4 推进结果

这一节是我把之前的计划真正跑成实验的部分。

### P0：review package

P0 主要是为了人工检查 proposal。  
我生成了 review gallery，每个 proposal 都能显示并放大。  
这一步的意义是：后续如果要把伪标签换成人工标签，需要一个可操作入口。

### P1：IDRID / MonuSeg 分域阈值

P1 发现 IDRID 和 MonuSeg 的最优特征不同。  
IDRID 更依赖 `feedback_learning_score`，MonuSeg 更依赖 `cluster_coverage`。

这说明不同医学领域不能简单共用一个 threshold，分域策略是必要的。

### P2：feedback 迭代模拟

P2 做的是模拟：如果每个 case 给少量 feedback，proposal policy 是否能改善。  
初步结果说明少量反馈确实可以把策略往正确方向推，但这里仍然需要注意：目前很多标签还是基于 GT/Dice 的伪标签，后续必须用人工 review 标签进一步验证。

### P3：50 例扩展

P3 把数据从 20 例扩展到了 50 例：

- IDRID 30 例
- MonuSeg 20 例
- 每例 12 个 proposals
- 总共 600 proposals

这个规模还不算大 benchmark，但已经足够验证这个主线不是只在一两个 case 上成立。

### P4：模型升级对比

P4 对比了 scalar threshold、decision stump 和 logistic regression。  
当前结果显示，可解释模型更稳定，复杂模型在小数据伪标签阶段不一定更好。

所以现阶段我会优先保留可解释 learning，而不是马上上复杂模型。

---

## 7. Feedback-Learned Matcher 医学主线

这里是整份汇报最像论文主线的一页。

我现在把系统定义成：

> Reference mask 定义目标，Matcher 自动找点，SAM 生成候选，feedback 和 learning 筛选 proposal。

20 例实验里，每个 case 生成 12 个 proposals，总共 240 个 proposal 标签。  
结果显示：

- Stage3 F1 比较低。
- Learned proposal policy 的 F1 明显更高。
- 说明候选池中有有效信息，但手写规则没有充分利用。

这张论文主表的讲法是：  
我们先不说模型已经达到最终临床可用，而是说在 proposal selection 这个子问题上，learning 明显优于 hand-crafted Stage3。

这里的科学结论是：

1. SAM proposals 的候选池里确实经常存在好 mask。
2. 主要瓶颈不一定是 SAM 生成不了，而是 Stage3 选不对。
3. 因此老师提出的 learning 方向是合理的，而且已经有定量证据。

---

## 8. 分域训练：从手写 Stage3 到可学习 proposal policy

这一页回应一个更进一步的问题：如果在单独某个医学领域训练一次，能不能迁移到同领域其它 case？

### 8.1 学习对象是什么

学习的不是端到端分割网络，而是 proposal keep/reject policy。

每一个训练样本是一行 SAM proposal：

- 输入：proposal 的各种特征。
- 标签：这个 proposal 是否应该保留。
- 输出：keep 或 reject。

所以模型不是凭空生成病灶，而是在候选池里选更可信的 mask。

这一点很重要，因为它决定了我们现在的突破边界：如果候选池里没有好 mask，learning 没法凭空造出来；如果候选池里有好 mask，learning 就有机会把它选出来。

### 8.2 同域训练，同域迁移

IDRID 和 MonuSeg 分别训练各自的策略。  
这样做是因为眼底病灶和细胞核在形态、大小、纹理、mask 面积分布上都不一样。

分域迁移结果显示，IDRID 和 MonuSeg 的 learned F1 都明显超过 Stage3。  
这说明“单独某一个领域训练一次，再用于同领域 case”这个方向是有希望的。

但我会谨慎表述：现在还不能说“所有病灶都能被标注出来”，因为上游候选池和 proposal 排序仍然会限制最终 Dice。

---

## 9. 50 例 case-level 5-fold：为什么是这些数字

这一页是为了回答老师可能会问的“你到底有没有用所有 case 训练测试”。

当前总共有 50 个 case：

- 30 个 IDRID
- 20 个 MonuSeg

每个 case 固定保留 12 个 proposals，所以总数是：

> 50 × 12 = 600 proposals

做 5-fold 的原因是：  
每折测试 10 个 case，5 折刚好覆盖全部 50 个 case。

每折训练测试比例是：

- 40 个 case 训练
- 10 个 case 测试

每个 case 的 12 个 proposals 不会拆开。  
也就是说，一个 case 要么全部在训练集，要么全部在测试集。

为什么不能随机拆 proposal？  
因为同一个 case 的 12 个 proposals 来自同一张 query 图、同一组 matched points、同一个 GT mask。如果把它们随机拆到 train/test，模型在训练时已经见过同一张图的其它候选，测试结果会偏乐观。

所以按 case 切分更接近真实使用场景：训练时见过一些病例，测试时面对整张新的病例图。

### 9.1 5-fold 定量结果怎么讲

50 例 5-fold 的结果是：

- Stage3 F1：0.165517
- Overall learned F1：0.489796
- Domain learned F1：0.611111

这说明 proposal keep/reject 层面，domain learned 明显优于手写 Stage3。

Dice 方面：

- Stage3 Dice：0.199561
- Domain learned Dice：0.196110
- Oracle Dice：0.323295

这里要特别解释：F1 提升明显，但 Dice 没有同步大幅提升，说明 learning 学会了更多正确 keep/reject 信号，但还没有完全学会选到每个 case 里 Dice 最高的 proposal。

因此下一步不是只继续追 F1，而是要优化 proposal 排序、候选质量和 domain-specific threshold。

---

## 10. 新增定性分析图：训练结果到底选了什么

这里是网页里新增的 `case_split_50_qualitative_grid.png`。

我会重点讲这张图，因为它比表格更直观。

这张图的来源是 `case_split_50_predictions.csv`，也就是 50 例 5-fold 的测试折预测，不是训练集可视化。

每一行是一个 held-out test case。  
每一列分别是：

1. 人工 GT，只看真实标注。
2. Stage3 手写规则在这个 case 里选到的最好 proposal。
3. Domain learned policy 在这个 case 里选到的最好 proposal。
4. Oracle proposal，也就是这个 case 候选池中 Dice 最高的 proposal。

颜色含义：

- 蓝色轮廓是人工 GT。
- 粉色轮廓是被选中的 SAM proposal。

### 10.1 第一行：MonuSeg monuseg_01

第一行是一个成功例子。  
Stage3 没有选中任何 proposal，但是 domain learned 选中了 group_009，Dice 是 0.613，而且 oracle 也是 group_009。

这说明在这个 case 上，候选池里有好 proposal，Stage3 没选中，而 learned policy 成功把它选出来了。

### 10.2 第二行：IDRID idrid_21

第二行也是正面例子。  
Stage3 选中了 group_008，Dice 约 0.374；domain learned 额外选中了 group_006，Dice 约 0.379，并且这个也是 oracle。

这里说明 learning 不是简单替代 Stage3，而是能补充 Stage3 漏掉的高质量候选。

### 10.3 第三行：MonuSeg monuseg_04

第三行是反面例子。  
Oracle 是 group_008，Dice 0.629，但 learned policy 选中的是 group_003，Dice 只有 0.088，而且 label 是 0。

这说明当前 learning 仍然会被一些大面积或弱 false positive 干扰。  
这个例子支持下一步要做更强的排序学习，不能只靠单阈值 keep/reject。

### 10.4 第四行：IDRID idrid_11

第四行也是反面例子。  
Stage3 已经选到了 group_002，Dice 0.925；oracle 是 group_005，Dice 0.926，两者都很好。  
但是 learned policy 这次过于保守，没有选中 proposal。

这个例子说明 learned policy 的 recall 还要继续提高，尤其是在 IDRID 这种微小病灶和大尺度眼底图里，阈值不能太硬。

### 10.5 这张图的结论

这张图要表达一个平衡结论：

learning 确实能显著改善 Stage3 的 keep/reject，尤其是在 Stage3 完全漏掉时；但 learning 还没有稳定达到 oracle，上游候选召回和下游排序都还有优化空间。

所以我们可以说这条主线成立，但不能夸大成“已经解决医学分割”。

---

## 11. IDRID idrid_11：点、图和 proposal 细节

这一节用 IDRID 讲清楚 reference mask 到 query matching 的具体过程。

输入里真正重要的是：

- support image：IDRiD_01
- support mask：IDRiD_01 的人工病灶 mask
- query image：IDRiD_12
- query mask：IDRiD_12 的人工病灶 mask

这里我要特别解释 reference box 的意义：  
真正用于 matching 的是人工 mask 内的 DINO feature，不是单纯拿 box 去 match。reference box 只是 mask 的外接范围，用于记录和可视化定位。

图里的 top matched points 都画在 query 图 IDRiD_12 上。  
这些点是由 reference mask prototype 和 query feature map 相似度计算得到的。

### 11.1 分层证据图怎么讲

这张图拆成六块：

- A：reference image + manual mask，说明目标来自人工标注。
- B：query GT，只看目标真实位置。
- C：Stage1 top-k points，只看 Matcher 找到的点。
- D：SAM proposal，只看某个 group 生成的 mask。
- E：GT/proposal 边界局部对比，不画 bbox，避免混淆。
- F：proposal 指标，包括 Dice、IoU、SAM score、mask area、stage3_keep、learned keep。

这样拆开的意义是避免颜色叠在一起看不清，也能回答“点在哪里、mask 在哪里、指标是多少”。

### 11.2 这个 case 的关键现象

IDRID idrid_11 的 best proposal group_005 Dice 达到 0.925738，IoU 0.861743，但是 stage3_keep = 0。  
这说明失败不在 Stage1 找不到点，也不在 SAM 生成不了 mask，而在 Stage3 手写规则误杀了好候选。

因此这个 case 是支持方向二的核心证据：把 keep/reject 从手写规则改成可学习策略是必要的。

---

## 12. MonuSeg monuseg_04：分域问题的直接证据

MonuSeg 这一节用来说明为什么分域策略必要。

输入是：

- support image：01_1
- support mask：01_1 的人工细胞核 mask
- query image：01_5
- query mask：01_5 的人工细胞核 mask

MonuSeg 的 reference box 覆盖整张 256×256 图，是因为细胞核 mask 分布在整图范围内。  
但这里也不是用空泛大框做 prototype，prototype 仍然来自人工 mask。

### 12.1 这个 case 的关键现象

MonuSeg monuseg_04 里，Stage1 已经找到了很多相似点，Stage2 也生成了可用 proposals。  
其中 best proposal group_008 Dice 是 0.629355，IoU 是 0.459167。

但是 stage3_keep = 0。  
这说明手写 Stage3 对 MonuSeg 不适配，不能直接复用 IDRID 或自然图中的阈值。

所以这个 case 支持 P1 的结论：MonuSeg 需要单独阈值或 domain-specific strategy。

### 12.2 和 5-fold 定性图的联系

不过我要注意，这个 case 在新的 5-fold learned policy 里仍然暴露问题：  
oracle 是 group_008，但 learned policy 选了弱 false positive。  

这说明分域学习虽然提升了总体 F1，但还需要进一步做排序学习、阈值校准或更多人工标签。

---

## 13. 可复现实验清单

这一页用来回答“你的结果能不能复现”。

主要命令包括：

```bash
python -B code\benchmarks\build_matcher_p0_p4_progress.py --expanded-root runs\research_direction\matcher_medical_expanded --out-dir runs\research_direction\matcher_p0_p4_progress
python -B code\benchmarks\check_matcher_p0_p4_progress.py
python -B code\benchmarks\train_domain_proposal_policy.py
python -B code\benchmarks\check_domain_proposal_policy.py
python -B code\benchmarks\train_case_split_proposal_policy.py
python -B code\benchmarks\check_case_split_proposal_policy.py
python -B code\benchmarks\build_report_77_detail_figures.py
python -B code\benchmarks\build_case_split_qualitative_figures.py
```

关键输出文件包括：

- `runs/research_direction/matcher_p0_p4_progress/p0_p4_progress.json`
- `runs/research_direction/matcher_p0_p4_progress/point_details.csv`
- `runs/research_direction/feedback_learned_matcher_mainline/domain_policy_transfer/domain_policy_transfer_summary.csv`
- `runs/research_direction/feedback_learned_matcher_mainline/case_split_50/case_split_50_summary.csv`
- `runs/research_direction/feedback_learned_matcher_mainline/case_split_50/case_split_50_predictions.csv`
- `docs/reports/7.7/assets/case_split_50_qualitative_grid.png`

这里我会强调：网页里的表格、点坐标、case 图和指标都不是手写出来的，而是从这些输出文件汇总来的。

---

## 14. 阶段结论与后续方向

最后总结成三点。

### 14.1 方向一结论

Feedback 可以进入 Matcher。  
正负点可以影响 prototype 和相似度图；accept/reject 可以形成 proposal-level learning 数据。  
这说明 feedback 不只是交互界面上的操作，而是可以变成模型策略的一部分。

### 14.2 方向二结论

Learning 可以进入 Matcher。  
目前最明确的证据是 proposal keep/reject：  
50 例 5-fold 下，Stage3 F1 是 0.165517，domain learned F1 是 0.611111。  
这说明 learning 明显优于手写 Stage3。

但是 Dice 还没有明显超过 Stage3，也低于 oracle。  
所以当前结论应该是：learning 解决了部分 selection 问题，但还没有完全解决最终 mask 质量问题。

### 14.3 下一步重点

下一步我建议沿着三个方向继续推进：

第一，人工标签确认。  
把 review gallery 里的 accept/reject 从 Dice 伪标签换成人工标签，避免学习器只是在学习阈值规则。

第二，proposal 排序学习。  
现在很多问题不是有没有保留，而是有没有把最好的 proposal 排到前面。下一步可以从 keep/reject 变成 ranking，例如学习每个 case 内的 best proposal。

第三，分域策略和轻量模型。  
IDRID 和 MonuSeg 需要不同策略。可以继续做 domain-specific threshold，也可以尝试轻量 ranking model，但仍然保持可解释。

最后，我会把这条主线概括为：

> Text prompt 不稳定时，用 reference mask 定义目标；Matcher 自动找点；SAM 生成候选；feedback 和 learning 负责把候选选对。当前证据说明这条路可行，但真正投稿前还需要人工标签、更强排序和更大规模验证。

---

## 15. 老师可能追问的问题与回答

### Q1：你这里的 reference box 到底有什么意义？是不是用 box 在 match？

不是。真正用于 matching 的是 reference mask 内的 DINO feature。  
reference box 只是人工 mask 的外接范围，用于记录目标大致位置和局部可视化，不是替代 mask 的训练信号。

### Q2：为什么候选区域既有点又有框？

点是 SAM 的 prompt 输入，来自 Matcher 的 top-k matched points 或它们的组合。  
框不是当前主要 prompt，主要是 proposal 生成后用于记录 mask 外接范围、计算 bbox area、做特征统计和可视化。  
所以点和框不是同一个阶段的东西：点用于生成 proposal，框用于描述 proposal。

### Q3：为什么 50 个 case 是 600 个 proposals？

因为每个 case 固定保留 12 个 SAM proposals。  
所以 50 × 12 = 600。

### Q4：为什么 5-fold 是 40 个 case 训练、10 个 case 测试？

因为总共有 50 个 case。  
5-fold 时每折测试 10 个 case，剩下 40 个 case 训练。  
5 折后每个 case 恰好测试一次。

### Q5：为什么不能随机拆 600 个 proposals？

因为同一个 case 的 12 个 proposals 来自同一张 query 图。  
如果随机拆 proposal，训练集和测试集会共享同一张图的信息，结果会偏乐观。  
所以必须按 case 切分。

### Q6：现在能不能说单独训练一个领域，就能标出这个领域所有病灶？

还不能这么说。  
现在可以说：在候选池已经包含好 proposal 的情况下，同域学习能明显改善 proposal keep/reject。  
但如果候选池没覆盖病灶，或者排序没选到最优 proposal，最终分割仍然会失败。

### Q7：为什么 F1 提升了，但 Dice 没有明显提升？

F1 衡量的是 proposal keep/reject 是否选对。  
Dice 衡量的是最终选中的 mask 和 GT 的重叠。  
学习器可能减少了很多错误 keep/reject，因此 F1 提升；但如果它没有选到每个 case 里 Dice 最高的 proposal，Dice 不一定同步提升。  
这说明下一步要从二分类 keep/reject 进一步做 proposal ranking。

### Q8：这条工作怎么写成论文？

可以写成一个 reference-guided, feedback-learned proposal selection 框架。  
核心卖点不是提出一个新的 SAM，而是解决 SAM prompt 和 Matcher 后处理中的两个问题：

1. Text prompt 不稳时，如何用 reference mask 自动找点。
2. 多个 SAM proposals 出来后，如何用 feedback 和 learning 替代手写 Stage3。

论文实验可以分成：

- 自然图 prompt failure 和 point advantage。
- Matcher 三阶段可解释闭环。
- 医学 20/50 例 proposal learning。
- 5-fold case-level 验证。
- 定性案例分析和失败分析。

---

## 16. 30 秒极简收尾

最后我想用一句话收尾：

这次工作把老师提出的 feedback 和 learning 两个方向都接到了 Matcher 里。现在我们已经能从 reference mask 自动找点、生成 SAM proposals，并用可解释 learning 改善 proposal keep/reject。50 例 5-fold 结果显示 domain learned F1 明显超过手写 Stage3，但 Dice 距 oracle 仍有差距。所以下一步重点是人工标签、proposal ranking 和分域策略，把这个闭环从“验证可行”推进到“可以写论文主实验”。
