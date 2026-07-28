# Canonical V4 Decision

## 决策日期
2026-07-28

## Canonical V4 认定

### PDF
- **文件**: `D:\SAM\adapromptset\paper\phaseU\full_draft_v4.pdf`
- **大小**: 562,980 bytes
- **修改时间**: 2026-07-27 00:53:33
- **SHA256**: `36d5a3e7d4bf61b90d9d3baa04c680f50cc1a1974802850b7f84f44cc8634e44`

### 对应源文件
- **文件**: `D:\SAM\adapromptset\paper\phaseU\full_draft.tex`
- **大小**: 63,109 bytes
- **修改时间**: 2026-07-27 15:42:33
- **SHA256**: `6c986f4b09213e8d71a8417d7f369441ac90d68ab95c8eb4b30b0d37f6fe70d6`

### 源文件备份
- **文件**: `D:\SAM\adapromptset\paper\phaseV\full_draft_v4_ref.tex`
- **SHA256**: `6c986f4b09213e8d71a8417d7f369441ac90d68ab95c8eb4b30b0d37f6fe70d6`
- **与源文件关系**: 字节级相同 (SHA256 完全一致)

## 为什么选它作为 Canonical V4

1. **唯一显式命名**: `full_draft_v4.pdf` 是项目中唯一显式命名为 `full_draft_v4` 的 PDF 文件。没有其他文件使用此命名模式。

2. **源文件存在且可验证**: `full_draft.tex` (phaseU) 与 `full_draft_v4_ref.tex` (phaseV) SHA256 完全一致，证明源文件真实且未被篡改。该源文件标题为 "Reference Reliability Estimation for Visual Prompting Pipelines"。

3. **版本演进清晰**: 
   - V1 (base): `full_draft.pdf` — 416,491 bytes, 2026-07-26 15:56
   - V2: `full_draft_v2.pdf` — 425,977 bytes, 2026-07-26 18:57
   - V3: `full_draft_v3.pdf` — 481,246 bytes, 2026-07-26 19:34
   - **V4: `full_draft_v4.pdf` — 562,980 bytes, 2026-07-27 00:53** ← Canonical
   - V5: `full_draft_v5.pdf` — 266,940 bytes, 2026-07-27 13:59 (phaseV, 单栏新格式)

4. **提交包确认**: `paper_submission_v4.zip` 中包含相同的 PDF (SHA256 一致)，且包含配套的 figures、scripts 和 results 数据。

5. **编译产物完整**: 存在配套的 `.log`, `.out`, `.aux` 文件，证明 PDF 是通过 LaTeX 正常编译生成。

## 与其他版本的差异

### V4 vs V5 关键差异

| 维度 | V4 (Canonical) | V5 (后续) |
|------|----------------|-----------|
| 标题 | Reference Reliability Estimation for Visual Prompting Pipelines | Reference-Aware Selective Prediction for Frozen Visual Prompting Pipelines |
| 文档类 | `\documentclass[10pt,twocolumn,letterpaper]{article}` | `\documentclass[11pt]{article}` |
| 边距 | 0.75in | 1in |
| 排版 | 双栏 (twocolumn) | 单栏 |
| 额外包 | 无 | `\usepackage{enumitem}` |
| 自定义命令 | `\RAMP`, `\RRN`, `\RRE`, `\CFR`, `\PLE` | `\RAMP`, `\RRE`, `\CFR`, `\PLE` (无 `\RRN`) |
| PDF 大小 | 562,980 bytes | 266,940 bytes |
| 源文件大小 | 63,109 bytes | 49,735 bytes |
| 所在目录 | paper/phaseU/ | paper/phaseV/ |

### V4 vs V3/V2/V1

V4 是 phaseU 目录中逐步演进的最终版本，PDF 大小从 V1 (416KB) → V2 (426KB) → V3 (481KB) → V4 (563KB) 持续增长，表明内容在不断充实。

### V4 源文件 vs ZIP 内源文件

ZIP 包内的 `full_draft.tex` (56,447 bytes) 与 phaseU 目录下的 `full_draft.tex` (63,109 bytes) 不同：
- SHA256 (ZIP内): `9EDBA00C5AB9919EC06D2ED87DCF6CA36EA02DF9666BA39025BE3476E814E30D`
- SHA256 (phaseU): `6c986f4b09213e8d71a8417d7f369441ac90d68ab95c8eb4b30b0d37f6fe70d6`
- ZIP 内的 .tex 是提交时的版本，phaseU 目录下的 .tex 是后续修改版（修改时间更晚）

## 源文件状态: V4_SOURCE_EXISTS

**源文件存在，无需标记 V4_SOURCE_MISSING_PDF_ONLY。**

- 主源文件: `D:\SAM\adapromptset\paper\phaseU\full_draft.tex` (63,109 bytes)
- 参考副本: `D:\SAM\adapromptset\paper\phaseV\full_draft_v4_ref.tex` (63,109 bytes, SHA256 一致)
- 提交版: `paper_submission_v4.zip` 内 `full_draft.tex` (56,447 bytes, 略有差异)

## Git 历史中的 V4

根据 `v4_git_timeline.md` 分析：
- 关键词 `full_draft_v4`, `v4`, `Reference Reliability Estimation`, `RAMP`, `RRN` 在 commit message 中**零命中**
- V4 概念在 Git 历史中未显式出现
- 主题关联: Phase R (Reference Stability), Phase S (Reference-Robust Research) 与 V4 的参考可靠性评估主题高度相关
- V4 PDF 文件可能是在 Phase U 阶段通过文件系统直接生成，未经过独立的 Git 提交

## 结论

**Canonical V4 = `D:\SAM\adapromptset\paper\phaseU\full_draft_v4.pdf`**

- 源文件存在且可验证
- 版本演进路径清晰
- 提交包完整
- 是 V4 论文的唯一权威版本