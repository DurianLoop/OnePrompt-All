# Stage 2 Cluster Prompt Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal reproducible Stage 2 sampler that turns feedback-reranked Matcher points into clustered multi-point prompt groups.

**Architecture:** The sampler reads `feedback_points.json` from P0-2, filters points by threshold/top ratio, clusters them with a dependency-light DBSCAN-style implementation, samples deterministic prompt groups per cluster, and writes CSV/JSON/overlay/summary artifacts. It does not call SAM3 yet; this step validates whether Stage 2 sampling and K estimation are plausible before adding mask generation.

**Tech Stack:** Python standard library, `numpy`, `Pillow`, local `code/benchmarks` scripts.

---

### Task 1: Static Check First

**Files:**
- Create: `code/benchmarks/check_stage2_sampler.py`
- Expected future target: `code/benchmarks/sample_stage2_prompts.py`

- [ ] **Step 1: Write the failing static check**

Create `code/benchmarks/check_stage2_sampler.py` with required-token checks for the future sampler:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "benchmarks" / "sample_stage2_prompts.py"


REQUIRED_TOKENS = [
    "--feedback-dir",
    "--top-ratio",
    "--cluster-method",
    "--eps",
    "--min-samples",
    "--sample-sizes",
    "--samples-per-cluster",
    "prompt_groups.csv",
    "prompt_groups.json",
    "cluster_overlay.png",
    "sample_overlay.png",
    "summary.md",
    "load_feedback_points",
    "select_candidate_points",
    "cluster_points",
    "sample_prompt_groups",
    "save_stage2_outputs",
]


def main() -> None:
    if not SCRIPT.exists():
        raise SystemExit(f"Missing expected sampler script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        raise SystemExit("Missing required Stage 2 sampler tokens: " + ", ".join(missing))
    print("Stage 2 sampler static checks passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_sampler.py
```

Expected: FAIL with `Missing expected sampler script`.

### Task 2: Implement Minimal Sampler

**Files:**
- Create: `code/benchmarks/sample_stage2_prompts.py`

- [ ] **Step 1: Implement data loading and candidate selection**

The script must read `feedback_points.json`, keep selected points first when `--selected-only` is enabled, otherwise keep the top `ceil(top_ratio * top_k)` points.

- [ ] **Step 2: Implement dependency-light clustering**

Implement:

```text
cluster-method = none | dbscan
```

For `dbscan`, use Euclidean distance in original image coordinates, `eps`, and `min_samples`. Noise points get `cluster_id = -1`.

- [ ] **Step 3: Implement deterministic prompt sampling**

For each non-noise cluster, generate up to `samples_per_cluster` prompt groups for each requested sample size. Groups are deterministic:

1. first point is the highest-ranked point in the cluster;
2. remaining points are spread across the sorted cluster list;
3. if cluster is smaller than sample size, skip that group.

- [ ] **Step 4: Save artifacts**

Write:

```text
prompt_groups.csv
prompt_groups.json
cluster_overlay.png
sample_overlay.png
summary.md
```

The summary must include candidate count, cluster count, noise count, estimated K, prompt group count, and commands used.

- [ ] **Step 5: Run checks and smoke command**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_sampler.py
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\stage1_feedback\elephant_pair --selected-only --cluster-method dbscan --eps 55 --min-samples 3 --sample-sizes 1,3,5 --samples-per-cluster 4 --out-dir runs\research_direction\stage2_sampling\elephant_pair
```

Expected: static check passes and sampler writes the Stage 2 output directory.

### Task 3: Chinese Report and Plan Update

**Files:**
- Create: `docs/matcher_stage2_sampling_report.md`
- Modify: `docs/research_direction_priority_plan.md`

- [ ] **Step 1: Write report**

Report sections:

1. 实验目的；
2. 输入和可复现命令；
3. 聚类与采样规则；
4. 结果表；
5. 对老师“sample N% points, repeat m times”问题的回答；
6. 下一步如何接 SAM3/SAM masks。

- [ ] **Step 2: Update total plan**

Under P0-3, add the completed smoke result, output directory, and interpretation.

- [ ] **Step 3: Final verification**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_sampler.py
Test-Path docs\matcher_stage2_sampling_report.md
Test-Path runs\research_direction\stage2_sampling\elephant_pair\prompt_groups.csv
Test-Path runs\research_direction\stage2_sampling\elephant_pair\cluster_overlay.png
```
