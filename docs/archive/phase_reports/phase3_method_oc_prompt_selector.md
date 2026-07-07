# Phase3 Method: OCSAM Prompt Selector

## 1. Method Summary

The Phase3 method is:

```text
OCSAM Prompt Selector:
domain-aware medical prompt generation + prompt reliability scoring + uncertainty-aware reranking.
```

It is designed for medical promptable segmentation with SAM-family models.

The method does not claim to train a new foundation model. Instead, it improves how prompts are generated and selected before calling SAM, MedSAM, or SAM2.

## 2. Motivation From Prior Work

The design follows several lightweight components commonly used in recent medical SAM papers:

- automatic prompt generation;
- dynamic prompting from coarse candidates;
- prompt perturbation robustness;
- uncertainty-guided multi-prompt adaptation;
- mask/prompt quality reranking;
- lightweight adapter-style scoring instead of full backbone training.

We implement the low-cost version:

```text
multiple candidate prompts -> multiple masks -> reliability/uncertainty score -> selected prompt/mask
```

## 3. Pipeline

### Step 1: Candidate Prompt Generation

For each image, generate several prompts:

- `auto_box`
- `auto_medical_box_v1`
- `auto_medical_point_v1`

IDRID uses lesion-oriented red/bright candidates.

MonuSeg uses dark/purple nuclei candidates.

### Step 2: SAM-family Inference

Each prompt is evaluated by available SAM-family backends:

- SAM ViT-B
- MedSAM ViT-B
- SAM2 Tiny

Unsupported prompt types are recorded explicitly. For example, MedSAM does not support point prompts in this benchmark.

### Step 3: Reliability Features

For each prompt/mask candidate, compute:

- model identity;
- dataset/domain;
- prompt type;
- SAM/SAM2 raw score when available;
- predicted mask area ratio;
- small/large mask flags;
- mask agreement with other auto-family candidates;
- prompt uncertainty = `1 - mask agreement`;
- mask area deviation from median candidate;
- mask area deviation from `auto_box`.

### Step 4: Prompt Selector

The selector predicts prompt reliability and chooses the best candidate.

Current implemented variants:

| selector | description |
| --- | --- |
| `data_aware_selector_v1` | hand-written domain/model rule |
| `prompt_selector_v2` | leave-one-case-out ridge reliability selector |
| `prompt_selector_v3` | v2 + prompt-induced uncertainty / mask stability |

### Step 5: Output

The method outputs:

- selected prompt;
- selected mask;
- confidence/reliability estimate;
- selection metadata;
- failure tags for analysis.

## 4. Current Evidence

On the current 40-case benchmark:

| method | mean Dice |
| --- | ---: |
| fixed `auto_box` | 0.3118 |
| `data_aware_selector_v1` | 0.3234 |
| `prompt_selector_v2` | 0.3263 |
| `prompt_selector_v3` | 0.3279 |
| oracle auto-family upper bound | 0.3584 |

Interpretation:

- The selector route is valid.
- Uncertainty/mask stability provides a small but positive gain.
- The oracle gap remains large enough to justify further Phase3 research.

## 5. Ablation Design

The paper should report:

1. Fixed prompt baselines:
   - GT box
   - perturbed box
   - centroid point
   - grid point
   - auto box
2. Auto medical prompt:
   - `auto_medical_box_v1`
   - `auto_medical_point_v1`
3. Selector ablation:
   - no selector
   - domain-aware rule
   - reliability scorer
   - reliability + uncertainty
4. Oracle:
   - auto-family oracle
   - all-prompt oracle

The oracle is not a method. It is only the upper bound.

## 6. Claim

A safe paper claim is:

```text
In medical promptable segmentation, prompt quality and prompt type are domain-dependent.
We show that structure-aware automatic prompt generation with uncertainty-aware prompt selection improves the reliability of SAM-family models without retraining the foundation model.
```

## 7. Next Work

Next milestone:

```text
prompt_selector_v4 = perturbation-based uncertainty + 100-case validation.
```

Recommended additions:

- jitter each auto box several times;
- perturb positive points slightly;
- collect mask variance;
- compute stability score;
- train reliability selector with stability features;
- expand from 40 cases to 100 cases;
- add SAM ViT-H, Matcher, and SAM3 only after selector remains useful.
