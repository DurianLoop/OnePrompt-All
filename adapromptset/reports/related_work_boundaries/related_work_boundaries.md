# Related Work Boundaries

Status: `PASS_RELATED_WORK_BOUNDARIES`

| Work | Reference-guided | Negative evidence | Learned prompt policy | Grouping | Budgeted complete action | AdaPromptSet boundary |
|---|---:|---|---|---|---:|---|
| GF-SAM | yes | `background_reference_evidence` | `none` | `graph_rule` | no | `deterministic_graph_baseline` |
| SynPo | yes | `explicit_rule_negatives` | `none` | `fixed_kmeans` | no | `fixed_negative_prompt_rule` |
| AlignSAM | no | `learned_label_module` | `sequential_rl` | `patch_center_actions` | no | `one_shot_offline_set_level` |
| Point Prompt Defender | no | `adversarial_prompt_optimization` | `adversarial_dqn` | `dual_space_graph_environment` | no | `not_reference_constructed_action_bank` |

Claim guardrails:

- Do not claim first-to-learn prompting.
- Do not describe GF-SAM background-reference evidence as SynPo-style explicit negative point prompting.
- Treat AlignSAM and Point Prompt Defender as learned/RL prompt-optimization boundaries rather than identical tasks.
