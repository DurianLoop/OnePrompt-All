# Fixed Rule / Oracle Pivot Analysis

Recommended claim: `oracle_analysis_only`

| Item | Value |
|---|---:|
| Episodes | 10 |
| Actions | 12 |
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | 0.371238 |
| Mean oracle Dice | 0.376250 |
| Mean regret to oracle | 0.005011 |
| AJI available | True |
| Mean best fixed AJI | 0.22834095079987268 |
| Mean oracle AJI | 0.23216115704701398 |
| Mean AJI regret to oracle | 0.0038202062471413 |
| Median regret to oracle | 0.000319 |
| Max regret to oracle | 0.018556 |
| Near-oracle episode fraction | 0.600 |
| Top oracle action fraction | 0.500 |

Interpretation:

- The current MoNuSAC evidence does not justify policy training.
- Oracle and best fixed are close, so the next valid MoNuSAC output is an action-oracle/representation analysis, not an adaptive-policy claim.
- A new-domain pilot remains allowed only if predeclared and kept separate from observed MoNuSAC/IDRID test behavior.
