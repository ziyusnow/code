# Oracle vs Uniform Random Action Reward

## Definition

- Each action reward is the median of its three branch repeats.
- `R_oracle` is the largest of the four action median rewards.
- `R_random` is the uniform average of the four action median rewards.
- `Delta R = R_oracle - R_random`.
- `Margin` is the best minus second-best action median reward.
- Clear Oracle requires `Margin > 2.0 x pooled_repeat_std`; otherwise it is Ambiguous Oracle.

## Overall

- Checkpoints: `25`
- Clear Oracle: `3/25`
- Ambiguous Oracle: `22/25`
- Median Delta R: `3.419288`
- Median Margin: `1.120355`
- Median pooled repeat standard deviation: `3.585811`

## By Checkpoint

| Checkpoint | Oracle median | Random median | Delta R median | Relative uplift median | Margin median | Clear | Ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20pct | 212.535314 | 127.709430 | 52.671952 | 37.13% | 34.165741 | 1 | 4 |
| 40pct | 81.335225 | 71.541564 | 9.793661 | 14.70% | 1.182064 | 0 | 5 |
| 60pct | 10.278929 | 9.190070 | 1.802738 | 28.56% | 0.828442 | 1 | 4 |
| 80pct | 5.293195 | 3.821415 | 1.423869 | 30.93% | 0.282225 | 0 | 5 |
| 90pct | 2.436869 | 1.880428 | 0.708043 | 35.01% | 0.165158 | 1 | 4 |

## Interpretation Boundary

`R_random` is a uniform-action proxy computed from four action medians, not a new Monte Carlo estimate.
With only three repeats per action, Clear/Ambiguous is a descriptive noise screen rather than a significance test.
The hindsight Oracle is an upper-bound diagnostic and is not an online deployable selector.
