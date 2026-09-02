# 代表性设备参数下的故障时刻扫描

| 参数 | 取值 |
|---|---:|
| ESS 容量 / 最大功率 | 15 MWh / 3 MW |
| G2 最大出力 / 爬坡 | 20 MW / 3 MW/h |
| G1 故障后最大出力 | 0 MW |
| 故障持续时间 | 4 h |
| 故障起点 | 5, 8, 11, 14, 17, 20 h |
| P1/P2 算法 | RA-LSHADE (`NP=600->4, K=500`) |
| 重试种子 | 20260826, 20260827, 20260828, 20260829, 20260830 |

| 故障区间 | No reserve 状态 | No reserve 非重要失负荷 | No reserve 重要负荷缺额 | Dynamic 状态 | Dynamic 非重要失负荷 | Dynamic 重要负荷缺额 |
|---|---|---:|---:|---|---:|---:|
| 5–8 h | RA_LSHADE_FEASIBLE | 1.734962 MWh (13.992%) | 0.000000 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
| 8–11 h | RA_LSHADE_FEASIBLE | 1.391804 MWh (12.427%) | 0.000000 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
| 11–14 h | INFEASIBLE_CRITICAL_LOAD | 3.000000 MWh (28.037%) | 0.592948 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
| 14–17 h | RA_LSHADE_FEASIBLE | 1.797568 MWh (17.798%) | 0.000000 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
| 17–20 h | RA_LSHADE_FEASIBLE | 2.299581 MWh (26.432%) | 0.000000 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
| 20–23 h | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh | RA_LSHADE_FEASIBLE | 0.000000 MWh (0.000%) | 0.000000 MWh |
