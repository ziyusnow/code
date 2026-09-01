# 面向 G1 降额故障的 ESS 韧性备用与应急能量调度数学模型

## 0. 建模目标与边界

本文研究全船电网保持互联条件下的 **G1 降额故障**。不考虑母线分岛和拓扑重构；故障后 \(G_1\)、\(G_2\)、ESS 和 PV 仍处于同一电网内，共同承担全船负荷。

故障阶段要解决的问题是：

> 在 \(G_1\) 发生降额后，利用 \(G_2\) 的剩余调节能力和故障前预留的 ESS 韧性备用，优先维持重要负荷供电，并在必要时切除非重要负荷，使故障期间的失负荷和运行成本尽可能小。

为控制论文工作量，故障阶段暂不重新优化航速，即沿用故障前正常调度得到的航速轨迹。

求解框架约束：本文正常运行阶段与故障阶段均采用 RA-LSHADE 求解。正常阶段 RA-LSHADE 根据实验策略生成无备用或考虑动态 ESS 韧性备用的预防性调度；故障场景给定后，以对应的正常阶段状态作为初始条件，再次调用 RA-LSHADE 完成校正性应急再调度。两阶段构成统一的 preventive-corrective RA-LSHADE framework。

故障阶段属于离线完整时域场景优化：仿真给定故障起点 \(t_f\)、实际故障持续时间 \(H_f\) 和降额系数 \(\alpha_F\)，P2 在完整故障时间集合 \(\mathcal T_F\) 上一次性求解。\(H_f\) 仅用于 P2 故障场景，不参与 P1 的备用设计。

---

## 1. 时间集合与索引

设故障在时段 \(t_f\) 开始，持续 \(H_f\) 个调度时段，则故障时间集合为

\[
\mathcal T_F=
\{t_f,t_f+1,\ldots,t_f+H_f-1\}.
\tag{F1}
\]

发电机集合为

\[
\mathcal N=\{1,2\}.
\tag{F2}
\]

其中 \(n=1\) 对应故障机组 \(G_1\)，\(n=2\) 对应健康机组 \(G_2\)。

---

## 2. 主要参数

| 符号 | 定义 |
|---|---|
| \(P_{g,n}^{\min},P_{g,n}^{\max}\) | 发电机 \(G_n\) 正常最小/最大出力 |
| \(R_{g,n}^{\max}\) | 发电机 \(G_n\) 最大爬坡功率 |
| \(\alpha_F\) | \(G_1\) 故障降额系数，\(0\le\alpha_F\le1\) |
| \(P_E^{\min},P_E^{\max}\) | ESS 最大充电/放电功率，约定 \(P_E>0\) 为放电 |
| \(E^{\min},E^{\max}\) | ESS 最小/最大允许储能量 |
| \(\eta_{\mathrm{in}},\eta_{\mathrm{out}}\) | ESS 充电/放电效率 |
| \(P_{PV}(t)\) | 时段 \(t\) 的 PV 输出功率 |
| \(P_{vi}(t)\) | 重要服务负荷 |
| \(P_{nv}(t)\) | 非重要服务负荷 |
| \(V^N(t)\) | 正常阶段已经优化得到的船速 |
| \(\xi_1,\xi_2\) | 航速—推进功率关系参数 |
| \(\Delta t\) | 调度时间间隔 |
| \(f_l\) | 失负荷惩罚系数 |

上标 \(N\) 表示正常运行阶段结果，上标 \(F\) 表示故障阶段变量。

---

## 3. 故障阶段决策变量

故障阶段优化变量为

\[
P_{g,1}^{F}(t),\quad
P_{g,2}^{F}(t),\quad
P_E^{F}(t),\quad
E^{F}(t),\quad
P_{\mathrm{sh}}(t),
\qquad t\in\mathcal T_F.
\tag{F3}
\]

其中：

- \(P_{g,1}^{F}(t)\)：故障后 \(G_1\) 实际出力；
- \(P_{g,2}^{F}(t)\)：故障后 \(G_2\) 实际出力；
- \(P_E^{F}(t)\)：故障阶段 ESS 功率，\(P_E^{F}>0\) 为放电；
- \(E^{F}(t)\)：故障阶段 ESS 剩余能量；
- \(P_{\mathrm{sh}}(t)\)：故障阶段切除的非重要负荷功率。

---

## 4. G1 降额故障模型

定义 \(G_1\) 故障后的最大可用出力为

\[
\boxed{
P_{g,1}^{F,\max}
=
\alpha_F P_{g,1}^{\max}
}
\tag{F4}
\]

其中

\[
0\le\alpha_F\le1.
\]

特殊情况：

\[
\alpha_F=1
\]

表示无故障；

\[
0<\alpha_F<1
\]

表示部分降额；

\[
\alpha_F=0
\]

表示 \(G_1\) 完全停机。

因此故障期间

\[
\boxed{
0\le
P_{g,1}^{F}(t)
\le
P_{g,1}^{F,\max}
}
\qquad
t\in\mathcal T_F.
\tag{F5}
\]

由于故障本身可以导致 \(G_1\) 瞬时掉功率，在故障发生时刻 \(t_f\) **不强制** \(G_1\) 满足由正常工况到故障工况的正常爬坡约束；但故障发生后的相邻时段仍满足

\[
-R_{g,1}^{\max}
\le
P_{g,1}^{F}(t)-P_{g,1}^{F}(t-1)
\le
R_{g,1}^{\max},
\quad
t>t_f.
\tag{F6}
\]

---

## 5. G2 应急支撑约束

健康机组 \(G_2\) 在故障发生后承担应急支撑，其出力满足

\[
\boxed{
P_{g,2}^{\min}
\le
P_{g,2}^{F}(t)
\le
P_{g,2}^{\max}
}
\tag{F7}
\]

以及爬坡约束

\[
-R_{g,2}^{\max}
\le
P_{g,2}^{F}(t)-P_{g,2}^{F}(t-1)
\le
R_{g,2}^{\max},
\quad
t>t_f.
\tag{F8}
\]

故障发生的第一个时段与正常阶段调度结果连接：

\[
\boxed{
-R_{g,2}^{\max}
\le
P_{g,2}^{F}(t_f)-P_{g,2}^{N}(t_f-1)
\le
R_{g,2}^{\max}
}
\tag{F9}
\]

因此 \(G_2\) 虽然可以增加出力，但其故障支撑能力同时受到**容量余量和爬坡能力**限制。

---

## 6. ESS 故障支撑模型

### 6.1 ESS 功率约束

本文统一采用如下符号约定：

\[
P_E^{F}(t)>0
\]

表示 ESS 放电；

\[
P_E^{F}(t)<0
\]

表示 ESS 充电。

故障阶段满足

\[
\boxed{
P_E^{\min}
\le
P_E^{F}(t)
\le
P_E^{\max}
}
\tag{F10}
\]

考虑到电池储能通过电力电子变换器可以快速响应，且本文采用能量调度时间尺度，因此不再单独设置 ESS 相邻调度时段之间的爬坡率约束。ESS 的故障支撑能力主要由最大充放电功率和剩余可用能量共同限制。

---

### 6.2 ESS 能量状态方程

故障期间 ESS 储能量满足

\[
\boxed{
E^{F}(t)=
\begin{cases}
E^{F}(t-1)
-
P_E^{F}(t)\eta_{\mathrm{in}}\Delta t,
&
P_E^{F}(t)\le0,
\\[2mm]
E^{F}(t-1)
-
\dfrac{P_E^{F}(t)}{\eta_{\mathrm{out}}}\Delta t,
&
P_E^{F}(t)>0 .
\end{cases}
}
\tag{F11}
\]

并满足

\[
\boxed{
E^{\min}
\le
E^{F}(t)
\le
E^{\max}
}
\tag{F12}
\]

---

### 6.3 与故障前韧性备用的接口

故障发生前，正常调度阶段已经得到 ESS 能量状态 \(E^N(t_f-1)\)。因此故障阶段初值为

\[
\boxed{
E^{F}(t_f-1)=E^{N}(t_f-1)
}
\tag{F13}
\]

正常阶段提出的韧性备用约束为

\[
\boxed{
E^{N}(t_f-1)
\ge
E^{\min}+E_{\mathrm{res}}(t_f,H_r)
}
\tag{F14}
\]

其中 \(E_{\mathrm{res}}\) 表示针对给定 \(G_1\) 降额程度和设计备用时长 \(H_r\) 所预留的 ESS 应急能量。实际故障持续时间 \(H_f\) 不作为正常阶段备用约束的输入。

式 (F13)–(F14) 是**正常阶段与故障阶段之间的状态接口**：正常阶段负责“提前留下多少电”，故障阶段负责“实际怎样使用这些电”。

---


## 6.4 动态韧性备用量 \(E_{\mathrm{res}}(t,H_r)\) 的闭环计算

式 (F14) 中的 \(E_{\mathrm{res}}(t,H_r)\) 不是预设常数，而是根据正常调度候选方案在每个时刻进行一次“虚拟 \(G_1\) 降额故障”计算得到。这里 \(H_r\) 表示设计的故障支撑时长（reserve horizon）。

首先定义故障后的净功率需求

\[
\boxed{
P_D(\tau)
=
P_{pr}^{N}(\tau)+P_{vi}(\tau)+P_{nv}(\tau)-P_{PV}(\tau)
}
\tag{R1}
\]

其中

\[
\tau=t,t+1,\ldots,t+H_r-1.
\]

\(G_1\) 降额后最大可用出力为

\[
\boxed{
\bar P_{g,1}^{F}(\tau)
=
\alpha_F P_{g,1}^{\max}
}
\tag{R2}
\]

健康机组 \(G_2\) 的最大应急可用出力同时受额定容量和爬坡能力约束。故障第一个时段有

\[
\boxed{
\bar P_{g,2}^{F}(t)
=
\min\left\{
P_{g,2}^{\max},
P_{g,2}^{N}(t-1)+R_{g,2}^{\max}
\right\}
}
\tag{R3}
\]

后续时段递推为

\[
\boxed{
\bar P_{g,2}^{F}(\tau)
=
\min\left\{
P_{g,2}^{\max},
\bar P_{g,2}^{F}(\tau-1)+R_{g,2}^{\max}
\right\}
}
\tag{R4}
\]

因此，在“不切负荷”的备用设计要求下，ESS 必须承担的最小功率缺额为

\[
\boxed{
P_{E,\mathrm{res}}(\tau)
=
\left[
P_D(\tau)
-
\bar P_{g,1}^{F}(\tau)
-
\bar P_{g,2}^{F}(\tau)
\right]^+
}
\tag{R5}
\]

其中 \([x]^+=\max(x,0)\)。由此得到动态 ESS 韧性备用能量

\[
\boxed{
E_{\mathrm{res}}(t,H_r)
=
\sum_{\tau=t}^{t+H_r-1}
\frac{P_{E,\mathrm{res}}(\tau)}{\eta_{\mathrm{out}}}\Delta t
}
\tag{R6}
\]

并将其反馈到正常阶段 ESS 能量约束：

\[
\boxed{
E^{N}(t-1)
\ge
E^{\min}+E_{\mathrm{res}}(t,H_r)
}
\tag{R7}
\]

此外，要实现“设计故障范围内零切负荷”，还必须满足 ESS 的功率可行性：

\[
\boxed{
0\le P_{E,\mathrm{res}}(\tau)\le P_E^{\max}
}
\tag{R8}
\]

因此，式 (R7) 用于保证 ESS 的能量备用充足，式 (R8) 用于保证每个故障时段所需的应急放电功率不超过 ESS 的最大放电能力。若式 (R8) 不满足，则说明即使 ESS 储能量充足，也无法仅依靠储能实现零切负荷，此时故障阶段必须允许 \(P_{\mathrm{sh}}>0\)。

## 7. 航速与推进负荷

为突出 ESS 韧性备用的作用并控制模型规模，故障阶段暂不重新优化航速：

\[
\boxed{
V^{F}(t)=V^{N}(t)
}
\tag{F15}
\]

因此推进功率直接由正常阶段航速决定：

\[
\boxed{
P_{pr}^{F}(t)
=
\xi_1
\left[V^{N}(t)\right]^{\xi_2}
}
\tag{F16}
\]

因此本文故障阶段不额外引入航程调整变量。

---

## 8. 非重要负荷切除模型

定义故障阶段切负荷功率

\[
P_{\mathrm{sh}}(t)\ge0.
\]

由于只允许切除非重要负荷，因此

\[
\boxed{
0
\le
P_{\mathrm{sh}}(t)
\le
P_{nv}(t)
}
\tag{F17}
\]

也可进一步定义非重要负荷切除比例 \(\rho^{F}(t)\)：

\[
\boxed{
P_{\mathrm{sh}}(t)
=
\rho^{F}(t)P_{nv}(t),
\qquad
0\le\rho^{F}(t)\le1
}
\tag{F18}
\]

因此实际被供电的服务负荷为

\[
\boxed{
P_L^{F}(t)
=
P_{vi}(t)
+
P_{nv}(t)
-
P_{\mathrm{sh}}(t)
}
\tag{F19}
\]

重要负荷 \(P_{vi}(t)\) 不参与切除。

---

## 9. 故障阶段全船功率平衡

由于本文不考虑孤岛，故障后仍只有一条全船功率平衡方程：

\[
\boxed{
P_{g,1}^{F}(t)
+
P_{g,2}^{F}(t)
+
P_E^{F}(t)
+
P_{PV}(t)
=
P_{pr}^{F}(t)
+
P_{vi}(t)
+
P_{nv}(t)
-
P_{\mathrm{sh}}(t)
}
\tag{F20}
\]

等价地，

\[
\boxed{
P_{\mathrm{sh}}(t)
=
P_{pr}^{F}(t)
+
P_{vi}(t)
+
P_{nv}(t)
-
P_{PV}(t)
-
P_{g,1}^{F}(t)
-
P_{g,2}^{F}(t)
-
P_E^{F}(t)
}
\tag{F21}
\]

并结合式 (F17) 保证 \(P_{\mathrm{sh}}(t)\) 非负且不超过非重要负荷。

式 (F20) 表达了本文最核心的故障调度关系：

\[
\boxed{
G_1\text{降额造成的供电损失}
\rightarrow
G_2\text{增发}
+
ESS\text{放电}
+
\text{必要的非重要负荷切除}
}
\]

---

## 10. 故障阶段目标函数

### 10.1 发电机运行成本

发电机运行成本采用二次函数表示：

\[
C_{G,n}^{F}(t)
=
\alpha_{g,n}
\left[P_{g,n}^{F}(t)\right]^2
\Delta t
+
\beta_{g,n}
P_{g,n}^{F}(t)
\Delta t
+
\gamma_{g,n}\Delta t.
\tag{F22}
\]

### 10.2 ESS 运行成本

\[
C_E^{F}(t)
=
\alpha_E
\left[P_E^{F}(t)\right]^2
\Delta t
+
\gamma_E\Delta t.
\tag{F23}
\]

### 10.3 失负荷代价

定义

\[
\boxed{
C_{\mathrm{sh}}^{F}(t)
=
f_l P_{\mathrm{sh}}(t)\Delta t
}
\tag{F24}
\]

其中 \(f_l\) 取较大值，使负荷切除成为最后使用的应急手段。

### 10.4 总目标函数

故障阶段优化问题写为

\[
\boxed{
\min J_F
=
\sum_{t\in\mathcal T_F}
\left[
\sum_{n=1}^{2}C_{G,n}^{F}(t)
+
C_E^{F}(t)
+
f_lP_{\mathrm{sh}}(t)\Delta t
\right]
}
\tag{F25}
\]

由于 \(P_{PV}(t)\) 在故障阶段被视为给定预测量，其成本项对优化结果是常数，因此可不写入目标函数。

当 \(f_l\) 足够大时，优化器的调度优先级为：

\[
\boxed{
\text{利用可用发电能力和 ESS}
\quad\rightarrow\quad
\text{仍无法平衡时再切非重要负荷}
}
\]

通过设置足够大的失负荷惩罚系数，可使优化器遵循“优先利用可用电源与 ESS，供电仍不足时再切负荷”的调度原则。

---

## 11. 可保留的 EEOI 排放约束

若需要同时考虑环保约束，可在故障阶段加入

\[
\boxed{
\frac{
\sum_{n=1}^{2}
F_n\!\left(P_{g,n}^{F}(t)\right)
}{
\xi_3 V^{N}(t)\Delta t
}
\le
EEOI^{\max}
}
\tag{F26}
\]

其中 \(F_n(\cdot)\) 为发电机 \(G_n\) 的 CO\(_2\) 排放函数。

若研究重点只讨论短时故障韧性，可保留式 (F26) 作为常规运行约束，而不将其作为新增贡献。

---

## 12. 故障阶段完整优化模型

最终，给定正常阶段状态

\[
P_{g,2}^{N}(t_f-1),\quad
E^{N}(t_f-1),\quad
V^{N}(t),
\]

以及故障参数

\[
\alpha_F,\quad H_f,
\]

求解

\[
\boxed{
\begin{aligned}
\min_{\substack{
P_{g,1}^{F},
P_{g,2}^{F},
P_E^{F},
E^{F},
P_{\mathrm{sh}}
}}
\quad
&
J_F
\\[1mm]
\mathrm{s.t.}\quad
&
\text{G1 降额与出力约束：}(F4)-(F6),
\\
&
\text{G2 出力与爬坡约束：}(F7)-(F9),
\\
&
\text{ESS 功率与能量约束：}(F10)-(F14),
\\
&
\text{固定航速与推进功率：}(F15)-(F16),
\\
&
\text{负荷切除约束：}(F17)-(F19),
\\
&
\text{全船功率平衡：}(F20),
\\
&
\text{必要时加入 EEOI：}(F26).
\end{aligned}
}
\tag{F27}
\]

该模型的输出包括

\[
P_{g,1}^{F}(t),\;
P_{g,2}^{F}(t),\;
P_E^{F}(t),\;
E^{F}(t),\;
P_{\mathrm{sh}}(t).
\]

---

## 13. 韧性评价指标

故障阶段最直接的韧性指标采用总失负荷电量：

\[
\boxed{
E_{\mathrm{sh}}
=
\sum_{t\in\mathcal T_F}
P_{\mathrm{sh}}(t)\Delta t
}
\tag{F28}
\]

\(E_{\mathrm{sh}}\) 越小，说明故障阶段维持负荷供电的能力越强。

也可定义非重要负荷供电保持率：

\[
\boxed{
R_{\mathrm{load}}
=
1-
\frac{
\sum_{t\in\mathcal T_F}
P_{\mathrm{sh}}(t)\Delta t
}{
\sum_{t\in\mathcal T_F}
P_{nv}(t)\Delta t
}
}
\tag{F29}
\]

其中

\[
0\le R_{\mathrm{load}}\le1.
\]

当

\[
R_{\mathrm{load}}=1
\]

时，故障期间无需切除任何非重要负荷。

---

## 14. 正常阶段—故障阶段的完整闭环关系

本文整个研究逻辑可以表示为

\[
\boxed{
\begin{array}{c}
\text{正常阶段联合经济调度}
\\
\downarrow
\\
E^{N}(t)\ge E^{\min}+E_{\mathrm{res}}(t,H)
\\
\downarrow
\\
\text{在 ESS 中预留韧性备用}
\\
\downarrow
\\
G_1\text{ 突发降额}
\\
\downarrow
\\
G_2\text{ 增发}+ESS\text{ 应急放电}
\\
\downarrow
\\
\text{仍有供电缺额时切除非重要负荷}
\\
\downarrow
\\
E_{\mathrm{sh}}\text{ 降低，故障韧性提高}
\end{array}
}
\tag{F30}
\]

因此，后续仿真真正需要比较的是：

1. **无备用策略（No reserve）**：正常阶段不设置 ESS 韧性备用；
2. **动态韧性备用策略（Dynamic reserve）**：根据潜在 \(G_1\) 降额程度和设计支撑时长动态确定 \(E_{\mathrm{res}}\)。

两种正常调度方案分别提供不同的

\[
E^{N}(t_f-1),
\]

再将其作为同一个故障模型 (F27) 的初始条件，比较

\[
J_F,\qquad
E_{\mathrm{sh}},\qquad
R_{\mathrm{load}}.
\]

这就形成完整的“**故障前备用配置—故障后应急响应—韧性效果评价**”闭环。

---

## 15. 调度方案的仿真分析框架

为了验证动态韧性备用的作用，设置两种调度策略，并在完全相同的故障条件下比较：

1. **无备用策略（No reserve）**：不设置故障备用；
2. **动态韧性备用策略（Dynamic reserve）**：采用式 (R1)–(R7) 的动态韧性备用。

分析分为三层。

### 15.1 正常航行经济性

比较两种策略的正常运行成本

\[
\boxed{
\Delta C_N
=
\frac{J_N^{\mathrm{method}}-J_N^{\mathrm{NoRes}}}
{J_N^{\mathrm{NoRes}}}\times100\%
}
\tag{A1}
\]

并比较正常阶段的 ESS 能量轨迹 \(E^N(t)\) 与动态备用曲线 \(E_{\mathrm{res}}(t,H_r)\)。这一部分回答“为了韧性备用付出了多少经济代价”。

### 15.2 代表性故障过程

给定同一故障时刻 \(t_f\)、降额系数 \(\alpha_F\) 和实际持续时间 \(H_f\)，比较：

- \(G_1\)、\(G_2\) 故障出力；
- ESS 应急放电功率与剩余能量；
- 切负荷功率 \(P_{\mathrm{sh}}(t)\)；
- 总失负荷电量 \(E_{\mathrm{sh}}\)。

重点展示“动态备用使故障前 SoC 更高，从而在故障后减少或推迟切负荷”的因果链条。

### 15.3 故障严重程度与持续时间敏感性

分别改变实际故障持续时间 \(H_f\) 和降额系数 \(\alpha_F\)，比较两种方法的

\[
E_{\mathrm{sh}},\qquad R_{\mathrm{load}},\qquad J_F.
\]

建议将设计备用时长 \(H_r\) 与实际故障持续时间 \(H_f\) 区分开：\(H_r\) 是正常调度时选择的韧性水平，\(H_f\) 是仿真中实际发生的故障持续时间。这样可以检验当 \(H_f\le H_r\) 时方案是否达到预期保护效果，以及当 \(H_f>H_r\) 时系统韧性如何逐渐退化，而不是只验证一个预设场景。

最终论文应重点说明的关系是

\[
\boxed{
\text{少量正常运行成本增加}
\quad\Longrightarrow\quad
\text{更高故障前 ESS 可用能量}
\quad\Longrightarrow\quad
\text{更低故障失负荷}
}
\tag{A2}
\]
