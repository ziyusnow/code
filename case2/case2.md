# Case 2 调度优化模型

## 1. 优化目标

在 24 h 内完成 240 nm 航程，联合优化船速、ESS 充放电和两台柴油发电机出力，使总运行成本最小。

系统包含：

- 发电机 \(G_1\)
- 发电机 \(G_2\)
- 能量存储系统 ESS
- 光伏 PV
- 服务负荷 \(P_l(t)\)
- 推进负荷 \(P_{pr}(t)\)

Case 2 不进行负荷切除。

---

## 2. 已知参数

### 2.1 时间与航程

\[
t=1,\dots,24
\]

\[
\Delta t=1\ \mathrm{h}
\]

\[
D=240\ \mathrm{nm}
\]

### 2.2 发电机参数

| 参数 | \(G_1\) | \(G_2\) |
|---|---:|---:|
| \(P_g^{min}\) | 0 MW | 0 MW |
| \(P_g^{max}\) | 10 MW | 20 MW |
| \(R_g^{max}\) | 2 MW/h | 3 MW/h |
| \(\alpha_g\) | 13 | 5.2 |
| \(\beta_g\) | 12 | 52 |
| \(\gamma_g\) | 430 | 340 |
| \(\dot\alpha_g\) | 13.5 | 5.2 |
| \(\dot\beta_g\) | 10 | 58 |
| \(\dot\gamma_g\) | 450 | 390 |
| \(f_g\) | 1 | 1 |

记录初始出力：

\[
P_{g1}(0)=0,\qquad P_{g2}(0)=0
\]

Ramp 约束从：

\[
t=2,\dots,24
\]

开始检查。

### 2.3 ESS 参数

ESS 功率符号：

- \(P_e(t)>0\)：放电
- \(P_e(t)<0\)：充电

\[
-3\le P_e(t)\le3\ \mathrm{MW}
\]

\[
R_e^{max}=1\ \mathrm{MW/h}
\]

\[
\eta_{in}=0.95,\qquad \eta_{out}=0.95
\]

\[
E_e^{min}=15\ \mathrm{MWh}
\]

\[
E_e^{max}=75\ \mathrm{MWh}
\]

\[
\alpha_e=4.3,\qquad \gamma_e=1,\qquad f_e=1
\]

初始 SOC：

\[
SOC(0)=0.5
\]

定义：

\[
SOC(t)=\frac{E_e(t)}{E_e^{max}}
\]

因此：

\[
\boxed{
E_e(0)=37.5\ \mathrm{MWh}
}
\]

SOC 允许范围：

\[
0.2\le SOC(t)\le1
\]

不设置终端 SOC 等式约束。

记录：

\[
P_e(0)=0
\]

ESS Ramp 同样从：

\[
t=2,\dots,24
\]

开始检查。

### 2.4 PV 参数

\[
0\le P_{pv}(t)\le4.2\ \mathrm{MW}
\]

PV 功率 \(P_{pv}(t)\) 作为 24 h 输入数据，不作为决策变量。

\[
\beta_{pv}=10.2,\qquad \gamma_{pv}=0,\qquad f_{pv}=1
\]

### 2.5 推进参数

\[
0\le V(t)\le11\ \mathrm{kn}
\]

\[
\xi_1=0.0022,\qquad \xi_2=3
\]

\[
\boxed{
P_{pr}(t)=0.0022V(t)^3
}
\]

### 2.6 EEOI 参数

\[
\xi_3=20
\]

\[
EEOI^{max}=23
\]

---

## 3. 输入数据

输入 24 h 数据：

\[
P_{vital}(t)
\]

\[
P_{nonvital}(t)
\]

\[
P_{pv}(t)
\]

服务负荷：

\[
\boxed{
P_l(t)=P_{vital}(t)+P_{nonvital}(t)
}
\]

建议数据格式：

```csv
hour,p_vital_mw,p_nonvital_mw,p_pv_mw
1,...
2,...
...
24,...
```

---

## 4. 决策变量

采用 72 维粒子：

\[
\boxed{
X=
[V(1{:}24),q_e(1{:}24),q_g(1{:}24)]
}
\]

其中：

\[
0\le q_e(t)\le1
\]

\[
0\le q_g(t)\le1
\]

- \(V(t)\)：船速
- \(q_e(t)\)：ESS 功率分配变量
- \(q_g(t)\)：发电机出力分配变量

\(P_e(t)\)、\(P_{g1}(t)\)、\(P_{g2}(t)\) 通过解码得到。

---

## 5. 航程约束

\[
\sum_{t=1}^{24}V(t)\Delta t=D
\]

由于 \(\Delta t=1\)：

\[
\boxed{
\sum_{t=1}^{24}V(t)=240
}
\]

每次位置更新后，将 \(V\) 投影到：

\[
0\le V(t)\le11
\]

\[
\sum_tV(t)=240
\]

采用：

\[
V_t=\operatorname{clip}(z_t-\tau,0,11)
\]

通过二分搜索 \(\tau\)，使：

\[
\sum_tV_t=240
\]

---

## 6. 推进功率

\[
\boxed{
P_{pr}(t)=0.0022V(t)^3
}
\]

---

## 7. ESS 与发电机解码

### 7.1 净功率需求

定义：

\[
\boxed{
A(t)=P_l(t)+P_{pr}(t)-P_{pv}(t)
}
\]

功率平衡可写为：

\[
P_{g1}(t)+P_{g2}(t)+P_e(t)=A(t)
\]

### 7.2 ESS 可行功率区间

由于：

\[
0\le P_{g1}(t)+P_{g2}(t)\le30
\]

有：

\[
A(t)-30\le P_e(t)\le A(t)
\]

结合：

\[
-3\le P_e(t)\le3
\]

得到：

\[
\boxed{
L_e(t)=\max[-3,A(t)-30]
}
\]

\[
\boxed{
U_e(t)=\min[3,A(t)]
}
\]

若：

\[
L_e(t)\le U_e(t)
\]

则：

\[
\boxed{
P_e(t)=L_e(t)+q_e(t)[U_e(t)-L_e(t)]
}
\]

### 7.3 容量可行性

系统静态净功率范围：

\[
-3\le A(t)\le33
\]

容量违反量：

\[
CV_D(t)
=
\max
\left(
0,
\frac{A(t)-33}{33}
\right)
+
\max
\left(
0,
\frac{-3-A(t)}{3}
\right)
\]

\[
CV_D=\sum_{t=1}^{24}CV_D(t)
\]

### 7.4 发电机总功率

\[
\boxed{
P_G(t)=A(t)-P_e(t)
}
\]

### 7.5 两台发电机出力分配

定义：

\[
L_g(t)=\max[0,P_G(t)-20]
\]

\[
U_g(t)=\min[10,P_G(t)]
\]

则：

\[
\boxed{
P_{g1}(t)
=
L_g(t)+q_g(t)[U_g(t)-L_g(t)]
}
\]

\[
\boxed{
P_{g2}(t)=P_G(t)-P_{g1}(t)
}
\]

由此自动满足：

\[
0\le P_{g1}(t)\le10
\]

\[
0\le P_{g2}(t)\le20
\]

以及：

\[
\boxed{
P_{g1}(t)+P_{g2}(t)+P_e(t)+P_{pv}(t)
=
P_l(t)+P_{pr}(t)
}
\]

---

## 8. ESS 能量模型

初始能量：

\[
E_e(0)=37.5\ \mathrm{MWh}
\]

当：

\[
P_e(t)\le0
\]

时：

\[
\boxed{
E_e(t)
=
E_e(t-1)-P_e(t)\eta_{in}\Delta t
}
\]

即：

\[
E_e(t)=E_e(t-1)-0.95P_e(t)
\]

当：

\[
P_e(t)>0
\]

时：

\[
\boxed{
E_e(t)
=
E_e(t-1)-\frac{P_e(t)}{\eta_{out}}\Delta t
}
\]

即：

\[
E_e(t)=E_e(t-1)-\frac{P_e(t)}{0.95}
\]

SOC：

\[
\boxed{
SOC(t)=\frac{E_e(t)}{75}
}
\]

约束：

\[
15\le E_e(t)\le75
\]

等价于：

\[
0.2\le SOC(t)\le1
\]

SOC 违反量：

\[
CV_{SOC}(t)
=
\max
\left(
0,
\frac{15-E_e(t)}{60}
\right)
+
\max
\left(
0,
\frac{E_e(t)-75}{60}
\right)
\]

\[
\boxed{
CV_{SOC}
=
\sum_{t=1}^{24}CV_{SOC}(t)
}
\]

---

## 9. Ramp 约束

### 9.1 \(G_1\)

\[
|P_{g1}(t)-P_{g1}(t-1)|\le2
\]

\[
CV_{R1}(t)
=
\max
\left(
0,
\frac{|P_{g1}(t)-P_{g1}(t-1)|-2}{2}
\right)
\]

### 9.2 \(G_2\)

\[
|P_{g2}(t)-P_{g2}(t-1)|\le3
\]

\[
CV_{R2}(t)
=
\max
\left(
0,
\frac{|P_{g2}(t)-P_{g2}(t-1)|-3}{3}
\right)
\]

### 9.3 ESS

\[
|P_e(t)-P_e(t-1)|\le1
\]

\[
CV_{Re}(t)
=
\max
\left(
0,
|P_e(t)-P_e(t-1)|-1
\right)
\]

总 Ramp 违反量：

\[
\boxed{
CV_R
=
\sum_{t=2}^{24}
\left[
CV_{R1}(t)+CV_{R2}(t)+CV_{Re}(t)
\right]
}
\]

---

## 10. 发电机排放模型

\(G_1\)：

\[
\boxed{
F_1(t)
=
13.5P_{g1}^2(t)+10P_{g1}(t)+450
}
\]

\(G_2\)：

\[
\boxed{
F_2(t)
=
5.2P_{g2}^2(t)+58P_{g2}(t)+390
}
\]

总排放：

\[
\boxed{
F_{total}
=
\sum_{t=1}^{24}[F_1(t)+F_2(t)]
}
\]

---

## 11. EEOI 约束

为防止船速为 0 时出现除零，定义：

\[
\boxed{
V_E(t)=\max[V(t),10^{-6}]
}
\]

EEOI：

\[
\boxed{
EEOI(t)
=
\frac{F_1(t)+F_2(t)}
{20V_E(t)}
}
\]

约束：

\[
EEOI(t)\le23
\]

违反量：

\[
CV_E(t)
=
\max
\left(
0,
\frac{EEOI(t)-23}{23}
\right)
\]

\[
\boxed{
CV_E
=
\sum_{t=1}^{24}CV_E(t)
}
\]

---

## 12. 运行成本

### 12.1 \(G_1\)

\[
\boxed{
C_{G1}(t)
=
13P_{g1}^2(t)+12P_{g1}(t)+430
}
\]

### 12.2 \(G_2\)

\[
\boxed{
C_{G2}(t)
=
5.2P_{g2}^2(t)+52P_{g2}(t)+340
}
\]

### 12.3 ESS

\[
\boxed{
C_E(t)
=
4.3P_e^2(t)+1
}
\]

### 12.4 PV

\[
\boxed{
C_{PV}(t)
=
10.2P_{pv}(t)
}
\]

总成本：

\[
\boxed{
J=
\sum_{t=1}^{24}
\left[
C_{G1}(t)
+
C_{G2}(t)
+
C_E(t)
+
C_{PV}(t)
\right]
}
\]

优化目标：

\[
\boxed{
\min J
}
\]

---

## 13. 总约束违反量

\[
\boxed{
CV
=
CV_R
+
CV_{SOC}
+
CV_E
+
CV_D
}
\]

可行解条件：

\[
\boxed{
CV\le10^{-10}
}
\]

---

## 14. Fitness

\[
\boxed{
f=J+\lambda CV
}
\]

取：

\[
\boxed{
\lambda=10^6
}
\]

更新 `pbest` 和 `gbest` 时采用 feasibility-first：

1. 可行解优于不可行解；
2. 两个都可行时，总成本 \(J\) 较小者优；
3. 两个都不可行时，总违反量 \(CV\) 较小者优；
4. 若 \(CV\) 相同，再比较 \(J\)。

---

## 15. MPPSO 参数

```yaml
population_size: 600
max_iterations: 500

omega_max: 0.9
omega_min: 0.4

c_max: 1.5
c_min: 0.5

alpha: 0.8
zeta: 0.8
c3: -0.4

velocity_clamp_v: 2.2
velocity_clamp_qe: 0.2
velocity_clamp_qg: 0.2

penalty_lambda: 1.0e6
feasibility_tolerance: 1.0e-10
numerical_epsilon: 1.0e-12
eeoi_velocity_epsilon: 1.0e-6
```

种群中心：

\[
\boxed{
p_{cen}^{k}
=
\frac{1}{I}
\sum_{i=1}^{I}p_i^k
}
\]

历史状态初始化：

\[
p_i^{-1}=p_i^0
\]

\[
v_i^{-1}=v_i^0
\]

取：

\[
v_i^0=0
\]

---

## 16. MPPSO 更新

惯性权重：

\[
\omega_i=
\begin{cases}
\omega_{\min}
-(\omega_{\min}-\omega_{\max})
\dfrac{f_i-f_{\mathrm{mean}}}
{f_{\max}-f_{\mathrm{mean}}},
&
f_i\ge f_{\mathrm{mean}}
\\[10pt]
\omega_{\min}
+(\omega_{\max}-\omega_{\min})
\dfrac{f_i-f_{\min}}
{f_{\mathrm{mean}}-f_{\min}},
&
f_i<f_{\mathrm{mean}}
\end{cases}
\]

学习系数：

\[
c_1=
\begin{cases}
c_{\max}
+
(c_{\max}-c_{\min})
\dfrac{f_i-f_{\min}}
{f_{\mathrm{mean}}-f_{\min}},
&
f_i\le f_{\mathrm{mean}}
\\[10pt]
c_{\min},
&
f_i>f_{\mathrm{mean}}
\end{cases}
\]

\[
\boxed{
c_2=2-c_1
}
\]

所有分母采用：

```text
max(abs(denominator), 1e-12)
```

定义：

\[
\bar p_{i,j}^{k}
=
\alpha p_{i,j}^{k}
+
(1-\alpha)p_{i,j}^{k-1}
\]

速度更新：

\[
\boxed{
\begin{aligned}
v_{i,j}^{k+1}
=&\
\omega_i
\left[
\zeta v_{i,j}^{k}
+
(1-\zeta)v_{i,j}^{k-1}
\right]
\\
&+
c_1
\left[
l_{i,j}^{k}
-
\bar p_{i,j}^{k}
\right]
\\
&+
c_2
\left[
p_{g,j}^{k}
-
\bar p_{i,j}^{k}
\right]
\\
&+
c_3r_{3,j}
\left[
p_{cen,j}^{k}
-
\bar p_{i,j}^{k}
\right]
\end{aligned}
}
\]

其中：

\[
r_{3,j}\sim U(0,1)
\]

位置更新：

\[
\boxed{
p_i^{k+1}=p_i^k+v_i^{k+1}
}
\]

---

## 17. Velocity Clamp 与位置修复

船速维度：

\[
-2.2\le v_V\le2.2
\]

ESS 分配维度：

\[
-0.2\le v_{q_e}\le0.2
\]

发电机分配维度：

\[
-0.2\le v_{q_g}\le0.2
\]

位置更新后：

\[
q_e\leftarrow\operatorname{clip}(q_e,0,1)
\]

\[
q_g\leftarrow\operatorname{clip}(q_g,0,1)
\]

\[
V\leftarrow
\operatorname{Proj}
\left\{
0\le V\le11,\;
\sum_tV_t=240
\right\}
\]

---

## 18. 求解流程

```text
输入：
    P_vital[24]
    P_nonvital[24]
    P_pv[24]

计算：
    P_load = P_vital + P_nonvital

初始化 600 个粒子：
    X = [V(1:24), qe(1:24), qg(1:24)]

    V  <- 航程投影
    qe <- clip(qe,0,1)
    qg <- clip(qg,0,1)

    v0 = 0

对每个粒子：

    P_prop = 0.0022 * V^3
    A = P_load + P_prop - P_pv

    Le = max(-3, A - 30)
    Ue = min( 3, A)

    计算 CV_D

    Pe = Le + qe*(Ue-Le)

    PG = A - Pe

    Lg = max(0, PG - 20)
    Ug = min(10, PG)

    Pg1 = Lg + qg*(Ug-Lg)
    Pg2 = PG - Pg1

    从 E0 = 37.5 开始递推 E
    SOC = E / 75

    计算：
        CV_R
        CV_SOC
        EEOI
        CV_E
        CV

    计算：
        C_G1
        C_G2
        C_ESS
        C_PV
        J

    f = J + 1e6*CV

初始化 pbest、gbest

for k = 1 to 500:

    计算：
        f_min
        f_mean
        f_max
        p_cen

    对每个粒子：

        更新 omega
        更新 c1、c2

        更新速度
        velocity clamp

        更新位置

        qe <- clip(qe,0,1)
        qg <- clip(qg,0,1)
        V  <- 航程投影

        重新解码：
            Pe
            Pg1
            Pg2

        重新计算：
            E
            SOC
            EEOI
            J
            CV
            f

        feasibility-first 更新 pbest

    feasibility-first 更新 gbest

输出最佳可行解
```

---

## 19. 输出结果

逐小时输出：

```csv
hour,
p_vital_mw,
p_nonvital_mw,
p_load_mw,
p_pv_mw,
v_kn,
p_propulsion_mw,
p_ess_mw,
energy_ess_mwh,
soc,
p_g1_mw,
p_g2_mw,
eeoi,
co2_g1,
co2_g2,
cost_g1,
cost_g2,
cost_ess,
cost_pv
```

同时输出：

\[
\boxed{
J^*
}
\]

\[
\boxed{
F_{total}
}
\]

\[
E_e(24)
\]

\[
SOC(24)
\]

以及收敛曲线。

---

## 20. 最终约束检查

```text
sum(V) = 240

0 <= V <= 11

power balance residual ≈ 0

0 <= Pg1 <= 10
0 <= Pg2 <= 20

-3 <= Pe <= 3

ramp G1 <= 2
ramp G2 <= 3
ramp ESS <= 1

15 <= E <= 75
0.2 <= SOC <= 1

EEOI <= 23

CV ≈ 0
```

功率平衡残差：

\[
r_P(t)
=
P_{g1}(t)
+
P_{g2}(t)
+
P_e(t)
+
P_{pv}(t)
-
P_l(t)
-
P_{pr}(t)
\]

要求：

\[
\boxed{
\max_t|r_P(t)|\approx0
}
\]