# Case 1 调度优化模型

## 1. 优化目标

在 24 h 内完成 240 nm 航程，联合优化船速与两台柴油发电机出力，使总运行成本最小。

系统仅包含：

- 发电机 \(G_1\)
- 发电机 \(G_2\)
- 服务负荷 \(P_l(t)\)
- 推进负荷 \(P_{pr}(t)\)

不包含 ESS、PV 和负荷切除。

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

初始出力：

\[
P_{g1}(0)=0,\qquad P_{g2}(0)=0
\]

主调度中 ramp 约束从 \(t=2\) 开始。

### 2.3 推进参数

\[
0\le V(t)\le11\ \mathrm{kn}
\]

\[
\xi_1=0.0022,\qquad \xi_2=3
\]

### 2.4 EEOI 参数

\[
\xi_3=20
\]

\[
EEOI^{max}=23
\]

---

## 3. 输入数据

输入 24 h 服务负荷：

\[
P_l(t),\qquad t=1,\dots,24
\]

若分别给出关键负荷与非关键负荷，则：

\[
P_l(t)=P_{vital}(t)+P_{nonvital}(t)
\]

建议数据格式：

```csv
hour,p_load_mw
1,...
2,...
...
24,...
```

---

## 4. 决策变量

采用 48 维粒子：

\[
X=
[V(1),\dots,V(24),q(1),\dots,q(24)]
\]

其中：

\[
0\le q(t)\le1
\]

\(q(t)\) 用于分配两台发电机的出力。

---

## 5. 推进功率

\[
P_{pr}(t)=\xi_1V(t)^{\xi_2}
\]

即：

\[
\boxed{
P_{pr}(t)=0.0022V(t)^3
}
\]

---

## 6. 航程约束

\[
\sum_{t=1}^{24}V(t)\Delta t=D
\]

由于 \(\Delta t=1\)：

\[
\boxed{
\sum_{t=1}^{24}V(t)=240
}
\]

每次粒子位置更新后，将 \(V\) 投影到：

\[
0\le V(t)\le11,\qquad \sum_tV(t)=240
\]

可采用：

\[
V_t=\operatorname{clip}(z_t-\tau,0,11)
\]

通过二分搜索 \(\tau\)，使：

\[
\sum_tV_t=240
\]

---

## 7. 功率平衡与发电机出力解码

总需求：

\[
P_D(t)=P_l(t)+P_{pr}(t)
\]

功率平衡：

\[
P_{g1}(t)+P_{g2}(t)=P_D(t)
\]

定义 \(G_1\) 的可行出力区间：

\[
L_t=\max(0,P_D(t)-20)
\]

\[
U_t=\min(10,P_D(t))
\]

若：

\[
L_t>U_t
\]

则当前粒子不可行。

否则：

\[
\boxed{
P_{g1}(t)=L_t+q(t)(U_t-L_t)
}
\]

\[
\boxed{
P_{g2}(t)=P_D(t)-P_{g1}(t)
}
\]

这样自动满足：

\[
0\le P_{g1}(t)\le10
\]

\[
0\le P_{g2}(t)\le20
\]

以及功率平衡。

---

## 8. 发电机运行成本

\[
C_{G1}(t)
=
13P_{g1}^2(t)+12P_{g1}(t)+430
\]

\[
C_{G2}(t)
=
5.2P_{g2}^2(t)+52P_{g2}(t)+340
\]

总成本：

\[
\boxed{
J=
\sum_{t=1}^{24}
\left[
C_{G1}(t)+C_{G2}(t)
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

## 9. Ramp 约束

\[
|P_{g1}(t)-P_{g1}(t-1)|\le2
\]

\[
|P_{g2}(t)-P_{g2}(t-1)|\le3
\]

主调度中：

\[
t=2,\dots,24
\]

约束违反量：

\[
CV_{R1}(t)=
\max
\left(
0,
\frac{|P_{g1}(t)-P_{g1}(t-1)|-2}{2}
\right)
\]

\[
CV_{R2}(t)=
\max
\left(
0,
\frac{|P_{g2}(t)-P_{g2}(t-1)|-3}{3}
\right)
\]

\[
CV_R=
\sum_{t=2}^{24}
\left[
CV_{R1}(t)+CV_{R2}(t)
\right]
\]

---

## 10. EEOI 约束

CO2 排放模型：

\[
F_1(t)=
13.5P_{g1}^2(t)+10P_{g1}(t)+450
\]

\[
F_2(t)=
5.2P_{g2}^2(t)+58P_{g2}(t)+390
\]

为避免 \(V(t)=0\) 导致除零：

\[
V_E(t)=\max(V(t),10^{-6})
\]

EEOI：

\[
EEOI(t)=
\frac{F_1(t)+F_2(t)}
{20V_E(t)}
\]

约束：

\[
EEOI(t)\le23
\]

违反量：

\[
CV_E(t)=
\max
\left(
0,
\frac{EEOI(t)-23}{23}
\right)
\]

\[
CV_E=\sum_{t=1}^{24}CV_E(t)
\]

---

## 11. 总约束违反量

若：

\[
P_D(t)>30
\]

定义容量违反量：

\[
CV_D(t)=
\max
\left(
0,
\frac{P_D(t)-30}{30}
\right)
\]

\[
CV_D=\sum_tCV_D(t)
\]

总违反量：

\[
\boxed{
CV=CV_R+CV_E+CV_D
}
\]

可行解条件：

\[
CV\le10^{-10}
\]

---

## 12. Fitness

\[
\boxed{
f=J+\lambda CV
}
\]

取：

\[
\lambda=10^6
\]

更新 `pbest` 和 `gbest` 时采用 feasibility-first：

1. 可行解优于不可行解；
2. 两个都可行时，总成本 \(J\) 较小者优；
3. 两个都不可行时，总违反量 \(CV\) 较小者优。

---

## 13. MPPSO 参数

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
velocity_clamp_q: 0.2

penalty_lambda: 1.0e6
numerical_epsilon: 1.0e-12
```

种群中心定义：

\[
p_{cen}^{k}
=
\frac{1}{I}
\sum_{i=1}^{I}p_i^k
\]

第一代历史状态：

\[
p_i^{-1}=p_i^0
\]

\[
v_i^{-1}=v_i^0
\]

---

## 14. MPPSO 更新

惯性权重：

\[
\omega_i=
\begin{cases}
\omega_{\min}
-(\omega_{\min}-\omega_{\max})
\dfrac{f_i-f_{\mathrm{mean}}}
{f_{\max}-f_{\mathrm{mean}}},
& f_i\ge f_{\mathrm{mean}}
\\[8pt]
\omega_{\min}
+(\omega_{\max}-\omega_{\min})
\dfrac{f_i-f_{\min}}
{f_{\mathrm{mean}}-f_{\min}},
& f_i<f_{\mathrm{mean}}
\end{cases}
\]

学习系数：

\[
c_1=
\begin{cases}
c_{\max}
+(c_{\max}-c_{\min})
\dfrac{f_i-f_{\min}}
{f_{\mathrm{mean}}-f_{\min}},
& f_i\le f_{\mathrm{mean}}
\\[8pt]
c_{\min},
& f_i>f_{\mathrm{mean}}
\end{cases}
\]

\[
c_2=2-c_1
\]

速度更新：

\[
\begin{aligned}
v_{i,j}^{k+1}
=&\ 
\omega_i
\left[
\zeta v_{i,j}^{k}
+(1-\zeta)v_{i,j}^{k-1}
\right]
\\
&+
c_1
\left[
l_{i,j}^{k}
-
\left(
\alpha p_{i,j}^{k}
+(1-\alpha)p_{i,j}^{k-1}
\right)
\right]
\\
&+
c_2
\left[
p_{g,j}^{k}
-
\left(
\alpha p_{i,j}^{k}
+(1-\alpha)p_{i,j}^{k-1}
\right)
\right]
\\
&+
c_3r_{3,j}
\left[
p_{cen,j}^{k}
-
\left(
\alpha p_{i,j}^{k}
+(1-\alpha)p_{i,j}^{k-1}
\right)
\right]
\end{aligned}
\]

其中：

\[
r_{3,j}\sim U(0,1)
\]

所有 fitness 分母使用：

```text
max(abs(denominator), 1e-12)
```

避免数值除零。

---

## 15. 求解流程

```text
输入 P_load[24]

初始化 600 个粒子：
X = [V(1:24), q(1:24)]

对每个粒子：
    q <- clip(q, 0, 1)
    V <- 投影到 [0,11] 且 sum(V)=240

    P_pr = 0.0022 * V^3
    P_D  = P_load + P_pr

    L = max(0, P_D - 20)
    U = min(10, P_D)

    Pg1 = L + q*(U-L)
    Pg2 = P_D - Pg1

    计算：
        J
        CV_R
        CV_E
        CV_D
        CV

    f = J + 1e6*CV

初始化 pbest、gbest

循环 500 代：
    计算 f_min、f_mean、f_max
    计算 p_cen

    对每个粒子：
        更新 omega
        更新 c1、c2
        更新粒子速度
        velocity clamp
        更新粒子位置

        q <- clip(q,0,1)
        V <- 航程投影

        重新解码 Pg1、Pg2
        重新计算 J、CV、f

        feasibility-first 更新 pbest

    feasibility-first 更新 gbest

输出最佳可行解
```

---

## 16. 输出结果

输出 24 h 调度结果：

```csv
hour,p_load_mw,v_kn,p_propulsion_mw,p_g1_mw,p_g2_mw,eeoi,cost_g1,cost_g2
```

同时输出：

\[
\boxed{J^*}
\]

即最优总运行成本。

并检查：

```text
sum(V) = 240
power balance residual ≈ 0
Pg1 ∈ [0,10]
Pg2 ∈ [0,20]
ramp G1 <= 2
ramp G2 <= 3
EEOI <= 23
CV ≈ 0
```