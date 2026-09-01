Algorithm 0: Unified Preventive–Corrective RA-LSHADE for Case 3

Input:
    System parameters
    Load and PV profiles
    Voyage requirement D_req = 240 nm
    Reserve horizon H_r
    Fault start t_f
    Fault duration H_f
    G1 derating factor alpha_F
    Strategy in {no_reserve, dynamic_reserve}
    Strict feasibility tolerance tol = 1.0e-8
    Load-shedding penalty f_l = 1.0e4
    P1 RA-LSHADE parameters
    P2 RA-LSHADE parameters

Output:
    P1 optimal normal-stage schedule
    P2 optimal fault-stage schedule
    P1/P2 convergence histories
    Resilience metrics

Comparative execution:
    Run Algorithm 0 once for no_reserve and once for dynamic_reserve.
    Use paired random seeds and the same P2 fault scenario.
    Each P1 result enters the same P2 RA-LSHADE formulation.

1:  result_N = Solve_P1_RA_LSHADE(..., strategy, ...)

2:  Extract from result_N:
        P_g1^N(t)
        P_g2^N(t)
        P_E^N(t)
        E^N(t)
        V^N(t)
        P_pr^N(t)

3:  Construct P1 -> P2 coupling states:
        E_F0    = E^N(t_f - 1)
        P_g2_F0 = P_g2^N(t_f - 1)
        V^F(t)  = V^N(t),  t in T_F

4:  result_F = Solve_P2_RA_LSHADE(
        result_N,
        E_F0,
        P_g2_F0,
        V^F,
        t_f,
        H_f,
        alpha_F,
        ...
    )

    P2 is an offline full-horizon scenario optimization.
    The complete fault interval T_F is known to P2 and is solved once.

5:  Verify:
        P1 feasibility
        sum_t V^N(t) * Delta_t = 240 nm
        P1 reserve feasibility
        P2 fault feasibility
        ESS state continuity
        G2 transition constraint
        V^F(t) = V^N(t)

6:  Return result_N, result_F


------------------------------------------------------------
Algorithm P1: Preventive RA-LSHADE
------------------------------------------------------------

Input:
    Normal-stage system data
    Reserve parameters
    Strategy in {no_reserve, dynamic_reserve}
    NP_N_max, NP_N_min
    K_N
    SHADE memory size H
    epsilon_N_0

Decision vector:
    x_i^N =
    [
        P_g2^N(1:T),
        P_E^N(1:T),
        V^N(1:T)
    ]

Derived variables:
    P_pr^N(t) = xi_1 * [V^N(t)]^xi_2

    P_g1^N(t)
    =
    P_vi(t)
    + P_nv(t)
    + P_pr^N(t)
    - P_PV(t)
    - P_g2^N(t)
    - P_E^N(t)

    E^N(t) obtained recursively from P_E^N(t)

Objective:
    minimize J_N

Constraint violation:
    CV_N
    =
    CV_base
    +
    lambda_r(k) * CV_res

    If strategy = no_reserve:
        CV_res = 0

    If strategy = dynamic_reserve:
        CV_res equals the normalized dynamic-reserve violation
        accumulated over all valid hypothetical fault starts

1:  Initialize population P_N with NP_N_max individuals

2:  Repair variable bounds

3:  Project V^N so that:
        V_min <= V^N(t) <= V_max
        sum_t V^N(t) * Delta_t = 240 nm

4:  Initialize:
        M_F  = 0.5
        M_CR = 0.5
        Archive A_N = empty

5:  For each individual x_i^N:

6:      Decode P_g2^N, P_E^N, V^N

7:      Compute P_pr^N

8:      Recover P_g1^N from power balance

9:      Propagate E^N(t)

10:     Compute normal-stage objective J_N

11:     Compute CV_base:
            G1 limits
            G2 limits
            G1 ramps
            G2 ramps
            ESS power limits
            ESS energy limits
            voyage constraint
            other retained constraints

12:     For each hypothetical fault start t satisfying:
            t - 1 is in the normal-stage horizon
            t + H_r - 1 is in the normal-stage horizon

13:         Set reserve window:
                T_r = {t, ..., t + H_r - 1}

14:         Compute net demand:
                P_D(tau)
                =
                P_pr^N(tau)
                + P_vi(tau)
                + P_nv(tau)
                - P_PV(tau)

15:         Compute derated G1 capability:
                P_g1_bar^F(tau)
                =
                alpha_F * P_g1_max

16:         Compute emergency G2 capability:

                P_g2_bar^F(t)
                =
                min(
                    P_g2_max,
                    P_g2^N(t-1) + R_g2
                )

                For tau > t:
                    P_g2_bar^F(tau)
                    =
                    min(
                        P_g2_max,
                        P_g2_bar^F(tau-1) + R_g2
                    )

17:         Compute required ESS reserve power:
                P_E_res(tau)
                =
                max(
                    0,
                    P_D(tau)
                    - P_g1_bar^F(tau)
                    - P_g2_bar^F(tau)
                )

18:         Compute reserve energy:
                E_res(t, H_r)
                =
                sum_tau
                [
                    P_E_res(tau)
                    / eta_out
                    * Delta_t
                ]

19:         Compute reserve violations:
                energy deficit:
                    max(
                        0,
                        E_min
                        + E_res(t,H_r)
                        - E^N(t-1)
                    )
                    / (E_max - E_min)

                power deficit:
                    max(
                        0,
                        P_E_res(tau)
                        - P_E_dis_max
                    )
                    / P_E_dis_max

20:     End hypothetical fault loop

21:     Compute CV_res

22:     Compute:
            CV_N
            =
            CV_base
            +
            lambda_r(k) * CV_res

23: End initial evaluation

24: For generation k = 1 to K_N:

25:     Compute the strictly feasible ratio using tol

26:     If k >= ceil(0.8 * K_N):
            epsilon_N(k) = 0
        Else:
            update epsilon_N(k) using the adaptive feasibility rule

27:     Update lambda_r(k)

28:     For each target x_i^N:

29:         Sample memory index r

30:         Sample:
                F_i ~ Cauchy(M_F[r], 0.1)
                CR_i ~ Normal(M_CR[r], 0.1)

31:         Select x_pbest from top p%

32:         Select x_r1 from population

33:         Select x_r2 from population union archive

34:         Mutation:
                v_i
                =
                x_i
                + F_i * (x_pbest - x_i)
                + F_i * (x_r1 - x_r2)

35:         Binomial crossover:
                u_i = crossover(x_i, v_i, CR_i)

36:         Repair bounds

37:         Project speed to 240-nm constraint

38:         Decode u_i

39:         Compute P_pr^N

40:         Recover P_g1^N

41:         Propagate E^N

42:         Evaluate J_N(u_i)

43:         Evaluate CV_base(u_i)

44:         Perform virtual fault reserve calculation

45:         Evaluate CV_res(u_i) according to strategy

46:         Compute CV_N(u_i)

47:         Selection using epsilon-feasibility:

            If trial epsilon-feasible
            and target not:
                accept trial

            Else if both epsilon-feasible:
                accept lower J_N

            Else if both epsilon-infeasible:
                accept lower CV_N

48:         If trial accepted:
                store successful F_i, CR_i
                add replaced target to archive

49:     End individual loop

50:     Update M_F

51:     Update M_CR

52:     Linearly reduce population:
            NP_N(k)
            =
            round(
                NP_N_max
                -
                k/K_N
                * (NP_N_max - NP_N_min)
            )

53:     Limit archive size

54:     Update the best strictly feasible P1 solution:
            CV_base <= tol
            CV_res <= tol

55:     Store P1 convergence history

56: End generation loop

57: If no strictly feasible P1 solution was found:
        report P1 optimization failure

58: Return best strictly feasible P1 solution


------------------------------------------------------------
Algorithm P2: Corrective Fault-Stage RA-LSHADE
------------------------------------------------------------

Input:
    P1 optimal result
    Fault start t_f
    Fault duration H_f
    G1 derating alpha_F
    Strict feasibility tolerance tol = 1.0e-8
    Load-shedding penalty f_l = 1.0e4
    NP_F_max, NP_F_min
    K_F
    SHADE memory size H
    epsilon_F_0

Fault horizon:
    T_F =
    {t_f, ..., t_f + H_f - 1}

Execution semantics:
    Offline full-horizon scenario optimization.
    The complete T_F is known and optimized once.

Inherited states:
    E^F(t_f - 1)
    =
    E^N(t_f - 1)

    P_g2_pre
    =
    P_g2^N(t_f - 1)

    V^F(t)
    =
    V^N(t)

Decision vector:
    x_i^F =
    [
        P_g2^F(T_F),
        P_E^F(T_F),
        P_sh^F(T_F)
    ]

Derived variables:
    P_pr^F(t)
    =
    xi_1 * [V^N(t)]^xi_2

    P_g1^F(t)
    =
    P_vi(t)
    + P_nv(t)
    - P_sh^F(t)
    + P_pr^F(t)
    - P_PV(t)
    - P_g2^F(t)
    - P_E^F(t)

    E^F(t) obtained recursively from P_E^F(t)

Objective:
    minimize J_F

    J_F
    =
    sum_t in T_F
    [
        C_G1(P_g1^F(t))
        + C_G2(P_g2^F(t))
        + C_E(P_E^F(t))
        + f_l * P_sh^F(t) * Delta_t
    ]

Constraint violation:
    CV_F

    CV_F
    =
    sum_t
    [
        max(0, -P_g1^F(t)) / P_g1_max
        + max(0, P_g1^F(t) - alpha_F * P_g1_max) / P_g1_max
    ]
    + sum_{t > t_f}
      max(0, abs(P_g1^F(t) - P_g1^F(t-1)) - R_g1) / R_g1
    + max(0, abs(P_g2^F(t_f) - P_g2^N(t_f-1)) - R_g2) / R_g2
    + sum_{t > t_f}
      max(0, abs(P_g2^F(t) - P_g2^F(t-1)) - R_g2) / R_g2
    + sum_t
    [
        max(0, E_min - E^F(t))
        + max(0, E^F(t) - E_max)
    ] / (E_max - E_min)

    P_g2^F, P_E^F, and P_sh^F bounds are enforced by repair
    and are not counted again in CV_F.
    Vital-load shedding is structurally absent from the decision vector.

1:  Initialize population P_F with NP_F_max individuals

2:  Bounds:
        P_g2_min <= P_g2^F(t) <= P_g2_max

        -P_E_ch_max
        <= P_E^F(t)
        <= P_E_dis_max

        0
        <= P_sh^F(t)
        <= P_nv(t)

3:  Initialize:
        M_F  = 0.5
        M_CR = 0.5
        Archive A_F = empty

4:  For each initial individual x_i^F:

5:      Decode:
            P_g2^F
            P_E^F
            P_sh^F

6:      Set:
            E^F(t_f - 1)
            =
            E^N(t_f - 1)

7:      For each t in T_F:

8:          Compute P_pr^F(t)

9:          Recover:
                P_g1^F(t)
                =
                P_vi(t)
                + P_nv(t)
                - P_sh^F(t)
                + P_pr^F(t)
                - P_PV(t)
                - P_g2^F(t)
                - P_E^F(t)

10:         Update ESS energy:

                If P_E^F(t) >= 0:
                    E^F(t)
                    =
                    E^F(t-1)
                    -
                    P_E^F(t)
                    / eta_out
                    * Delta_t

                Else:
                    E^F(t)
                    =
                    E^F(t-1)
                    -
                    P_E^F(t)
                    * eta_in
                    * Delta_t

11:     End fault-period loop

12:     Compute J_F

13:     Compute CV_F including:

            G1 derated capacity:
                0
                <= P_g1^F(t)
                <= alpha_F * P_g1_max

            G1 post-fault ramps:
                |P_g1^F(t) - P_g1^F(t-1)|
                <= R_g1
                for t > t_f

            No normal-to-fault G1 ramp
            at t = t_f

            First fault-period G2 coupling:
                |P_g2^F(t_f)
                 - P_g2^N(t_f-1)|
                <= R_g2

            Subsequent G2 ramps:
                |P_g2^F(t)
                 - P_g2^F(t-1)|
                <= R_g2

            ESS energy:
                E_min
                <= E^F(t)
                <= E_max

            The repaired G2, ESS-power, and load-shedding bounds
            are validated separately but not added again to CV_F.

            Vital-load shedding remains structurally absent
            from the decision vector.

14: End initial population evaluation

15: Select initial best solution using feasibility-first rule

16: For generation k = 1 to K_F:

17:     Compute the strictly feasible ratio using tol

18:     If k >= ceil(0.8 * K_F):
            epsilon_F(k) = 0
        Else:
            update epsilon_F(k) using the adaptive feasibility rule

19:     For each target x_i^F:

20:         Sample memory index r

21:         Sample:
                F_i ~ Cauchy(M_F[r], 0.1)
                CR_i ~ Normal(M_CR[r], 0.1)

22:         Select x_pbest from top p%

23:         Select x_r1 from population

24:         Select x_r2 from population union archive

25:         Mutation:
                v_i
                =
                x_i
                + F_i * (x_pbest - x_i)
                + F_i * (x_r1 - x_r2)

26:         Binomial crossover:
                u_i
                =
                crossover(x_i, v_i, CR_i)

27:         Repair bounds

28:         Decode u_i

29:         Set inherited initial state:
                E^F(t_f-1)
                =
                E^N(t_f-1)

30:         For each t in T_F:

31:             Recover P_g1^F(t)

32:             Propagate E^F(t)

33:         End loop

34:         Evaluate J_F(u_i)

35:         Evaluate normalized CV_F(u_i)

36:         Selection using epsilon-feasibility:

            If trial epsilon-feasible
            and target not:
                accept trial

            Else if both epsilon-feasible:
                accept lower J_F

            Else if both epsilon-infeasible:
                accept lower CV_F

37:         If trial accepted:
                store successful F_i, CR_i
                add replaced target to archive

38:     End individual loop

39:     Update M_F

40:     Update M_CR

41:     Linearly reduce population:
            NP_F(k)
            =
            round(
                NP_F_max
                -
                k/K_F
                * (NP_F_max - NP_F_min)
            )

42:     Limit archive size

43:     Update the best strictly feasible P2 solution:
            CV_F <= tol

44:     Store:
            best J_F
            best CV_F
            feasible ratio
            total load shedding
            peak load shedding
            minimum ESS energy

45: End generation loop

46: If no strictly feasible P2 solution was found:
        report P2 optimization failure

47: Select best strictly feasible P2 solution

48: Calculate:
        E_sh
        =
        sum_t
        P_sh^F(t) * Delta_t

49: Calculate:
        R_load
        =
        1
        -
        E_sh
        /
        sum_t
        P_nv(t) * Delta_t

50: Return:
        P_g1^F
        P_g2^F
        P_E^F
        E^F
        P_sh^F
        J_F
        E_sh
        R_load
        P2 convergence history
