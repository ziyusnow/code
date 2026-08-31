Algorithm: RA-LSHADE for resilience-oriented ship energy scheduling

Input:
    System parameters:
        Generator limits and ramp-rate limits
        ESS power and energy limits
        PV forecast
        Vital and non-vital load profiles
        Propulsion model parameters
        Fault derating factor α_F
        Reserve horizon H_r

    Algorithm parameters:
        Maximum population size NP_max
        Minimum population size NP_min
        Maximum iteration number K
        Historical memory size H
        Initial ε-constraint threshold ε_0
        Resilience penalty bounds λ_r,min and λ_r,max

Output:
    Optimal normal-stage scheduling solution x_best
    Corresponding operating cost J_N
    ESS resilience reserve trajectory E_res(t, H_r)


1:  Initialize population P = {x_i | i = 1,2,...,NP_max}
2:  Each individual is encoded as

        x_i =
        [ P_g,2^N(1:T),
          P_E^N(1:T),
          V^N(1:T) ]

3:  Initialize historical memories

        M_F = 0.5
        M_CR = 0.5

4:  Initialize external archive A = ∅

5:  for each individual x_i do

6:      Calculate propulsion power

            P_pr^N(t) = ξ_1 [V^N(t)]^ξ_2

7:      Recover G1 output from the power balance equation

            P_g,1^N(t)
            =
            P_pr^N(t)
            + P_L(t)
            - P_PV(t)
            - P_g,2^N(t)
            - P_E^N(t)

8:      Calculate ESS energy trajectory E^N(t)

9:      Evaluate operating cost J_N(x_i)

10:     Calculate conventional constraint violation

            CV_base(x_i)

        including:
            generator power limits
            generator ramp-rate limits
            ESS power limits
            ESS energy limits
            voyage-distance constraint

11:     for each possible fault-starting time t do

12:         Calculate net post-fault demand

                P_D(τ)
                =
                P_pr^N(τ)
                + P_vi(τ)
                + P_nv(τ)
                - P_PV(τ)

13:         Calculate maximum available output of derated G1

14:         Calculate maximum emergency output of G2
            considering capacity and ramp-rate constraints

15:         Calculate required ESS reserve power

                P_E,res(τ)
                =
                [ P_D(τ)
                  - P̄_g,1^F(τ)
                  - P̄_g,2^F(τ) ]^+

16:         Calculate required ESS reserve energy

                E_res(t,H_r)
                =
                Σ P_E,res(τ) Δt / η_out

17:         Calculate resilience constraint violation

                CV_res(x_i)
                =
                Σ [ E_min + E_res(t,H_r) - E^N(t-1) ]^+

18:         Add ESS reserve-power violation

                CV_res(x_i)
                +=
                Σ [ P_E,res(τ) - P_E^max ]^+

19:     end for

20:     Calculate total constraint violation

            CV(x_i)
            =
            CV_base(x_i)
            + λ_r CV_res(x_i)

21: end for


22: for iteration k = 1 to K do

23:     Calculate feasible ratio

            r_f(k)
            =
            N_feasible / NP(k)

24:     Adapt ε-constraint threshold according to feasible ratio

25:     if r_f(k) is too low then
26:         increase ε(k)
27:     else if r_f(k) is sufficiently high then
28:         decrease ε(k)
29:     end if

30:     Update resilience weight

            λ_r(k)
            =
            λ_r,min
            +
            (λ_r,max - λ_r,min)
            (k / K)^β

31:     for each individual x_i do

32:         Randomly select one memory index r

33:         Generate adaptive mutation factor

                F_i ~ Cauchy(M_F[r], 0.1)

34:         Generate adaptive crossover rate

                CR_i ~ Normal(M_CR[r], 0.1)

35:         Select x_pbest randomly from the top p% individuals

36:         Select x_r1 from current population

37:         Select x_r2 from population ∪ archive

38:         Generate mutant vector using current-to-pbest/1

                v_i
                =
                x_i
                + F_i (x_pbest - x_i)
                + F_i (x_r1 - x_r2)

39:         Perform binomial crossover

                u_i = crossover(x_i, v_i, CR_i)

40:         Repair variable-bound violations of u_i

41:         Decode u_i

42:         Calculate:
                P_pr^N(t)
                P_g,1^N(t)
                E^N(t)

43:         Evaluate:
                J_N(u_i)
                CV_base(u_i)

44:         Perform virtual G1-derating analysis over reserve horizon H_r

45:         Calculate:
                P_E,res(τ)
                E_res(t,H_r)
                CV_res(u_i)

46:         Calculate total violation

                CV(u_i)
                =
                CV_base(u_i)
                + λ_r(k) CV_res(u_i)

47:         Compare u_i and x_i using adaptive ε-feasibility rule

48:         if CV(u_i) ≤ ε(k) and CV(x_i) > ε(k) then
49:             accept u_i
50:         else if both are ε-feasible then
51:             accept the one with lower J_N
52:         else if both are ε-infeasible then
53:             accept the one with lower CV
54:         end if

55:         if u_i replaces x_i then
56:             store F_i and CR_i in successful parameter sets
57:             add old x_i into archive A
58:         end if

59:     end for

60:     Update M_F using weighted Lehmer mean
        of successful F values

61:     Update M_CR using weighted arithmetic mean
        of successful CR values

62:     Linearly reduce population size

            NP(k)
            =
            round[
                NP_max
                -
                (k/K)(NP_max - NP_min)
            ]

63:     Remove worst individuals if population size exceeds NP(k)

64:     Maintain archive size

65:     Update global best feasible solution x_best

66: end for

67: Return x_best, J_N(x_best), E_res(t,H_r)
