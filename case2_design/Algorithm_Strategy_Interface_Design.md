# Case 2 Algorithm Strategy and Interface Design

## 1. Purpose and Evidence Boundary

This document defines the search actions, selector interfaces, evaluation budget,
and experiment protocol for the existing Case 2 scheduling model.

The model and evaluator in `case2.md` and `solve_case2.py` remain the source of
truth. The strategy layer must not change:

- the 72-dimensional particle encoding
  \(X=[V(1{:}24),q_e(1{:}24),q_g(1{:}24)]\);
- the hourly EEOI constraint;
- ramp checks over adjacent in-horizon periods \(t=2,\ldots,24\);
- the absence of a terminal SOC equality constraint;
- the total violation definition
  \(CV=CV_R+CV_{SOC}+CV_E+CV_D\);
- feasibility-first comparison of candidate solutions.

All conclusions from the proposed experiments are restricted to the current,
fixed Case 2 load and PV instance. Thirty random seeds are repeated stochastic
optimization runs on one instance; they are not 30 independent operating
scenarios and do not establish cross-scenario robustness or general algorithmic
superiority.

The existing result obtained with seed `20260814`, cost `42433.447716`, and
total violation `CV=0` is a numerical regression reference for A1. It is not a
known global optimum.

## 2. Shared Search Semantics

### 2.1 Canonical particle representation

Every action returns a candidate population in the original 72-dimensional
encoding. Before evaluation, the outer loop applies one shared operation:

```text
canonicalize(X):
    require shape == (population_size, 72)
    require every value to be finite
    project V rows onto 0 <= V(t) <= 11 and sum_t V(t) = 240
    clip q_e and q_g to [0, 1]
    return canonical population
```

Power balance and voyage distance remain construction invariants. They are not
added to `CV`. An action must not directly edit decoded values such as
\(P_{G1}\), \(P_{G2}\), or \(P_{ESS}\), because those values cannot be changed
independently without breaking the particle encoding and coupled constraints.

### 2.2 Unified comparison rule

All population ranking, pbest updates, gbest updates, action targeting, and
survivor decisions use the same feasibility-first comparator:

1. A feasible solution is better than an infeasible solution.
2. Between feasible solutions, the lower-cost solution is better.
3. Between infeasible solutions, the lower-\(CV\) solution is better.
4. If two infeasible solutions have equal \(CV\), the lower-cost solution is
   better.

The penalized fitness

\[
F(X)=J(X)+10^6CV(X)
\]

is retained only for the adaptive MPPSO coefficient calculations already used
by the baseline. It is not the final comparison rule and is not an experiment
outcome.

### 2.3 Function-evaluation budget

One evaluation of one particle counts as one NFE, even when the evaluator is
called on a vectorized population. Initialization therefore costs 600 NFE and
each of 500 search iterations costs 600 NFE:

\[
B=600+500\times600=300600\ \text{NFE}.
\]

All four actions transform an unevaluated MPPSO base population and must not
call the evaluator internally. The outer loop evaluates the final population
exactly once per iteration. The single final evaluation used for independent
reporting or validation is recorded as an audit evaluation and is excluded from
the search budget.

This single-evaluation protocol intentionally studies a `DE-inspired` action
and an `elite-perturbation` action. A2 and A3 must not be described as complete
classical DE or a greedy local-search algorithm because they do not perform an
extra trial-versus-base evaluation.

### 2.4 Random-number streams

For a main seed \(s\), random-number ownership is fixed as follows:

```text
core_rng     = default_rng(s)
A2_rng       = default_rng(SeedSequence([s, 102]))
A3_rng       = default_rng(SeedSequence([s, 103]))
A4_rng       = default_rng(SeedSequence([s, 104]))
selector_rng = default_rng(SeedSequence([s, 200]))
```

Only `core_rng` is used by initialization and the MPPSO backbone. Each action
uses only its named stream, and selectors use only `selector_rng`. Thus, action
sampling cannot shift the random sequence used by the MPPSO core, and methods
with the same main seed share the initial population and core random arrays.

## 3. Search State and Iteration Flow

### 3.1 Internal state

`SearchState` is owned by the solver and is not directly exposed to selectors:

```text
SearchState:
    iteration
    search_nfe
    positions
    velocities
    previous_positions
    previous_velocities
    evaluation
    personal_best_positions
    personal_best_cost
    personal_best_cv
    global_best_position
    global_best_cost
    global_best_cv
    first_feasible_nfe
    active_action
    stagnation_iterations
```

Actions may return positions only. They must not mutate `SearchState`, evaluate
candidates, update pbest/gbest, or consume the MPPSO random stream.

### 3.2 Read-only selector summary

At the beginning of a decision window, the solver derives a read-only
`SelectorSummary` from the last evaluated population:

```text
SelectorSummary:
    iteration
    search_nfe
    budget_fraction
    phase                       # feasibility or cost
    global_best_cost
    global_best_cv
    cv_r, cv_soc, cv_e, cv_d
    feasible_fraction
    population_cv_q25
    population_cv_median
    population_cv_q75
    normalized_diversity
    stagnation_iterations
    previous_action
```

Normalized diversity is the mean per-dimension population standard deviation
divided by the corresponding encoding range (`11` for speed and `1` for both
allocation blocks), then averaged over all 72 dimensions. Selectors receive no
future evaluations and no writable optimizer arrays.

### 3.3 Main loop

The decision interval is 20 iterations. With 500 iterations, each run has 25
decision windows. The action chosen at a window boundary remains active through
the end of that window.

```text
initialize SearchState and evaluate 600 particles

for k = 1, ..., 500:
    if k starts a 20-iteration decision window:
        summary = make_selector_summary(state)
        decision = selector.select(summary, selector_rng)
        action = decision.applied_action

    X_base, V_base = mppso_step(state, core_rng)
    X_action = apply_action(action, X_base, state, action_rng)
    X_new = canonicalize(X_action)
    result = evaluate(X_new)
    search_nfe += population_size

    advance position and velocity history using X_new and V_base
    update pbest and gbest with the shared comparator
    append iteration and action history

    if k ends the decision window:
        outcome = summarize_window(start_snapshot, state)
        selector.observe(action, outcome)
```

When an action replaces an MPPSO position, the associated row of `V_base` is
still retained as the velocity for the next iteration. The action-induced jump
is not reinterpreted as velocity. `previous_positions` and
`previous_velocities` advance in the same order as the A1 baseline.

## 4. Fixed Search Actions

Actions select their target row indices using the previous evaluated
population and the shared comparator. They then transform corresponding rows in
the unevaluated `X_base` population.

### 4.1 A1 - MPPSO identity

```text
strategy_A1(X_base, state, rng):
    return X_base
```

A1 consumes no action random numbers and must reproduce the existing MPPSO
baseline under the same seed and numerical environment.

### 4.2 A2 - DE-inspired global exploration

A2 targets the worst 25% of the population, exactly 150 rows when the population
size is 600. For every target row \(i\), select indices \(r_1,r_2,r_3\) from the
full base population such that all four indices are different. Sampling is
without replacement for each target.

With \(F=0.5\), form

\[
D_i=X^{base}_{r_1}+F\left(X^{base}_{r_2}-X^{base}_{r_3}\right).
\]

Use binomial crossover with \(CR=0.9\):

\[
U_{i,j}=\begin{cases}
D_{i,j}, & u_j<CR\ \text{or}\ j=j_{rand},\\
X^{base}_{i,j}, & \text{otherwise},
\end{cases}
\]

where \(j_{rand}\) is sampled uniformly from the 72 dimensions. The forced
dimension guarantees that every target receives at least one donor component.
Replace the target row in the action population with \(U_i\). Do not evaluate
or greedily compare \(U_i\) with \(X^{base}_i\) inside A2.

### 4.3 A3 - Elite perturbation

A3 targets the best 15% of the previous evaluated population, exactly 90 rows.
Each target receives one perturbation around the current gbest:

\[
X_i^{trial}=X_{gbest}+\sigma(p)S\odot N(0,I),
\]

where

\[
S=[\underbrace{11,\ldots,11}_{24},
   \underbrace{1,\ldots,1}_{24},
   \underbrace{1,\ldots,1}_{24}],
\]

\[
\sigma(p)=\sigma_{min}+(\sigma_0-\sigma_{min})(1-p),
\quad \sigma_0=0.05,\quad \sigma_{min}=0.005,
\]

and \(p=\min(1,\text{search\_nfe}/B)\). Replace the target rows with these
trials. Do not perform an internal greedy comparison.

### 4.4 A4 - CV-guided feasibility search

A4 targets the worst 25% of the previous evaluated population, exactly 150
rows. Its reference point is:

- the lowest-cost feasible pbest if at least one feasible pbest exists;
- otherwise, the lowest-\(CV\) pbest, using cost as the tie-breaker.

For each target, independently sample \(\beta_i\sim U(0.2,0.8)\) and form

\[
X_i^{trial}=X_i^{base}+\beta_i\left(X_{ref}-X_i^{base}\right).
\]

Replace the target rows with the trials. A4 performs no sequential ramp repair,
no direct power clipping, no inverse decoding, and no internal evaluation.

## 5. Selector Interfaces

Every selector implements:

```text
select(summary: SelectorSummary, rng) -> SelectionDecision
observe(action, outcome: StrategyOutcome) -> None
```

`SelectionDecision` records the requested action, the action actually applied,
whether a fallback was used, the error category, selector elapsed time, and an
optional LLM call identifier. Local selector configuration, lifecycle, replay,
and implementation errors propagate and fail the run. Only the documented LLM
provider, JSON, schema, and invalid-action failures are converted to a
phase-aware fallback.

`StrategyOutcome` contains the phase at window start, start/end NFE, start/end
best cost and CV, whether the first feasible solution appeared, relative
improvement, selector fallback status, and elapsed selector time.

### 5.1 Manual debug selector

Manual mode accepts either one fixed action or a predeclared 25-action sequence.
It is used for debugging and replay only and is excluded from the formal method
comparison.

### 5.2 Uniform random selector

At each window boundary, select A1-A4 with probability \(0.25\) using
`selector_rng`.

### 5.3 Rule selector

Apply the first matching rule in this fixed order:

1. If no feasible pbest exists, select A4.
2. If at least 80% of the NFE budget has been used, select A3.
3. If a feasible pbest exists and the best feasible cost has not improved for
   20 consecutive iterations, select A2.
4. Otherwise, select A1.

An improvement means a strict reduction in the stored best feasible cost under
the same floating-point `<` comparison used by the pbest/gbest comparator. No
percentage-improvement threshold is applied to the Rule selector.

### 5.4 Phase-aware UCB1 selector

UCB1 maintains separate statistics for two permanent phases:

- `feasibility`: no feasible pbest has yet been found;
- `cost`: begins after the first feasible pbest and never returns to
  `feasibility`, because pbest/gbest preserve the feasible solution.

Within each phase, actions are initialized in the deterministic order
`A1,A2,A3,A4`, once each. Later selections maximize

\[
\bar r_a+\sqrt{\frac{2\ln n}{n_a}},
\]

with ties resolved by action order.

For a window that starts in the feasibility phase,

\[
r=clip\left(
\frac{CV_{start}-CV_{end}}{\max(CV_{start},10^{-12})},-1,1
\right).
\]

If the first feasible solution appears during that window, the reward is first
recorded in the feasibility statistics. At the next window boundary those
statistics are archived and a new cost-phase UCB1 instance is initialized.

For a window that starts in the cost phase,

\[
r=clip\left(
\frac{J_{start}-J_{end}}{\max(|J_{start}|,10^{-12})},-1,1
\right).
\]

Mixing feasibility and cost rewards in one action mean is prohibited.

### 5.5 LLM-E and LLM-EP selectors

Both LLM selectors are called once at each decision boundary and must return the
structured object

```json
{"action": "A1"}
```

with `action` restricted to `A1`, `A2`, `A3`, or `A4`.

- `LLM-E` receives only the current `SelectorSummary` and short, fixed
  definitions of the four actions.
- `LLM-EP` receives the same input plus current-phase action counts, mean
  rewards, feasibility-transition counts, invalid-output counts, and fallback
  counts. Archived statistics from a previous phase are labelled separately and
  are not mixed into current-phase means.

Formal LLM experiments require the following fields to be filled in the
experiment configuration before execution:

```text
provider
model_id
model_version_or_snapshot
prompt_version
temperature = 0
response_schema_version
timeout_seconds
```

An unspecified field means that only non-LLM methods may run.
Each decision permits one call and no automatic retry. A timeout, provider
error, schema error, or invalid action triggers:

- A4 if no feasible pbest exists;
- A1 otherwise.

The run continues, and the fallback and its reason are retained as selector
failures. LLM quality is compared under the same NFE budget; wall time, number
of calls, latency, token usage, and provider cost are reported separately.

Diagnostic `without_A2`, `without_A3`, and `without_A4` variants remove the
unavailable action from the prompt definitions, allowed-action payload, JSON
schema, and response parser. If A4 is unavailable during the feasibility phase,
the fallback is A1. The provider implementation is injected at runtime and is
never serialized; provider name and model/version metadata remain in the run
configuration.

Every call record contains the request, raw response, parsed action, model
fields, timing, token counts, and fallback reason. Replay mode consumes the
recorded action sequence without contacting the provider and is used only to
verify downstream optimizer reproducibility, not as a separate primary method.

## 6. Research Questions and Methods

### 6.1 Research questions

- **RQ1:** Do A2, A3, or A4 change final feasibility and cost relative to A1 on
  the current Case 2 instance?
- **RQ2:** Does adaptive selection outperform A1 and uniform random selection
  under the same NFE budget?
- **RQ3:** How do Rule, UCB1, LLM-E, and LLM-EP differ in feasibility, cost, and
  selector overhead?
- **RQ4:** Does action-performance history supplied to LLM-EP provide evidence
  of incremental benefit over LLM-E?

### 6.2 Formal methods

The formal experiment contains nine methods:

```text
Stage 1: A1-only, A2-only, A3-only, A4-only
Stage 2: UniformRandom, Rule, UCB1, LLM-E, LLM-EP
```

Stage 1 is completed and the four action implementations are frozen before any
Stage 2 result is interpreted. This separates action quality from selector
quality. `Manual` is not a formal method.

## 7. Experiment Protocol

### 7.1 Validation and formal seeds

- Seed `20260814` is used only for A1 regression against the existing result.
- Seeds `300001..300005` are implementation-validation runs for output,
  runtime, failure handling, and NFE accounting. They are not used to choose
  parameters or in inferential statistics.
- Seeds `310001..310030` are the 30 paired formal seeds for all nine methods.
- Seeds `320001..320010` are reserved for descriptive ablation and sensitivity
  analysis and are not mixed with formal results.

The corresponding default task counts are 1 baseline run, 45 validation runs,
270 formal runs, and 170 diagnostic runs. Reports use the manifest task count
for their denominator rather than assuming every group contains 30 seeds.

No formal seed may be replaced. An infrastructure interruption may be rerun
with exactly the same method, configuration, and seed. Optimization failure,
numerical failure, selector failure, or an LLM fallback is part of the method
outcome and must not be removed or replaced.

Thirty seeds are a fixed sample size, not a claim of guaranteed statistical
power. Confidence intervals and the observed number of paired outcomes must be
reported with every inferential result.

### 7.2 Outcomes

The primary outcomes are hierarchical:

1. Whether a valid feasible solution is found within `300600 NFE`.
2. Final best feasible cost, compared only on seeds where both paired methods
   found valid feasible solutions.

This two-part reporting prevents a method with a high failure rate from looking
competitive merely because failed runs are omitted. Penalized fitness is never
used as a substitute for economic cost.

Secondary outcomes are:

- first feasible NFE;
- final total CV and `CV_R`, `CV_SOC`, `CV_E`, `CV_D` for failed runs;
- best feasible cost at 25%, 50%, 75%, and 100% of the NFE budget;
- pairwise area under the best-so-far cost curve, starting at the later of the
  two methods' first-feasible NFE and ending at `300600 NFE`, divided by that
  NFE interval; this is reported only for jointly feasible seed pairs;
- wall-clock search time excluding and including selector overhead;
- total emissions of the final valid schedule;
- action frequencies and phase-specific action rewards;
- selector invalid-output, fallback, timeout, call, token, and cost totals.

### 7.3 Statistical analysis

For every method, report the feasible-run count and Wilson 95% confidence
interval. For each paired comparison, use exact McNemar testing for feasibility.

On jointly feasible seeds, define every directional difference as
`candidate - reference`, where the candidate is the left-hand method and the
reference is the right-hand method in the preregistered comparison list. Report:

- both methods' median and IQR;
- paired median absolute cost difference;
- paired median relative cost difference;
- paired percentile-bootstrap 95% confidence intervals for both paired
  differences using 10,000 paired resamples and bootstrap seed `330001`;
- a two-sided paired Wilcoxon signed-rank test with zero differences discarded;
- rank-biserial correlation as effect size;
- paired win/tie/loss counts, treating absolute cost differences no greater
  than `1e-6` as ties.

If fewer than 10 jointly feasible seed pairs are available, report descriptive
results only and do not make a significance claim for cost.

The preregistered comparison family is:

```text
A2-only       vs A1-only
A3-only       vs A1-only
A4-only       vs A1-only
UniformRandom vs A1-only
Rule          vs UniformRandom
UCB1          vs UniformRandom
UCB1          vs A1-only
LLM-E         vs UCB1
LLM-EP        vs LLM-E
LLM-EP        vs UCB1
```

Apply Holm correction at family-wise `alpha=0.05` across these ten comparisons
separately for feasibility and cost endpoints. Other pairwise comparisons are
exploratory and labelled as such.

No minimum practical-effect threshold is imposed. Reports state the observed
effect, interval, and corrected statistical evidence; they must not claim
"practical significance", cross-scenario robustness, global optimality, or
universal superiority.

### 7.4 Diagnostic ablation and sensitivity

Using only seeds `320001..320010`, report descriptive results for:

- LLM-EP with A2, A3, or A4 individually unavailable;
- A2 with \(F\in\{0.3,0.5,0.7\}\),
  \(CR\in\{0.7,0.9\}\), and target fraction
  \(\{0.20,0.25,0.30\}\), varied one factor at a time;
- A3 with \(\sigma_0\in\{0.02,0.05,0.10\}\) and target fraction
  \(\{0.10,0.15,0.20\}\), varied one factor at a time;
- Rule and selector decision intervals of 10, 20, and 40 iterations.

Sensitivity results must not be used to retune the already completed formal
experiment and then re-report it as confirmatory evidence.

## 8. Reproducible Outputs

All future implementation, configuration, tests, logs, and generated results
for this design must stay under `case2_design`. A run uses the layout:

```text
case2_design/results/<experiment_id>/manifest.json
case2_design/results/<experiment_id>/<method>/seed_<seed>/
    config.json
    summary.json
    history.csv
    actions.jsonl
    schedule.csv
    validation.json
    llm_calls.jsonl        # LLM methods only
    attempts.jsonl         # only when an attempt fails before completion
```

The manifest is written before any task starts and lists the complete requested
task matrix with `pending`, `running`, completed, or failed status. Therefore an
interrupted launch cannot make unstarted jobs disappear from the denominator.
`--resume` reruns an incomplete seed from initialization; there are no
within-window checkpoints.

`history.csv` records at least:

```text
iteration, search_nfe, action, phase,
best_cost, best_cv, feasible, feasible_fraction,
cv_r, cv_soc, cv_e, cv_d,
median_cv, normalized_diversity,
improved, stagnation_iterations, selector_fallback
```

Every numerical schedule field retains at least 12 decimal places so that an
independent check is not invalidated by export rounding. Aggregate tables and
figures must be recomputable from per-run files. Failures remain present in the
aggregate data with an explicit failure category.

## 9. Verification and Acceptance Criteria

### 9.1 Model and operator tests

- `canonicalize` rejects wrong shapes and non-finite inputs.
- Every action returns finite `(600,72)` output with projected voyage distance
  and bounded allocation variables.
- A1 is the identity action and consumes no action RNG.
- A2 donors are mutually distinct, exclude the target, and crossover always
  copies at least one donor dimension.
- A3 uses the declared scale vector and never reaches zero perturbation scale.
- A4 handles an empty feasible set by selecting the minimum-CV pbest and never
  edits decoded powers directly.
- No action calls the evaluator.
- Power balance remains exact by decoding; ramp checks do not introduce a
  period-zero boundary; terminal SOC equality is not added.

### 9.2 Solver and selector tests

- Initialization uses 600 NFE, each search iteration uses 600 NFE, and every
  complete run ends at exactly `300600` search NFE.
- The same method, seed, and configuration reproduce the same non-LLM action
  history and numerical result in the same environment.
- Additional action RNG draws do not change the MPPSO core random sequence.
- Documented LLM provider and response failures use the phase-aware fallback
  and are recorded without replacing the seed; local selector and replay
  implementation errors propagate instead of being silently converted.
- UCB1 archives feasibility rewards and starts fresh cost-phase statistics at
  the first window after feasibility.
- LLM replay reproduces downstream optimization from the recorded actions.

### 9.3 Baseline and result validation

- With the default A1 configuration and seed `20260814`, total cost is
  `42433.447716` within `1e-6` and total CV is within the existing feasibility
  tolerance.
- Every result labelled feasible passes the existing independent checks for
  distance, power balance, ESS energy recursion, SOC, ramp, and hourly EEOI.
- Formal reports include all 30 seeds per method or explicitly identify the
  unrecovered infrastructure record; algorithm and selector failures are never
  silently dropped.
- Summary statistics can be regenerated from the per-run records without
  hidden manual corrections.

No hash, frozen contract, or release gate is added. Ordinary numerical
regression, deterministic unit tests, independent constraint checks, and
recomputable experiment summaries are sufficient for this reversible,
single-directory research workflow.
