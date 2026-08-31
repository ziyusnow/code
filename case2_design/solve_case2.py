from pathlib import Path
import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .experiment_types import (
        Action,
        ActionParameters,
        Phase,
        RandomStreams,
        SearchConfig,
        SearchRun,
        SearchState,
        SelectionDecision,
        SelectorSummary,
        StrategyOutcome,
    )
except ImportError:
    from experiment_types import (
        Action,
        ActionParameters,
        Phase,
        RandomStreams,
        SearchConfig,
        SearchRun,
        SearchState,
        SelectionDecision,
        SelectorSummary,
        StrategyOutcome,
    )


HOURS = 24
POPULATION_SIZE = 600
MAX_ITERATIONS = 500
SEED = 20260814

OMEGA_MAX = 0.9
OMEGA_MIN = 0.4
C_MAX = 1.5
C_MIN = 0.5
ALPHA = 0.8
ZETA = 0.8
C3 = -0.4

PENALTY_LAMBDA = 1.0e6
FEASIBILITY_TOLERANCE = 1.0e-10
NUMERICAL_EPSILON = 1.0e-12
EEOI_VELOCITY_EPSILON = 1.0e-6

DISTANCE = 240.0
V_MAX = 11.0
VELOCITY_CLAMP_V = 2.2
VELOCITY_CLAMP_QE = 0.2
VELOCITY_CLAMP_QG = 0.2

ESS_ENERGY_INITIAL = 37.5
ESS_ENERGY_MIN = 15.0
ESS_ENERGY_MAX = 75.0
ESS_EFFICIENCY_IN = 0.95
ESS_EFFICIENCY_OUT = 0.95


def load_input_data(path):
    service_rows = []
    pv_rows = []
    service_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$"
    )
    pv_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*$"
    )

    for line in path.read_text(encoding="utf-8").splitlines():
        service_match = service_pattern.match(line)
        if service_match:
            hour, vital, nonvital = service_match.groups()
            service_rows.append((int(hour), float(vital), float(nonvital)))
            continue

        pv_match = pv_pattern.match(line)
        if pv_match:
            hour, pv = pv_match.groups()
            pv_rows.append((int(hour), float(pv)))

    expected_hours = list(range(HOURS))
    if [row[0] for row in service_rows] != expected_hours:
        raise ValueError("The service-load table must contain hours 0 through 23")
    if [row[0] for row in pv_rows] != expected_hours:
        raise ValueError("The PV table must contain hours 0 through 23")

    hours = np.array(expected_hours, dtype=int)
    p_vital = np.array([row[1] for row in service_rows], dtype=float)
    p_nonvital = np.array([row[2] for row in service_rows], dtype=float)
    p_pv = np.array([row[1] for row in pv_rows], dtype=float)

    if not all(np.isfinite(values).all() for values in (p_vital, p_nonvital, p_pv)):
        raise ValueError("Input data contains NaN or infinite values")
    if np.any(p_vital < 0.0) or np.any(p_nonvital < 0.0):
        raise ValueError("Service loads cannot be negative")
    if np.any((p_pv < 0.0) | (p_pv > 4.2)):
        raise ValueError("PV power must remain within [0, 4.2] MW")

    return hours, p_vital, p_nonvital, p_pv


def project_speeds(values):
    """Project speed rows onto 0 <= V <= 11 and sum(V) = 240."""
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1
    rows = values.reshape(1, -1) if one_dimensional else values
    if rows.ndim != 2 or rows.shape[1] != HOURS:
        raise ValueError("Each speed vector must contain 24 values")

    lower = np.min(rows - V_MAX, axis=1)
    upper = np.max(rows, axis=1)
    for _ in range(60):
        tau = (lower + upper) / 2.0
        sums = np.clip(rows - tau[:, None], 0.0, V_MAX).sum(axis=1)
        lower = np.where(sums > DISTANCE, tau, lower)
        upper = np.where(sums > DISTANCE, upper, tau)

    projected = np.clip(rows - ((lower + upper) / 2.0)[:, None], 0.0, V_MAX)
    return projected[0] if one_dimensional else projected


def evaluate(
    positions,
    p_vital,
    p_nonvital,
    p_pv,
    penalty_lambda=PENALTY_LAMBDA,
):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    if positions.ndim != 2 or positions.shape[1] != 3 * HOURS:
        raise ValueError("Each Case 2 particle must contain 72 values")

    speeds = positions[:, :HOURS]
    q_ess = positions[:, HOURS : 2 * HOURS]
    q_generator = positions[:, 2 * HOURS :]

    p_load = p_vital + p_nonvital
    p_propulsion = 0.0022 * speeds**3
    net_demand = p_load[None, :] + p_propulsion - p_pv[None, :]

    lower_ess = np.maximum(-3.0, net_demand - 30.0)
    upper_ess = np.minimum(3.0, net_demand)
    p_ess = lower_ess + q_ess * (upper_ess - lower_ess)

    cv_d_hourly = np.maximum(0.0, (net_demand - 33.0) / 33.0) + np.maximum(
        0.0, (-3.0 - net_demand) / 3.0
    )
    cv_d = cv_d_hourly.sum(axis=1)

    p_generator = net_demand - p_ess
    lower_g1 = np.maximum(0.0, p_generator - 20.0)
    upper_g1 = np.minimum(10.0, p_generator)
    p_g1 = lower_g1 + q_generator * (upper_g1 - lower_g1)
    p_g2 = p_generator - p_g1

    energy_change = np.where(
        p_ess <= 0.0,
        -p_ess * ESS_EFFICIENCY_IN,
        -p_ess / ESS_EFFICIENCY_OUT,
    )
    energy_ess = ESS_ENERGY_INITIAL + np.cumsum(energy_change, axis=1)
    soc = energy_ess / ESS_ENERGY_MAX
    cv_soc_hourly = np.maximum(
        0.0, (ESS_ENERGY_MIN - energy_ess) / 60.0
    ) + np.maximum(0.0, (energy_ess - ESS_ENERGY_MAX) / 60.0)
    cv_soc = cv_soc_hourly.sum(axis=1)

    ramp_g1 = np.abs(np.diff(p_g1, axis=1))
    ramp_g2 = np.abs(np.diff(p_g2, axis=1))
    ramp_ess = np.abs(np.diff(p_ess, axis=1))
    cv_r = (
        np.maximum(0.0, (ramp_g1 - 2.0) / 2.0).sum(axis=1)
        + np.maximum(0.0, (ramp_g2 - 3.0) / 3.0).sum(axis=1)
        + np.maximum(0.0, ramp_ess - 1.0).sum(axis=1)
    )

    emissions_g1 = 13.5 * p_g1**2 + 10.0 * p_g1 + 450.0
    emissions_g2 = 5.2 * p_g2**2 + 58.0 * p_g2 + 390.0
    eeoi = (emissions_g1 + emissions_g2) / (
        20.0 * np.maximum(speeds, EEOI_VELOCITY_EPSILON)
    )
    cv_e_hourly = np.maximum(0.0, (eeoi - 23.0) / 23.0)
    cv_e = cv_e_hourly.sum(axis=1)

    cost_g1 = 13.0 * p_g1**2 + 12.0 * p_g1 + 430.0
    cost_g2 = 5.2 * p_g2**2 + 52.0 * p_g2 + 340.0
    cost_ess = 4.3 * p_ess**2 + 1.0
    cost_pv = 10.2 * p_pv[None, :]
    total_cost = (cost_g1 + cost_g2 + cost_ess + cost_pv).sum(axis=1)

    cv = cv_r + cv_soc + cv_e + cv_d
    fitness = total_cost + penalty_lambda * cv

    return {
        "speeds": speeds,
        "q_ess": q_ess,
        "q_generator": q_generator,
        "p_load": p_load,
        "p_propulsion": p_propulsion,
        "net_demand": net_demand,
        "p_ess": p_ess,
        "p_generator": p_generator,
        "p_g1": p_g1,
        "p_g2": p_g2,
        "energy_ess": energy_ess,
        "soc": soc,
        "emissions_g1": emissions_g1,
        "emissions_g2": emissions_g2,
        "eeoi": eeoi,
        "cost_g1": cost_g1,
        "cost_g2": cost_g2,
        "cost_ess": cost_ess,
        "cost_pv": np.broadcast_to(cost_pv, speeds.shape),
        "total_emissions": (emissions_g1 + emissions_g2).sum(axis=1),
        "total_cost": total_cost,
        "cv_r": cv_r,
        "cv_soc": cv_soc,
        "cv_e": cv_e,
        "cv_d": cv_d,
        "cv": cv,
        "fitness": fitness,
    }


def is_better(
    cost,
    cv,
    reference_cost,
    reference_cv,
    feasibility_tolerance=FEASIBILITY_TOLERANCE,
):
    feasible = cv <= feasibility_tolerance
    reference_feasible = reference_cv <= feasibility_tolerance
    both_infeasible = ~feasible & ~reference_feasible
    return (
        (feasible & ~reference_feasible)
        | (feasible & reference_feasible & (cost < reference_cost))
        | (both_infeasible & (cv < reference_cv))
        | (both_infeasible & (cv == reference_cv) & (cost < reference_cost))
    )


def best_index(cost, cv, feasibility_tolerance=FEASIBILITY_TOLERANCE):
    feasible = cv <= feasibility_tolerance
    if np.any(feasible):
        candidates = np.flatnonzero(feasible)
        return int(candidates[np.argmin(cost[candidates])])
    return int(np.lexsort((cost, cv))[0])


def protected_denominator(value):
    return max(abs(value), NUMERICAL_EPSILON)


def default_search_config():
    return SearchConfig(
        hours=HOURS,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        distance=DISTANCE,
        speed_max=V_MAX,
        velocity_clamp_v=VELOCITY_CLAMP_V,
        velocity_clamp_qe=VELOCITY_CLAMP_QE,
        velocity_clamp_qg=VELOCITY_CLAMP_QG,
        omega_max=OMEGA_MAX,
        omega_min=OMEGA_MIN,
        c_max=C_MAX,
        c_min=C_MIN,
        alpha=ALPHA,
        zeta=ZETA,
        c3=C3,
        penalty_lambda=PENALTY_LAMBDA,
        feasibility_tolerance=FEASIBILITY_TOLERANCE,
        numerical_epsilon=NUMERICAL_EPSILON,
    )


def validate_experiment_configuration(config, parameters):
    if config.hours != HOURS or config.distance != DISTANCE or config.speed_max != V_MAX:
        raise ValueError("Case 2 encoding constants cannot be changed")
    if config.population_size < 4:
        raise ValueError("Case 2 actions require at least four particles")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.decision_interval <= 0:
        raise ValueError("decision_interval must be positive")
    if not np.isfinite(config.penalty_lambda) or config.penalty_lambda <= 0.0:
        raise ValueError("penalty_lambda must be finite and positive")
    if (
        not np.isfinite(config.feasibility_tolerance)
        or config.feasibility_tolerance < 0.0
    ):
        raise ValueError("feasibility_tolerance must be finite and nonnegative")
    if not np.isfinite(config.numerical_epsilon) or config.numerical_epsilon <= 0.0:
        raise ValueError("numerical_epsilon must be finite and positive")

    fractions = (
        parameters.a2_target_fraction,
        parameters.a3_target_fraction,
        parameters.a4_target_fraction,
    )
    if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("Action target fractions must be finite and within (0, 1]")
    if not 0.0 <= parameters.a2_crossover_probability <= 1.0:
        raise ValueError("A2 crossover probability must be within [0, 1]")
    if not np.isfinite(parameters.a2_differential_weight):
        raise ValueError("A2 differential weight must be finite")
    if not (
        0.0 < parameters.a3_sigma_min <= parameters.a3_sigma_0
        and np.isfinite(parameters.a3_sigma_0)
    ):
        raise ValueError("A3 sigma values must satisfy 0 < sigma_min <= sigma_0")
    if not (
        0.0 <= parameters.a4_beta_min <= parameters.a4_beta_max <= 1.0
    ):
        raise ValueError("A4 beta bounds must lie within [0, 1]")


def canonicalize(positions, config=None):
    """Return a finite Case 2 population in the canonical 72-D encoding."""
    config = config or default_search_config()
    values = np.asarray(positions, dtype=float)
    expected_shape = (config.population_size, config.dimensions)
    if values.shape != expected_shape:
        raise ValueError(f"Case 2 population must have shape {expected_shape}")
    if not np.isfinite(values).all():
        raise ValueError("Case 2 population contains NaN or infinite values")
    canonical = values.copy()
    canonical[:, :HOURS] = project_speeds(canonical[:, :HOURS])
    canonical[:, HOURS : 2 * HOURS] = np.clip(
        canonical[:, HOURS : 2 * HOURS], 0.0, 1.0
    )
    canonical[:, 2 * HOURS :] = np.clip(canonical[:, 2 * HOURS :], 0.0, 1.0)
    return canonical


def rank_indices(cost, cv, feasibility_tolerance=FEASIBILITY_TOLERANCE):
    """Rank rows from best to worst with the shared feasibility-first rule."""
    cost = np.asarray(cost, dtype=float)
    cv = np.asarray(cv, dtype=float)
    if cost.ndim != 1 or cv.shape != cost.shape:
        raise ValueError("Cost and CV must be one-dimensional arrays of equal length")
    indices = np.arange(cost.size)
    feasible = cv <= feasibility_tolerance
    group = (~feasible).astype(int)
    primary = np.where(feasible, cost, cv)
    secondary = np.where(feasible, 0.0, cost)
    return np.lexsort((indices, secondary, primary, group))


def make_random_streams(seed):
    return RandomStreams(
        core=np.random.default_rng(seed),
        a2=np.random.default_rng(np.random.SeedSequence([seed, 102])),
        a3=np.random.default_rng(np.random.SeedSequence([seed, 103])),
        a4=np.random.default_rng(np.random.SeedSequence([seed, 104])),
        selector=np.random.default_rng(np.random.SeedSequence([seed, 200])),
    )


def _cv_components(evaluation):
    return np.column_stack(
        (
            evaluation["cv_r"],
            evaluation["cv_soc"],
            evaluation["cv_e"],
            evaluation["cv_d"],
        )
    )


def _global_components(components, index):
    return tuple(float(value) for value in components[index])


def initialize_search(evaluator, core_rng, config=None):
    config = config or default_search_config()
    speeds = project_speeds(
        core_rng.uniform(0.0, V_MAX, (config.population_size, HOURS))
    )
    q_ess = core_rng.uniform(0.0, 1.0, (config.population_size, HOURS))
    q_generator = core_rng.uniform(0.0, 1.0, (config.population_size, HOURS))
    positions = np.hstack((speeds, q_ess, q_generator))
    velocities = np.zeros_like(positions)
    evaluation = evaluator(positions)
    personal_best_cost = evaluation["total_cost"].copy()
    personal_best_cv = evaluation["cv"].copy()
    personal_best_components = _cv_components(evaluation)
    global_index = best_index(
        personal_best_cost,
        personal_best_cv,
        config.feasibility_tolerance,
    )
    global_cv = float(personal_best_cv[global_index])

    return SearchState(
        iteration=0,
        search_nfe=config.population_size,
        positions=positions,
        velocities=velocities,
        previous_positions=positions.copy(),
        previous_velocities=velocities.copy(),
        evaluation=evaluation,
        personal_best_positions=positions.copy(),
        personal_best_cost=personal_best_cost,
        personal_best_cv=personal_best_cv,
        personal_best_cv_components=personal_best_components,
        global_best_position=positions[global_index].copy(),
        global_best_cost=float(personal_best_cost[global_index]),
        global_best_cv=global_cv,
        global_best_cv_components=_global_components(
            personal_best_components, global_index
        ),
        first_feasible_nfe=(
            config.population_size
            if global_cv <= config.feasibility_tolerance
            else None
        ),
        active_action=None,
        stagnation_iterations=0,
    )


def mppso_step(state, core_rng, config=None):
    """Produce an uncanonicalized MPPSO base population and its velocity."""
    config = config or default_search_config()
    fitness = state.evaluation["fitness"]
    f_min = float(np.min(fitness))
    f_mean = float(np.mean(fitness))
    f_max = float(np.max(fitness))

    omega = np.empty(config.population_size)
    upper_group = fitness >= f_mean
    denominator = max(abs(f_max - f_mean), config.numerical_epsilon)
    omega[upper_group] = config.omega_min - (
        config.omega_min - config.omega_max
    ) * ((fitness[upper_group] - f_mean) / denominator)
    denominator = max(abs(f_mean - f_min), config.numerical_epsilon)
    omega[~upper_group] = config.omega_min + (
        config.omega_max - config.omega_min
    ) * ((fitness[~upper_group] - f_min) / denominator)

    c1 = np.full(config.population_size, config.c_min)
    lower_group = fitness <= f_mean
    c1[lower_group] = config.c_max + (config.c_max - config.c_min) * (
        (fitness[lower_group] - f_min) / denominator
    )
    c2 = 2.0 - c1

    population_center = state.positions.mean(axis=0)
    blended_positions = config.alpha * state.positions + (
        1.0 - config.alpha
    ) * state.previous_positions
    r3 = core_rng.uniform(0.0, 1.0, state.positions.shape)
    velocities = (
        omega[:, None]
        * (
            config.zeta * state.velocities
            + (1.0 - config.zeta) * state.previous_velocities
        )
        + c1[:, None] * (state.personal_best_positions - blended_positions)
        + c2[:, None]
        * (state.global_best_position[None, :] - blended_positions)
        + config.c3
        * r3
        * (population_center[None, :] - blended_positions)
    )
    velocity_limits = np.r_[
        np.full(HOURS, config.velocity_clamp_v),
        np.full(HOURS, config.velocity_clamp_qe),
        np.full(HOURS, config.velocity_clamp_qg),
    ]
    velocities = np.clip(velocities, -velocity_limits, velocity_limits)
    return state.positions + velocities, velocities


def _target_count(population_size, fraction):
    count = int(population_size * fraction)
    if count <= 0 or count > population_size:
        raise ValueError("Action target fraction selects an invalid row count")
    return count


def strategy_a1(x_base, state, rng, parameters=None, config=None):
    return x_base


def _sample_a2_donors(targets, population_size, rng):
    if population_size < 4:
        raise ValueError("A2 requires at least four particles")
    keys = rng.random((targets.size, population_size))
    keys[np.arange(targets.size), targets] = np.inf
    return np.argsort(keys, axis=1)[:, :3]


def strategy_a2(x_base, state, rng, parameters=None, config=None):
    parameters = parameters or ActionParameters()
    config = config or default_search_config()
    count = _target_count(config.population_size, parameters.a2_target_fraction)
    order = rank_indices(
        state.evaluation["total_cost"],
        state.evaluation["cv"],
        config.feasibility_tolerance,
    )
    targets = order[-count:]
    donors = _sample_a2_donors(targets, config.population_size, rng)
    mutant = x_base[donors[:, 0]] + parameters.a2_differential_weight * (
        x_base[donors[:, 1]] - x_base[donors[:, 2]]
    )
    crossover = rng.random((count, config.dimensions)) < (
        parameters.a2_crossover_probability
    )
    forced_dimensions = rng.integers(0, config.dimensions, size=count)
    crossover[np.arange(count), forced_dimensions] = True
    transformed = np.asarray(x_base, dtype=float).copy()
    transformed[targets] = np.where(crossover, mutant, x_base[targets])
    return transformed


def strategy_a3(x_base, state, rng, parameters=None, config=None):
    parameters = parameters or ActionParameters()
    config = config or default_search_config()
    count = _target_count(config.population_size, parameters.a3_target_fraction)
    order = rank_indices(
        state.evaluation["total_cost"],
        state.evaluation["cv"],
        config.feasibility_tolerance,
    )
    targets = order[:count]
    progress = min(1.0, state.search_nfe / config.search_budget)
    sigma = parameters.a3_sigma_min + (
        parameters.a3_sigma_0 - parameters.a3_sigma_min
    ) * (1.0 - progress)
    scale = np.r_[np.full(HOURS, V_MAX), np.ones(2 * HOURS)]
    noise = rng.normal(0.0, 1.0, (count, config.dimensions))
    transformed = np.asarray(x_base, dtype=float).copy()
    transformed[targets] = (
        state.global_best_position[None, :] + sigma * scale[None, :] * noise
    )
    return transformed


def strategy_a4(x_base, state, rng, parameters=None, config=None):
    parameters = parameters or ActionParameters()
    config = config or default_search_config()
    count = _target_count(config.population_size, parameters.a4_target_fraction)
    order = rank_indices(
        state.evaluation["total_cost"],
        state.evaluation["cv"],
        config.feasibility_tolerance,
    )
    targets = order[-count:]
    reference_index = best_index(
        state.personal_best_cost,
        state.personal_best_cv,
        config.feasibility_tolerance,
    )
    reference = state.personal_best_positions[reference_index]
    beta = rng.uniform(
        parameters.a4_beta_min, parameters.a4_beta_max, (count, 1)
    )
    transformed = np.asarray(x_base, dtype=float).copy()
    transformed[targets] = x_base[targets] + beta * (
        reference[None, :] - x_base[targets]
    )
    return transformed


def apply_action(action, x_base, state, streams, parameters=None, config=None):
    strategies = {
        Action.A1: strategy_a1,
        Action.A2: strategy_a2,
        Action.A3: strategy_a3,
        Action.A4: strategy_a4,
    }
    if not isinstance(action, Action):
        raise ValueError(f"Unknown action: {action!r}")
    return strategies[action](
        x_base,
        state,
        streams.for_action(action),
        parameters or ActionParameters(),
        config or default_search_config(),
    )


def _phase(state, config):
    if state.global_best_cv <= config.feasibility_tolerance:
        return Phase.COST
    return Phase.FEASIBILITY


def _normalized_diversity(positions):
    ranges = np.r_[np.full(HOURS, V_MAX), np.ones(2 * HOURS)]
    return float(np.mean(np.std(positions, axis=0) / ranges))


def make_selector_summary(state, config=None):
    config = config or default_search_config()
    cv = state.evaluation["cv"]
    cv_r, cv_soc, cv_e, cv_d = state.global_best_cv_components
    return SelectorSummary(
        window_index=state.iteration // config.decision_interval,
        iteration=state.iteration,
        search_nfe=state.search_nfe,
        budget_fraction=min(1.0, state.search_nfe / config.search_budget),
        phase=_phase(state, config),
        global_best_cost=state.global_best_cost,
        global_best_cv=state.global_best_cv,
        cv_r=cv_r,
        cv_soc=cv_soc,
        cv_e=cv_e,
        cv_d=cv_d,
        feasible_fraction=float(
            np.mean(cv <= config.feasibility_tolerance)
        ),
        population_cv_q25=float(np.quantile(cv, 0.25)),
        population_cv_median=float(np.median(cv)),
        population_cv_q75=float(np.quantile(cv, 0.75)),
        normalized_diversity=_normalized_diversity(state.positions),
        stagnation_iterations=state.stagnation_iterations,
        previous_action=state.active_action,
    )


def advance_search_state(state, positions, velocities, evaluation, config=None):
    config = config or default_search_config()
    old_positions = state.positions
    old_velocities = state.velocities
    old_global_cost = state.global_best_cost
    old_global_cv = state.global_best_cv
    old_feasible = old_global_cv <= config.feasibility_tolerance

    state.previous_positions = old_positions
    state.positions = positions
    state.previous_velocities = old_velocities
    state.velocities = velocities
    state.evaluation = evaluation

    better = is_better(
        evaluation["total_cost"],
        evaluation["cv"],
        state.personal_best_cost,
        state.personal_best_cv,
        config.feasibility_tolerance,
    )
    state.personal_best_positions[better] = positions[better]
    state.personal_best_cost[better] = evaluation["total_cost"][better]
    state.personal_best_cv[better] = evaluation["cv"][better]
    current_components = _cv_components(evaluation)
    state.personal_best_cv_components[better] = current_components[better]

    global_index = best_index(
        state.personal_best_cost,
        state.personal_best_cv,
        config.feasibility_tolerance,
    )
    state.global_best_position = state.personal_best_positions[global_index].copy()
    state.global_best_cost = float(state.personal_best_cost[global_index])
    state.global_best_cv = float(state.personal_best_cv[global_index])
    state.global_best_cv_components = _global_components(
        state.personal_best_cv_components, global_index
    )

    new_feasible = state.global_best_cv <= config.feasibility_tolerance
    improved = bool(
        is_better(
            np.array([state.global_best_cost]),
            np.array([state.global_best_cv]),
            np.array([old_global_cost]),
            np.array([old_global_cv]),
            config.feasibility_tolerance,
        )[0]
    )
    if new_feasible and not old_feasible:
        state.stagnation_iterations = 0
    elif new_feasible and state.global_best_cost < old_global_cost:
        state.stagnation_iterations = 0
    elif new_feasible:
        state.stagnation_iterations += 1
    else:
        state.stagnation_iterations = 0
    if state.first_feasible_nfe is None and new_feasible:
        state.first_feasible_nfe = state.search_nfe
    return improved


def _history_row(state, decision, improved, config):
    cv_r, cv_soc, cv_e, cv_d = state.global_best_cv_components
    summary = make_selector_summary(state, config)
    return {
        "iteration": state.iteration,
        "search_nfe": state.search_nfe,
        "action": decision.applied_action.value,
        "phase": summary.phase.value,
        "best_cost": state.global_best_cost,
        "best_cv": state.global_best_cv,
        "best_fitness": state.global_best_cost
        + config.penalty_lambda * state.global_best_cv,
        "feasible": state.global_best_cv <= config.feasibility_tolerance,
        "feasible_fraction": summary.feasible_fraction,
        "cv_r": cv_r,
        "cv_soc": cv_soc,
        "cv_e": cv_e,
        "cv_d": cv_d,
        "median_cv": summary.population_cv_median,
        "normalized_diversity": summary.normalized_diversity,
        "improved": bool(improved),
        "stagnation_iterations": state.stagnation_iterations,
        "requested_action": (
            decision.requested_action.value
            if decision.requested_action is not None
            else None
        ),
        "selector_fallback": decision.fallback_used,
        "selector_error_kind": (
            decision.error_kind.value if decision.error_kind is not None else None
        ),
        "selector_elapsed_seconds": decision.elapsed_seconds,
        "llm_call_id": decision.llm_call_id,
    }


def _select_action(selector, state, streams, config):
    if selector is None:
        return SelectionDecision(
            requested_action=Action.A1,
            applied_action=Action.A1,
        )
    decision = selector.select(make_selector_summary(state, config), streams.selector)
    if not isinstance(decision, SelectionDecision):
        raise TypeError("Selector must return SelectionDecision")
    if not isinstance(decision.applied_action, Action):
        raise TypeError("SelectionDecision.applied_action must be Action")
    return decision


def _make_outcome(snapshot, state, decision, config):
    window_index, phase, start_iteration, start_nfe, start_cost, start_cv = snapshot
    if phase == Phase.FEASIBILITY:
        denominator = max(abs(start_cv), config.numerical_epsilon)
        relative_improvement = (start_cv - state.global_best_cv) / denominator
    else:
        denominator = max(abs(start_cost), config.numerical_epsilon)
        relative_improvement = (start_cost - state.global_best_cost) / denominator
    return StrategyOutcome(
        window_index=window_index,
        phase_at_start=phase,
        phase_at_end=_phase(state, config),
        start_iteration=start_iteration,
        end_iteration=state.iteration,
        start_nfe=start_nfe,
        end_nfe=state.search_nfe,
        start_best_cost=start_cost,
        end_best_cost=state.global_best_cost,
        start_best_cv=start_cv,
        end_best_cv=state.global_best_cv,
        first_feasible_appeared=(
            phase == Phase.FEASIBILITY
            and state.global_best_cv <= config.feasibility_tolerance
        ),
        first_feasible_nfe=(
            state.first_feasible_nfe
            if phase == Phase.FEASIBILITY
            and state.global_best_cv <= config.feasibility_tolerance
            else None
        ),
        relative_improvement=float(np.clip(relative_improvement, -1.0, 1.0)),
        requested_action=decision.requested_action,
        applied_action=decision.applied_action,
        fallback_used=decision.fallback_used,
        error_kind=decision.error_kind,
        elapsed_seconds=decision.elapsed_seconds,
        llm_call_id=decision.llm_call_id,
    )


def solve_experiment(
    p_vital,
    p_nonvital,
    p_pv,
    *,
    seed=SEED,
    selector=None,
    config=None,
    action_parameters=None,
    progress_callback=None,
    checkpoint_callback=None,
):
    config = config or default_search_config()
    parameters = action_parameters or ActionParameters()
    validate_experiment_configuration(config, parameters)
    streams = make_random_streams(seed)
    evaluator = lambda positions: evaluate(
        positions,
        p_vital,
        p_nonvital,
        p_pv,
        penalty_lambda=config.penalty_lambda,
    )
    state = initialize_search(evaluator, streams.core, config)
    initial_decision = SelectionDecision(Action.A1, Action.A1)
    history = [_history_row(state, initial_decision, False, config)]
    outcomes = []
    decision = initial_decision
    window_snapshot = None

    for iteration in range(1, config.max_iterations + 1):
        if (iteration - 1) % config.decision_interval == 0:
            decision = _select_action(selector, state, streams, config)
            state.active_action = decision.applied_action
            window_snapshot = (
                state.iteration // config.decision_interval,
                _phase(state, config),
                state.iteration,
                state.search_nfe,
                state.global_best_cost,
                state.global_best_cv,
            )

        x_base, v_base = mppso_step(state, streams.core, config)
        x_action = apply_action(
            decision.applied_action,
            x_base,
            state,
            streams,
            parameters,
            config,
        )
        positions = canonicalize(x_action, config)
        evaluation = evaluator(positions)
        state.search_nfe += config.population_size
        state.iteration = iteration
        improved = advance_search_state(
            state, positions, v_base, evaluation, config
        )
        row = _history_row(state, decision, improved, config)
        history.append(row)
        if progress_callback is not None:
            progress_callback(state, row)
        if checkpoint_callback is not None:
            checkpoint_callback(state, streams, row)

        window_ended = iteration % config.decision_interval == 0
        if iteration == config.max_iterations:
            window_ended = True
        if window_ended:
            outcome = _make_outcome(window_snapshot, state, decision, config)
            outcomes.append(outcome)
            if selector is not None:
                selector.observe(decision.applied_action, outcome)

    audit_evaluation = evaluator(state.global_best_position)
    return SearchRun(
        best_position=state.global_best_position.copy(),
        audit_evaluation=audit_evaluation,
        history=history,
        outcomes=outcomes,
        search_nfe=state.search_nfe,
        audit_nfe=1,
        first_feasible_nfe=state.first_feasible_nfe,
    )


def _legacy_progress(state, row):
    if state.iteration == 1 or state.iteration % 50 == 0:
        print(
            f"iteration={state.iteration:3d} "
            f"cost={state.global_best_cost:.6f} cv={state.global_best_cv:.3e}"
        )


def solve(p_vital, p_nonvital, p_pv):
    run = solve_experiment(
        p_vital,
        p_nonvital,
        p_pv,
        seed=SEED,
        selector=None,
        progress_callback=_legacy_progress,
    )
    legacy_keys = ("iteration", "best_cost", "best_cv", "best_fitness", "feasible")
    legacy_history = [
        {key: row[key] for key in legacy_keys} for row in run.history
    ]
    return run.best_position, run.audit_evaluation, legacy_history


def validation_metrics(result, p_vital, p_nonvital, p_pv):
    speeds = result["speeds"][0]
    p_propulsion = result["p_propulsion"][0]
    p_ess = result["p_ess"][0]
    p_g1 = result["p_g1"][0]
    p_g2 = result["p_g2"][0]
    energy_ess = result["energy_ess"][0]
    soc = result["soc"][0]
    eeoi = result["eeoi"][0]
    p_load = p_vital + p_nonvital
    power_residual = p_g1 + p_g2 + p_ess + p_pv - p_load - p_propulsion

    return {
        "distance_nm": float(speeds.sum()),
        "distance_error_nm": float(abs(speeds.sum() - DISTANCE)),
        "power_balance_residual_mw": float(np.max(np.abs(power_residual))),
        "speed_min_kn": float(speeds.min()),
        "speed_max_kn": float(speeds.max()),
        "p_ess_min_mw": float(p_ess.min()),
        "p_ess_max_mw": float(p_ess.max()),
        "p_g1_min_mw": float(p_g1.min()),
        "p_g1_max_mw": float(p_g1.max()),
        "p_g2_min_mw": float(p_g2.min()),
        "p_g2_max_mw": float(p_g2.max()),
        "ramp_g1_max_mw_per_h": float(np.max(np.abs(np.diff(p_g1)))),
        "ramp_g2_max_mw_per_h": float(np.max(np.abs(np.diff(p_g2)))),
        "ramp_ess_max_mw_per_h": float(np.max(np.abs(np.diff(p_ess)))),
        "energy_min_mwh": float(energy_ess.min()),
        "energy_max_mwh": float(energy_ess.max()),
        "energy_final_mwh": float(energy_ess[-1]),
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "soc_final": float(soc[-1]),
        "eeoi_max": float(eeoi.max()),
        "cv_r": float(result["cv_r"][0]),
        "cv_soc": float(result["cv_soc"][0]),
        "cv_e": float(result["cv_e"][0]),
        "cv_d": float(result["cv_d"][0]),
        "cv_total": float(result["cv"][0]),
    }


def validation_failures(
    result,
    metrics,
    feasibility_tolerance=FEASIBILITY_TOLERANCE,
):
    failures = []
    tolerance = 1.0e-9
    if not all(np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray)):
        failures.append("result contains NaN or infinite values")
    if metrics["distance_error_nm"] > 1.0e-8:
        failures.append("distance equality is violated")
    if metrics["power_balance_residual_mw"] > 1.0e-9:
        failures.append("power balance is violated")
    if metrics["speed_min_kn"] < -tolerance or metrics["speed_max_kn"] > 11.0 + tolerance:
        failures.append("speed bound is violated")
    if metrics["p_ess_min_mw"] < -3.0 - tolerance or metrics["p_ess_max_mw"] > 3.0 + tolerance:
        failures.append("ESS power bound is violated")
    if metrics["p_g1_min_mw"] < -tolerance or metrics["p_g1_max_mw"] > 10.0 + tolerance:
        failures.append("G1 power bound is violated")
    if metrics["p_g2_min_mw"] < -tolerance or metrics["p_g2_max_mw"] > 20.0 + tolerance:
        failures.append("G2 power bound is violated")
    if metrics["ramp_g1_max_mw_per_h"] > 2.0 + tolerance:
        failures.append("G1 ramp bound is violated")
    if metrics["ramp_g2_max_mw_per_h"] > 3.0 + tolerance:
        failures.append("G2 ramp bound is violated")
    if metrics["ramp_ess_max_mw_per_h"] > 1.0 + tolerance:
        failures.append("ESS ramp bound is violated")
    if metrics["energy_min_mwh"] < 15.0 - tolerance or metrics["energy_max_mwh"] > 75.0 + tolerance:
        failures.append("ESS energy bound is violated")
    if metrics["soc_min"] < 0.2 - tolerance or metrics["soc_max"] > 1.0 + tolerance:
        failures.append("SOC bound is violated")
    if metrics["eeoi_max"] > 23.0 + tolerance:
        failures.append("EEOI bound is violated")
    if metrics["cv_total"] > feasibility_tolerance:
        failures.append("total constraint violation exceeds tolerance")
    return failures


def write_schedule(path, hours, p_vital, p_nonvital, p_pv, result):
    fieldnames = [
        "hour",
        "p_vital_mw",
        "p_nonvital_mw",
        "p_load_mw",
        "p_pv_mw",
        "v_kn",
        "p_propulsion_mw",
        "p_ess_mw",
        "energy_ess_mwh",
        "soc",
        "p_g1_mw",
        "p_g2_mw",
        "eeoi",
        "co2_g1",
        "co2_g2",
        "cost_g1",
        "cost_g2",
        "cost_ess",
        "cost_pv",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, hour in enumerate(hours):
            writer.writerow(
                {
                    "hour": int(hour),
                    "p_vital_mw": f"{p_vital[index]:.6f}",
                    "p_nonvital_mw": f"{p_nonvital[index]:.6f}",
                    "p_load_mw": f"{p_vital[index] + p_nonvital[index]:.6f}",
                    "p_pv_mw": f"{p_pv[index]:.6f}",
                    "v_kn": f"{result['speeds'][0, index]:.12f}",
                    "p_propulsion_mw": f"{result['p_propulsion'][0, index]:.12f}",
                    "p_ess_mw": f"{result['p_ess'][0, index]:.12f}",
                    "energy_ess_mwh": f"{result['energy_ess'][0, index]:.12f}",
                    "soc": f"{result['soc'][0, index]:.12f}",
                    "p_g1_mw": f"{result['p_g1'][0, index]:.12f}",
                    "p_g2_mw": f"{result['p_g2'][0, index]:.12f}",
                    "eeoi": f"{result['eeoi'][0, index]:.12f}",
                    "co2_g1": f"{result['emissions_g1'][0, index]:.12f}",
                    "co2_g2": f"{result['emissions_g2'][0, index]:.12f}",
                    "cost_g1": f"{result['cost_g1'][0, index]:.12f}",
                    "cost_g2": f"{result['cost_g2'][0, index]:.12f}",
                    "cost_ess": f"{result['cost_ess'][0, index]:.12f}",
                    "cost_pv": f"{result['cost_pv'][0, index]:.12f}",
                }
            )


def write_convergence(path, history):
    fieldnames = ["iteration", "best_cost", "best_cv", "best_fitness", "feasible"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "iteration": row["iteration"],
                    "best_cost": f"{row['best_cost']:.9f}",
                    "best_cv": f"{row['best_cv']:.12e}",
                    "best_fitness": f"{row['best_fitness']:.9f}",
                    "feasible": int(row["feasible"]),
                }
            )


def write_convergence_figure(path, history):
    iterations = np.array([row["iteration"] for row in history])
    costs = np.array([row["best_cost"] for row in history])
    cvs = np.array([row["best_cv"] for row in history])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    axes[0].plot(iterations, costs, color="#0F4D92", linewidth=1.8)
    axes[0].set_ylabel("Best cost")
    axes[0].grid(alpha=0.22, linewidth=0.7)

    axes[1].semilogy(
        iterations,
        np.maximum(cvs, 1.0e-16),
        color="#B33A3A",
        linewidth=1.8,
    )
    axes[1].axhline(
        FEASIBILITY_TOLERANCE,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Feasibility tolerance",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best CV")
    axes[1].grid(alpha=0.22, linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, result, metrics):
    lines = [
        "# CASE2 调度优化结果",
        "",
        f"- 随机种子：`{SEED}`",
        f"- MPPSO 参数：种群 `{POPULATION_SIZE}`，迭代 `{MAX_ITERATIONS}`",
        f"- 本次 MPPSO 最佳可行总运行成本：`{result['total_cost'][0]:.6f}`",
        f"- 总排放：`{result['total_emissions'][0]:.6f}`",
        "- 时段标签：CSV 的 `hour=0..23` 依次对应模型的 `t=1..24`",
        f"- 总航程：`{metrics['distance_nm']:.9f} nm`",
        f"- 最大功率平衡残差：`{metrics['power_balance_residual_mw']:.3e} MW`",
        f"- 航速范围：`[{metrics['speed_min_kn']:.6f}, {metrics['speed_max_kn']:.6f}] kn`",
        f"- G1 出力范围：`[{metrics['p_g1_min_mw']:.6f}, {metrics['p_g1_max_mw']:.6f}] MW`",
        f"- G2 出力范围：`[{metrics['p_g2_min_mw']:.6f}, {metrics['p_g2_max_mw']:.6f}] MW`",
        f"- ESS 功率范围：`[{metrics['p_ess_min_mw']:.6f}, {metrics['p_ess_max_mw']:.6f}] MW`",
        f"- G1 最大爬坡：`{metrics['ramp_g1_max_mw_per_h']:.6f} MW/h`",
        f"- G2 最大爬坡：`{metrics['ramp_g2_max_mw_per_h']:.6f} MW/h`",
        f"- ESS 最大爬坡：`{metrics['ramp_ess_max_mw_per_h']:.6f} MW/h`",
        f"- ESS 能量范围：`[{metrics['energy_min_mwh']:.6f}, {metrics['energy_max_mwh']:.6f}] MWh`",
        f"- 终端 ESS 能量：`{metrics['energy_final_mwh']:.6f} MWh`",
        f"- SOC 范围：`[{metrics['soc_min']:.6f}, {metrics['soc_max']:.6f}]`",
        f"- 终端 SOC：`{metrics['soc_final']:.6f}`",
        f"- 最大 EEOI：`{metrics['eeoi_max']:.6f}`",
        f"- 约束违反量：`CV_R={metrics['cv_r']:.3e}`，`CV_SOC={metrics['cv_soc']:.3e}`，`CV_E={metrics['cv_e']:.3e}`，`CV_D={metrics['cv_d']:.3e}`",
        f"- 总约束违反量：`{metrics['cv_total']:.3e}`",
        "",
        "可行性结论：满足 `case2.md` 定义的全部约束。",
        "",
        "> 该结果是固定随机种子下本次 MPPSO 找到的最佳可行解，不代表已经证明全局最优。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data.md"
    hours, p_vital, p_nonvital, p_pv = load_input_data(data_path)

    _, result, history = solve(p_vital, p_nonvital, p_pv)
    metrics = validation_metrics(result, p_vital, p_nonvital, p_pv)

    convergence_path = base_dir / "case2_convergence.csv"
    write_convergence(convergence_path, history)
    figure_dir = base_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    write_convergence_figure(figure_dir / "case2_convergence.png", history)

    failures = validation_failures(result, metrics)
    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            f"MPPSO did not produce a valid Case 2 schedule: {detail}; "
            f"CV={metrics['cv_total']:.6e}"
        )

    write_schedule(
        base_dir / "case2_schedule.csv",
        hours,
        p_vital,
        p_nonvital,
        p_pv,
        result,
    )
    write_summary(base_dir / "case2_result.md", result, metrics)
    print(f"best_total_cost={result['total_cost'][0]:.6f}")
    print(f"total_emissions={result['total_emissions'][0]:.6f}")
    print(f"terminal_soc={metrics['soc_final']:.6f}")
    print(f"total_cv={metrics['cv_total']:.3e}")


if __name__ == "__main__":
    main()
