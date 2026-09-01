from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import numpy as np


HOURS = 24
DELTA_T = 1.0
DISTANCE_NM = 240.0
V_MAX = 11.0

G1_MAX = 10.0
G2_MAX = 20.0
G1_RAMP = 2.0
G2_RAMP = 3.0

ESS_POWER_MIN = -3.0
ESS_POWER_MAX = 3.0
ESS_RAMP = 1.0
ESS_ENERGY_INITIAL = 7.5
ESS_ENERGY_MIN = 3.0
ESS_ENERGY_MAX = 15.0
ESS_EFFICIENCY_IN = 0.95
ESS_EFFICIENCY_OUT = 0.95

FAULT_ALPHA = 0.0
FAULT_G1_MAX = FAULT_ALPHA * G1_MAX
RESERVE_HORIZON = 2
FAULT_START_HOUR = 5
FAULT_DURATION = 4
LOAD_SHEDDING_PENALTY = 1.0e4

NP_MAX = 600
NP_MIN = 4
MAX_ITERATIONS = 500
MEMORY_SIZE = 6
PBEST_RATE = 0.11
LAMBDA_R_MIN = 1.0
LAMBDA_R_MAX = 100.0
LAMBDA_R_BETA = 2.0
FEASIBILITY_TOLERANCE = 1.0e-8
PAIRED_SEEDS = (20260826, 20260827, 20260828, 20260829, 20260830)


def configure_ess_capacity(capacity_mwh):
    global ESS_ENERGY_INITIAL, ESS_ENERGY_MIN, ESS_ENERGY_MAX
    capacity_mwh = float(capacity_mwh)
    if not math.isfinite(capacity_mwh) or capacity_mwh <= 0.0:
        raise ValueError("ESS capacity must be a positive finite value")
    ESS_ENERGY_MAX = capacity_mwh
    ESS_ENERGY_INITIAL = 0.5 * capacity_mwh
    ESS_ENERGY_MIN = 0.2 * capacity_mwh


def configure_fault_g1_max(max_power_mw):
    global FAULT_ALPHA, FAULT_G1_MAX
    max_power_mw = float(max_power_mw)
    if not math.isfinite(max_power_mw) or not 0.0 <= max_power_mw <= G1_MAX:
        raise ValueError(f"Faulted G1 maximum must remain within [0, {G1_MAX:g}] MW")
    FAULT_G1_MAX = max_power_mw
    FAULT_ALPHA = max_power_mw / G1_MAX


def configure_g2_max(max_power_mw):
    global G2_MAX
    max_power_mw = float(max_power_mw)
    if not math.isfinite(max_power_mw) or max_power_mw <= 0.0:
        raise ValueError("G2 maximum power must be a positive finite value")
    G2_MAX = max_power_mw


def load_input_data(path: Path):
    service_rows = []
    pv_rows = []
    service_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$"
    )
    pv_pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*$")

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

    p_vital = np.array([row[1] for row in service_rows], dtype=float)
    p_nonvital = np.array([row[2] for row in service_rows], dtype=float)
    p_pv = np.array([row[1] for row in pv_rows], dtype=float)
    if not all(np.isfinite(values).all() for values in (p_vital, p_nonvital, p_pv)):
        raise ValueError("Input data contains NaN or infinite values")
    if np.any(p_vital < 0.0) or np.any(p_nonvital < 0.0):
        raise ValueError("Service loads cannot be negative")
    if np.any((p_pv < 0.0) | (p_pv > 4.2)):
        raise ValueError("PV power must remain within [0, 4.2] MW")
    return np.arange(HOURS), p_vital, p_nonvital, p_pv


def project_speeds(values):
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
        lower = np.where(sums > DISTANCE_NM, tau, lower)
        upper = np.where(sums > DISTANCE_NM, upper, tau)
    projected = np.clip(rows - ((lower + upper) / 2.0)[:, None], 0.0, V_MAX)
    return projected[0] if one_dimensional else projected


def ess_energy_trajectory(p_ess, initial_energy=None):
    p_ess = np.asarray(p_ess, dtype=float)
    if initial_energy is None:
        initial_energy = ESS_ENERGY_INITIAL
    change = np.where(
        p_ess <= 0.0,
        -p_ess * ESS_EFFICIENCY_IN * DELTA_T,
        -p_ess / ESS_EFFICIENCY_OUT * DELTA_T,
    )
    return initial_energy + np.cumsum(change, axis=-1)


def reserve_requirements(net_demand, p_g2):
    net_demand = np.asarray(net_demand, dtype=float)
    p_g2 = np.asarray(p_g2, dtype=float)
    one_dimensional = net_demand.ndim == 1
    demand_rows = net_demand.reshape(1, -1) if one_dimensional else net_demand
    g2_rows = p_g2.reshape(1, -1) if one_dimensional else p_g2
    if demand_rows.shape != g2_rows.shape or demand_rows.shape[1] != HOURS:
        raise ValueError("Reserve inputs must have matching 24-hour shapes")

    reserve_energy = np.full(demand_rows.shape, np.nan)
    reserve_power_max = np.full(demand_rows.shape, np.nan)
    last_start = HOURS - RESERVE_HORIZON
    for start in range(1, last_start + 1):
        available_g2 = np.minimum(G2_MAX, g2_rows[:, start - 1] + G2_RAMP)
        required_power = []
        for hour in range(start, start + RESERVE_HORIZON):
            if hour > start:
                available_g2 = np.minimum(G2_MAX, available_g2 + G2_RAMP)
            required_power.append(
                np.maximum(0.0, demand_rows[:, hour] - FAULT_G1_MAX - available_g2)
            )
        required_power = np.stack(required_power, axis=1)
        reserve_energy[:, start] = (
            required_power.sum(axis=1) * DELTA_T / ESS_EFFICIENCY_OUT
        )
        reserve_power_max[:, start] = required_power.max(axis=1)

    if one_dimensional:
        return reserve_energy[0], reserve_power_max[0]
    return reserve_energy, reserve_power_max


def _repair_positions(positions):
    positions = np.asarray(positions, dtype=float).copy()
    positions[:, :HOURS] = np.clip(positions[:, :HOURS], 0.0, G2_MAX)
    positions[:, HOURS : 2 * HOURS] = np.clip(
        positions[:, HOURS : 2 * HOURS], ESS_POWER_MIN, ESS_POWER_MAX
    )
    positions[:, 2 * HOURS :] = project_speeds(positions[:, 2 * HOURS :])
    return positions


def evaluate_normal(positions, p_vital, p_nonvital, p_pv, strategy, lambda_r=1.0):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    if positions.ndim != 2 or positions.shape[1] != 3 * HOURS:
        raise ValueError("Each Case 3 candidate must contain 72 values")
    if strategy not in {"no_reserve", "dynamic_reserve"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    p_g2 = positions[:, :HOURS]
    p_ess = positions[:, HOURS : 2 * HOURS]
    speeds = positions[:, 2 * HOURS :]
    p_load = p_vital + p_nonvital
    p_propulsion = 0.0022 * speeds**3
    net_demand = p_load[None, :] + p_propulsion - p_pv[None, :]
    p_g1 = net_demand - p_g2 - p_ess
    energy_ess = ess_energy_trajectory(p_ess)

    cv_g1 = (
        np.maximum(0.0, -p_g1 / G1_MAX)
        + np.maximum(0.0, (p_g1 - G1_MAX) / G1_MAX)
    ).sum(axis=1)
    cv_energy = (
        np.maximum(0.0, (ESS_ENERGY_MIN - energy_ess) / (ESS_ENERGY_MAX - ESS_ENERGY_MIN))
        + np.maximum(0.0, (energy_ess - ESS_ENERGY_MAX) / (ESS_ENERGY_MAX - ESS_ENERGY_MIN))
    ).sum(axis=1)
    cv_ramp = (
        np.maximum(0.0, (np.abs(np.diff(p_g1, axis=1)) - G1_RAMP) / G1_RAMP).sum(axis=1)
        + np.maximum(0.0, (np.abs(np.diff(p_g2, axis=1)) - G2_RAMP) / G2_RAMP).sum(axis=1)
        + np.maximum(0.0, (np.abs(np.diff(p_ess, axis=1)) - ESS_RAMP) / ESS_RAMP).sum(axis=1)
    )

    emissions_g1 = 13.5 * p_g1**2 + 10.0 * p_g1 + 450.0
    emissions_g2 = 5.2 * p_g2**2 + 58.0 * p_g2 + 390.0
    eeoi = (emissions_g1 + emissions_g2) / (20.0 * np.maximum(speeds, 1.0e-6))
    cv_eeoi = np.maximum(0.0, (eeoi - 23.0) / 23.0).sum(axis=1)
    cv_base = cv_g1 + cv_energy + cv_ramp + cv_eeoi

    reserve_energy, reserve_power_max = reserve_requirements(net_demand, p_g2)
    cv_res = np.zeros(positions.shape[0])
    if strategy == "dynamic_reserve":
        starts = np.arange(1, HOURS - RESERVE_HORIZON + 1)
        pre_fault_energy = energy_ess[:, starts - 1]
        energy_deficit = np.maximum(
            0.0,
            ESS_ENERGY_MIN + reserve_energy[:, starts] - pre_fault_energy,
        )
        power_deficit = np.maximum(
            0.0, reserve_power_max[:, starts] - ESS_POWER_MAX
        )
        cv_res = (
            energy_deficit / (ESS_ENERGY_MAX - ESS_ENERGY_MIN)
        ).sum(axis=1) + (power_deficit / ESS_POWER_MAX).sum(axis=1)

    cost_g1 = 13.0 * p_g1**2 + 12.0 * p_g1 + 430.0
    cost_g2 = 5.2 * p_g2**2 + 52.0 * p_g2 + 340.0
    cost_ess = 4.3 * p_ess**2 + 1.0
    cost_pv = 10.2 * p_pv[None, :]
    total_cost = (cost_g1 + cost_g2 + cost_ess + cost_pv).sum(axis=1)
    cv = cv_base + lambda_r * cv_res

    return {
        "speeds": speeds,
        "p_propulsion": p_propulsion,
        "net_demand": net_demand,
        "p_g1": p_g1,
        "p_g2": p_g2,
        "p_ess": p_ess,
        "energy_ess": energy_ess,
        "soc": energy_ess / ESS_ENERGY_MAX,
        "eeoi": eeoi,
        "reserve_energy": reserve_energy,
        "reserve_power_max": reserve_power_max,
        "cost_g1": cost_g1,
        "cost_g2": cost_g2,
        "cost_ess": cost_ess,
        "cost_pv": np.broadcast_to(cost_pv, speeds.shape),
        "total_cost": total_cost,
        "cv_base": cv_base,
        "cv_res": cv_res,
        "cv": cv,
    }


def _initial_population(rng, p_vital, p_nonvital, p_pv, size):
    speeds = project_speeds(rng.normal(10.0, 0.6, (size, HOURS)))
    p_ess = rng.normal(0.0, 0.35, (size, HOURS))
    p_ess = np.apply_along_axis(lambda row: np.convolve(row, np.ones(3) / 3.0, mode="same"), 1, p_ess)
    p_ess = np.clip(p_ess, -0.8, 0.8)
    net_demand = p_vital + p_nonvital + 0.0022 * speeds**3 - p_pv
    total_generation = net_demand - p_ess
    p_g1_economic = np.clip((10.4 * total_generation + 40.0) / 36.4, 0.0, G1_MAX)
    p_g2 = np.clip(total_generation - p_g1_economic, 0.0, G2_MAX)
    p_g2 += rng.normal(0.0, 0.25, p_g2.shape)

    positions = np.hstack((p_g2, p_ess, speeds))
    positions = _repair_positions(positions)

    # Keep one deterministic, balanced candidate in every run.
    base_speed = np.full(HOURS, DISTANCE_NM / HOURS)
    base_demand = p_vital + p_nonvital + 0.0022 * base_speed**3 - p_pv
    base_g1 = np.clip((10.4 * base_demand + 40.0) / 36.4, 0.0, G1_MAX)
    positions[0] = np.r_[base_demand - base_g1, np.zeros(HOURS), base_speed]
    return _repair_positions(positions)


def _epsilon_better(trial_cost, trial_cv, target_cost, target_cv, epsilon):
    trial_feasible = trial_cv <= epsilon
    target_feasible = target_cv <= epsilon
    return (
        (trial_feasible & ~target_feasible)
        | (trial_feasible & target_feasible & (trial_cost < target_cost))
        | (~trial_feasible & ~target_feasible & (trial_cv < target_cv))
        | (~trial_feasible & ~target_feasible & (trial_cv == target_cv) & (trial_cost < target_cost))
    )


def _rank_indices(cost, cv, epsilon):
    feasible = cv <= epsilon
    primary = (~feasible).astype(int)
    secondary = np.where(feasible, cost, cv)
    return np.lexsort((cost, secondary, primary))


def _candidate_result(results, index, population_size):
    candidate = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == population_size:
            candidate[key] = value[index].copy()
        else:
            candidate[key] = value
    return candidate


def _validate_ra_lshade_budget(np_max, np_min, iterations):
    if np_min < 4 or np_max < np_min:
        raise ValueError("RA-LSHADE population bounds must satisfy NP_max >= NP_min >= 4")
    if iterations < 0:
        raise ValueError("RA-LSHADE iteration count cannot be negative")


def _run_ra_lshade(
    rng,
    initial_population,
    repair,
    evaluate,
    strictly_feasible,
    np_min,
    iterations,
    failure_message,
    history_metrics=None,
):
    population = np.asarray(initial_population, dtype=float).copy()
    np_max = len(population)
    if population.ndim != 2:
        raise ValueError("RA-LSHADE population must be a two-dimensional array")
    _validate_ra_lshade_budget(np_max, np_min, iterations)

    archive = np.empty((0, population.shape[1]))
    memory_f = np.full(MEMORY_SIZE, 0.5)
    memory_cr = np.full(MEMORY_SIZE, 0.5)
    memory_index = 0

    results = evaluate(population, 0.0)
    epsilon_cap = float(np.max(results["cv"]))
    epsilon = float(np.quantile(results["cv"], 0.2))
    best_position = None
    best_result = None
    best_cost = math.inf
    history = []

    def update_best(current_population, current_results):
        nonlocal best_position, best_result, best_cost
        feasible = strictly_feasible(current_results)
        if not np.any(feasible):
            return
        indices = np.flatnonzero(feasible)
        index = int(indices[np.argmin(current_results["total_cost"][indices])])
        if current_results["total_cost"][index] < best_cost:
            best_cost = float(current_results["total_cost"][index])
            best_position = current_population[index].copy()
            best_result = _candidate_result(current_results, index, len(current_population))

    for iteration in range(iterations + 1):
        progress = iteration / max(iterations, 1)
        results = evaluate(population, progress)
        update_best(population, results)
        feasible_ratio = float(np.mean(strictly_feasible(results)))
        row = {
            "iteration": iteration,
            "population": len(population),
            "epsilon": epsilon,
            "feasible_ratio": feasible_ratio,
            "best_cost": best_cost if best_position is not None else math.nan,
            "minimum_cv": float(np.min(results["cv"])),
        }
        if history_metrics is not None:
            row.update(history_metrics(results, best_result))
        history.append(row)
        if iteration == iterations:
            break

        if iteration + 1 >= math.ceil(0.8 * iterations):
            epsilon = 0.0
        elif feasible_ratio < 0.2:
            epsilon = min(epsilon_cap, epsilon * 1.05 + 1.0e-12)
        elif feasible_ratio > 0.5:
            epsilon *= 0.9

        population_size = len(population)
        rank = _rank_indices(results["total_cost"], results["cv"], epsilon)
        pbest_count = min(
            population_size,
            max(2, int(math.ceil(PBEST_RATE * population_size))),
        )

        memory_choices = rng.integers(0, MEMORY_SIZE, population_size)
        f_values = np.empty(population_size)
        for i, memory_choice in enumerate(memory_choices):
            value = -1.0
            while value <= 0.0:
                value = memory_f[memory_choice] + 0.1 * math.tan(
                    math.pi * (rng.random() - 0.5)
                )
            f_values[i] = min(value, 1.0)
        cr_values = np.clip(
            rng.normal(memory_cr[memory_choices], 0.1, population_size),
            0.0,
            1.0,
        )

        mutants = np.empty_like(population)
        union = np.vstack((population, archive)) if len(archive) else population
        for i in range(population_size):
            pbest_index = int(rng.choice(rank[:pbest_count]))
            r1_candidates = np.delete(np.arange(population_size), i)
            r1 = int(rng.choice(r1_candidates))
            while True:
                r2 = int(rng.integers(0, len(union)))
                if r2 != i and r2 != r1:
                    break
            mutants[i] = (
                population[i]
                + f_values[i] * (population[pbest_index] - population[i])
                + f_values[i] * (population[r1] - union[r2])
            )

        crossover = rng.random(population.shape) <= cr_values[:, None]
        forced_dimensions = rng.integers(0, population.shape[1], population_size)
        crossover[np.arange(population_size), forced_dimensions] = True
        trials = repair(np.where(crossover, mutants, population))
        trial_results = evaluate(trials, progress)
        accepted = _epsilon_better(
            trial_results["total_cost"],
            trial_results["cv"],
            results["total_cost"],
            results["cv"],
            epsilon,
        )

        if np.any(accepted):
            successful_f = f_values[accepted]
            successful_cr = cr_values[accepted]
            parent_quality = results["total_cost"] + 1.0e6 * results["cv"]
            trial_quality = trial_results["total_cost"] + 1.0e6 * trial_results["cv"]
            improvements = np.maximum(
                1.0e-12,
                parent_quality[accepted] - trial_quality[accepted],
            )
            weights = improvements / improvements.sum()
            memory_f[memory_index] = np.sum(weights * successful_f**2) / np.sum(
                weights * successful_f
            )
            memory_cr[memory_index] = np.sum(weights * successful_cr)
            memory_index = (memory_index + 1) % MEMORY_SIZE

            archive = np.vstack((archive, population[accepted]))
            population[accepted] = trials[accepted]

        if len(archive) > population_size:
            archive = archive[
                rng.choice(len(archive), population_size, replace=False)
            ]

        results = evaluate(population, progress)
        update_best(population, results)
        if iterations > 0:
            target_size = int(
                round(
                    np_max
                    - ((iteration + 1) / iterations) * (np_max - np_min)
                )
            )
            target_size = max(np_min, target_size)
            if population_size > target_size:
                keep = _rank_indices(
                    results["total_cost"], results["cv"], epsilon
                )[:target_size]
                population = population[keep]
                if len(archive) > target_size:
                    archive = archive[
                        rng.choice(len(archive), target_size, replace=False)
                    ]

    if best_position is None:
        raise RuntimeError(failure_message)
    return best_position, best_result, history


def solve_ra_lshade(
    p_vital,
    p_nonvital,
    p_pv,
    strategy,
    seed,
    np_max=NP_MAX,
    np_min=NP_MIN,
    iterations=MAX_ITERATIONS,
):
    _validate_ra_lshade_budget(np_max, np_min, iterations)
    rng = np.random.default_rng(seed)
    population = _initial_population(rng, p_vital, p_nonvital, p_pv, np_max)
    def evaluate(positions, progress):
        lambda_r = LAMBDA_R_MIN + (
            LAMBDA_R_MAX - LAMBDA_R_MIN
        ) * progress**LAMBDA_R_BETA
        return evaluate_normal(
            positions,
            p_vital,
            p_nonvital,
            p_pv,
            strategy,
            lambda_r,
        )

    def strictly_feasible(results):
        return (
            (results["cv_base"] <= FEASIBILITY_TOLERANCE)
            & (results["cv_res"] <= FEASIBILITY_TOLERANCE)
        )

    best_position, _, history = _run_ra_lshade(
        rng,
        population,
        _repair_positions,
        evaluate,
        strictly_feasible,
        np_min,
        iterations,
        f"RA-LSHADE did not find a feasible {strategy} solution for seed {seed}",
    )
    best_result = evaluate_normal(
        best_position, p_vital, p_nonvital, p_pv, strategy, LAMBDA_R_MAX
    )
    return best_position, {key: value[0] for key, value in best_result.items()}, history


def _validate_fault_interval(fault_start_hour, fault_duration):
    fault_start_hour = int(fault_start_hour)
    fault_duration = int(fault_duration)
    if fault_start_hour < 1:
        raise ValueError("Fault start hour must be at least 1 to connect a pre-fault state")
    if fault_duration < 1:
        raise ValueError("Fault duration must be positive")
    if fault_start_hour + fault_duration > HOURS:
        raise ValueError("Fault interval must remain within the 24-hour schedule")
    return fault_start_hour, fault_duration


def _repair_fault_positions(positions, p_nonvital_fault):
    positions = np.asarray(positions, dtype=float).copy()
    duration = len(p_nonvital_fault)
    positions[:, :duration] = np.clip(positions[:, :duration], 0.0, G2_MAX)
    positions[:, duration : 2 * duration] = np.clip(
        positions[:, duration : 2 * duration], ESS_POWER_MIN, ESS_POWER_MAX
    )
    positions[:, 2 * duration :] = np.clip(
        positions[:, 2 * duration :],
        0.0,
        np.asarray(p_nonvital_fault, dtype=float)[None, :],
    )
    return positions


def evaluate_fault(
    positions,
    normal_result,
    p_vital,
    p_nonvital,
    p_pv,
    fault_start_hour=FAULT_START_HOUR,
    fault_duration=FAULT_DURATION,
):
    fault_start_hour, fault_duration = _validate_fault_interval(
        fault_start_hour, fault_duration
    )
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    if positions.ndim != 2 or positions.shape[1] != 3 * fault_duration:
        raise ValueError(
            f"Each fault-stage candidate must contain {3 * fault_duration} values"
        )

    hours = np.arange(fault_start_hour, fault_start_hour + fault_duration)
    positions = _repair_fault_positions(positions, p_nonvital[hours])
    demand = (
        p_vital[hours]
        + p_nonvital[hours]
        + normal_result["p_propulsion"][hours]
        - p_pv[hours]
    )
    pre_fault_energy = float(normal_result["energy_ess"][fault_start_hour - 1])
    pre_fault_g2 = float(normal_result["p_g2"][fault_start_hour - 1])

    p_g2 = positions[:, :fault_duration]
    p_ess = positions[:, fault_duration : 2 * fault_duration]
    p_shed = positions[:, 2 * fault_duration :]
    p_g1 = demand[None, :] - p_g2 - p_ess - p_shed
    energy = ess_energy_trajectory(p_ess, pre_fault_energy)

    cv_g1 = (
        np.maximum(0.0, -p_g1 / G1_MAX)
        + np.maximum(0.0, (p_g1 - FAULT_G1_MAX) / G1_MAX)
    ).sum(axis=1)
    if fault_duration > 1:
        cv_g1_ramp = np.maximum(
            0.0,
            (np.abs(np.diff(p_g1, axis=1)) - G1_RAMP) / G1_RAMP,
        ).sum(axis=1)
        cv_g2_ramp = np.maximum(
            0.0,
            (np.abs(np.diff(p_g2, axis=1)) - G2_RAMP) / G2_RAMP,
        ).sum(axis=1)
    else:
        cv_g1_ramp = np.zeros(len(positions))
        cv_g2_ramp = np.zeros(len(positions))
    cv_g2_transition = np.maximum(
        0.0,
        (np.abs(p_g2[:, 0] - pre_fault_g2) - G2_RAMP) / G2_RAMP,
    )
    energy_range = ESS_ENERGY_MAX - ESS_ENERGY_MIN
    cv_energy = (
        np.maximum(0.0, (ESS_ENERGY_MIN - energy) / energy_range)
        + np.maximum(0.0, (energy - ESS_ENERGY_MAX) / energy_range)
    ).sum(axis=1)
    cv = cv_g1 + cv_g1_ramp + cv_g2_transition + cv_g2_ramp + cv_energy

    cost_g1 = 13.0 * p_g1**2 + 12.0 * p_g1 + 430.0
    cost_g2 = 5.2 * p_g2**2 + 52.0 * p_g2 + 340.0
    cost_ess = 4.3 * p_ess**2 + 1.0
    total_cost = np.sum(
        (cost_g1 + cost_g2 + cost_ess + LOAD_SHEDDING_PENALTY * p_shed)
        * DELTA_T,
        axis=1,
    )

    slack_parts = [
        p_g1,
        FAULT_G1_MAX - p_g1,
        p_g2,
        G2_MAX - p_g2,
        p_ess - ESS_POWER_MIN,
        ESS_POWER_MAX - p_ess,
        energy - ESS_ENERGY_MIN,
        ESS_ENERGY_MAX - energy,
        p_shed,
        p_nonvital[hours][None, :] - p_shed,
        (G2_RAMP - np.abs(p_g2[:, 0] - pre_fault_g2))[:, None],
    ]
    if fault_duration > 1:
        slack_parts.extend(
            [
                G1_RAMP - np.abs(np.diff(p_g1, axis=1)),
                G2_RAMP - np.abs(np.diff(p_g2, axis=1)),
            ]
        )
    minimum_slack = np.min(np.concatenate(slack_parts, axis=1), axis=1)
    balance_residual = p_g1 + p_g2 + p_ess + p_shed - demand[None, :]
    shed_energy = np.sum(p_shed, axis=1) * DELTA_T
    nonvital_energy = float(np.sum(p_nonvital[hours]) * DELTA_T)
    load_retention = (
        1.0 - shed_energy / nonvital_energy
        if nonvital_energy > 0.0
        else np.ones(len(positions))
    )

    return {
        "p_g1": p_g1,
        "p_g2": p_g2,
        "p_ess": p_ess,
        "energy_ess": energy,
        "p_shed": p_shed,
        "demand": np.broadcast_to(demand, p_g1.shape),
        "total_cost": total_cost,
        "shed_energy": shed_energy,
        "load_retention": load_retention,
        "balance_residual": balance_residual,
        "minimum_inequality_slack": minimum_slack,
        "cv_g1": cv_g1,
        "cv_g1_ramp": cv_g1_ramp,
        "cv_g2_transition": cv_g2_transition,
        "cv_g2_ramp": cv_g2_ramp,
        "cv_energy": cv_energy,
        "cv_f": cv,
        "cv": cv,
    }


def _fault_feasible_seed(normal_result, demand, p_nonvital_fault, fault_start_hour):
    duration = len(demand)
    p_g1 = np.zeros(duration)
    p_g2 = np.zeros(duration)
    p_ess = np.zeros(duration)
    p_shed = np.zeros(duration)
    energy = float(normal_result["energy_ess"][fault_start_hour - 1])
    previous_g2 = float(normal_result["p_g2"][fault_start_hour - 1])
    previous_g1 = 0.0

    for t in range(duration):
        g2_low = max(0.0, previous_g2 - G2_RAMP)
        g2_high = min(G2_MAX, previous_g2 + G2_RAMP)
        if t == 0:
            g1_low, g1_high = 0.0, FAULT_G1_MAX
        else:
            g1_low = max(0.0, previous_g1 - G1_RAMP)
            g1_high = min(FAULT_G1_MAX, previous_g1 + G1_RAMP)

        ess_high = min(
            ESS_POWER_MAX,
            max(0.0, energy - ESS_ENERGY_MIN)
            * ESS_EFFICIENCY_OUT
            / DELTA_T,
        )
        ess_low = -min(
            -ESS_POWER_MIN,
            max(0.0, ESS_ENERGY_MAX - energy)
            / (ESS_EFFICIENCY_IN * DELTA_T),
        )
        generation_low = g1_low + g2_low
        generation_high = g1_high + g2_high
        feasible_low = max(
            generation_low,
            demand[t] - ess_high - p_nonvital_fault[t],
        )
        feasible_high = min(generation_high, demand[t] - ess_low)
        if feasible_low <= feasible_high:
            generation = float(np.clip(demand[t], feasible_low, feasible_high))
        else:
            generation = float(np.clip(demand[t], generation_low, generation_high))

        split_low = max(g2_low, generation - g1_high)
        split_high = min(g2_high, generation - g1_low)
        target_g2 = float(normal_result["p_g2"][fault_start_hour + t])
        p_g2[t] = float(np.clip(target_g2, split_low, split_high))
        p_g1[t] = generation - p_g2[t]
        residual = demand[t] - generation
        p_ess[t] = float(np.clip(residual, ess_low, ess_high))
        p_shed[t] = float(
            np.clip(residual - p_ess[t], 0.0, p_nonvital_fault[t])
        )
        energy = float(ess_energy_trajectory(np.array([p_ess[t]]), energy)[0])
        previous_g1 = p_g1[t]
        previous_g2 = p_g2[t]

    return np.r_[p_g2, p_ess, p_shed]


def _initial_fault_population(
    rng,
    normal_result,
    p_vital,
    p_nonvital,
    p_pv,
    fault_start_hour,
    fault_duration,
    size,
):
    hours = np.arange(fault_start_hour, fault_start_hour + fault_duration)
    demand = (
        p_vital[hours]
        + p_nonvital[hours]
        + normal_result["p_propulsion"][hours]
        - p_pv[hours]
    )
    baseline = _fault_feasible_seed(
        normal_result,
        demand,
        p_nonvital[hours],
        fault_start_hour,
    )
    population = np.tile(baseline, (size, 1))
    population[:, :fault_duration] += rng.normal(
        0.0, 1.0, (size, fault_duration)
    )
    population[:, fault_duration : 2 * fault_duration] += rng.normal(
        0.0, 0.6, (size, fault_duration)
    )
    population[:, 2 * fault_duration :] += rng.normal(
        0.0, 0.3, (size, fault_duration)
    )
    population[0] = baseline
    return _repair_fault_positions(population, p_nonvital[hours])


def solve_fault(
    normal_result,
    p_vital,
    p_nonvital,
    p_pv,
    fault_start_hour=FAULT_START_HOUR,
    fault_duration=FAULT_DURATION,
    seed=PAIRED_SEEDS[0],
    np_max=NP_MAX,
    np_min=NP_MIN,
    iterations=MAX_ITERATIONS,
):
    _validate_ra_lshade_budget(np_max, np_min, iterations)
    fault_start_hour, fault_duration = _validate_fault_interval(
        fault_start_hour, fault_duration
    )
    hours = np.arange(fault_start_hour, fault_start_hour + fault_duration)
    rng = np.random.default_rng(seed)
    population = _initial_fault_population(
        rng,
        normal_result,
        p_vital,
        p_nonvital,
        p_pv,
        fault_start_hour,
        fault_duration,
        np_max,
    )
    pre_fault_energy = float(normal_result["energy_ess"][fault_start_hour - 1])
    repair = lambda positions: _repair_fault_positions(
        positions, p_nonvital[hours]
    )

    def evaluate(positions, _progress):
        return evaluate_fault(
            positions,
            normal_result,
            p_vital,
            p_nonvital,
            p_pv,
            fault_start_hour,
            fault_duration,
        )

    def strictly_feasible(results):
        return results["cv_f"] <= FEASIBILITY_TOLERANCE

    def history_metrics(results, best_result):
        if best_result is None:
            index = int(np.argmin(results["cv_f"]))
            reference = _candidate_result(results, index, len(results["cv_f"]))
        else:
            reference = best_result
        return {
            "best_cv": float(reference["cv_f"]),
            "total_load_shedding": float(reference["shed_energy"]),
            "peak_load_shedding": float(np.max(reference["p_shed"])),
            "minimum_ess_energy": float(np.min(reference["energy_ess"])),
        }

    best_position, _, history = _run_ra_lshade(
        rng,
        population,
        repair,
        evaluate,
        strictly_feasible,
        np_min,
        iterations,
        f"Fault-stage RA-LSHADE did not find a feasible solution for seed {seed}",
        history_metrics,
    )
    evaluated = evaluate(best_position, 1.0)
    best = _candidate_result(evaluated, 0, 1)
    result = {
        "hours": hours,
        "fault_start_hour": fault_start_hour,
        "fault_duration": fault_duration,
        "p_g1": best["p_g1"],
        "p_g2": best["p_g2"],
        "p_ess": best["p_ess"],
        "energy_ess": best["energy_ess"],
        "p_shed": best["p_shed"],
        "demand": best["demand"],
        "pre_fault_energy": pre_fault_energy,
        "total_cost": float(best["total_cost"]),
        "shed_energy": float(best["shed_energy"]),
        "load_retention": float(best["load_retention"]),
        "balance_residual": best["balance_residual"],
        "minimum_inequality_slack": float(best["minimum_inequality_slack"]),
        "cv_f": float(best["cv_f"]),
        "seed": int(seed),
        "convergence_history": history,
    }
    return result


def normal_validation(result, strategy):
    balance = result["p_g1"] + result["p_g2"] + result["p_ess"] - result["net_demand"]
    starts = np.arange(1, HOURS - RESERVE_HORIZON + 1)
    reserve_margin = result["energy_ess"][starts - 1] - (
        ESS_ENERGY_MIN + result["reserve_energy"][starts]
    )
    return {
        "distance_error": float(abs(np.sum(result["speeds"]) - DISTANCE_NM)),
        "balance_error": float(np.max(np.abs(balance))),
        "g1_ramp_max": float(np.max(np.abs(np.diff(result["p_g1"])))),
        "g2_ramp_max": float(np.max(np.abs(np.diff(result["p_g2"])))),
        "ess_ramp_max": float(np.max(np.abs(np.diff(result["p_ess"])))),
        "energy_min": float(np.min(result["energy_ess"])),
        "energy_max": float(np.max(result["energy_ess"])),
        "eeoi_max": float(np.max(result["eeoi"])),
        "reserve_margin_min": float(np.min(reserve_margin)) if strategy == "dynamic_reserve" else math.nan,
        "reserve_power_max": float(np.nanmax(result["reserve_power_max"])),
    }


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_normal_schedule(path, hours, p_vital, p_nonvital, p_pv, result):
    rows = []
    for t, hour in enumerate(hours):
        rows.append(
            {
                "hour": int(hour),
                "p_vital_mw": f"{p_vital[t]:.9f}",
                "p_nonvital_mw": f"{p_nonvital[t]:.9f}",
                "p_pv_mw": f"{p_pv[t]:.9f}",
                "v_kn": f"{result['speeds'][t]:.9f}",
                "p_propulsion_mw": f"{result['p_propulsion'][t]:.9f}",
                "p_g1_mw": f"{result['p_g1'][t]:.9f}",
                "p_g2_mw": f"{result['p_g2'][t]:.9f}",
                "p_ess_mw": f"{result['p_ess'][t]:.9f}",
                "energy_ess_mwh": f"{result['energy_ess'][t]:.9f}",
                "soc": f"{result['soc'][t]:.9f}",
                "eeoi": f"{result['eeoi'][t]:.9f}",
                "reserve_energy_mwh": "" if np.isnan(result["reserve_energy"][t]) else f"{result['reserve_energy'][t]:.9f}",
                "reserve_power_max_mw": "" if np.isnan(result["reserve_power_max"][t]) else f"{result['reserve_power_max'][t]:.9f}",
            }
        )
    _write_csv(path, rows)


def write_fault_schedule(path, p_vital, p_nonvital, p_pv, normal_result, fault_result):
    rows = []
    for i, hour in enumerate(fault_result["hours"]):
        rows.append(
            {
                "hour": int(hour),
                "p_vital_mw": f"{p_vital[hour]:.9f}",
                "p_nonvital_mw": f"{p_nonvital[hour]:.9f}",
                "p_pv_mw": f"{p_pv[hour]:.9f}",
                "v_kn": f"{normal_result['speeds'][hour]:.9f}",
                "p_propulsion_mw": f"{normal_result['p_propulsion'][hour]:.9f}",
                "p_g1_mw": f"{fault_result['p_g1'][i]:.9f}",
                "p_g2_mw": f"{fault_result['p_g2'][i]:.9f}",
                "p_ess_mw": f"{fault_result['p_ess'][i]:.9f}",
                "energy_ess_mwh": f"{fault_result['energy_ess'][i]:.9f}",
                "p_shed_mw": f"{fault_result['p_shed'][i]:.9f}",
                "balance_residual_mw": f"{fault_result['balance_residual'][i]:.3e}",
            }
        )
    _write_csv(path, rows)


def run_experiment(
    base_dir: Path,
    output_directory_name="results",
    result_filename="case3_result.md",
):
    root_dir = base_dir.parent
    output_dir = base_dir / output_directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    hours, p_vital, p_nonvital, p_pv = load_input_data(root_dir / "data.md")

    run_rows = []
    all_runs = {"no_reserve": [], "dynamic_reserve": []}
    for strategy in all_runs:
        for seed in PAIRED_SEEDS:
            started = time.perf_counter()
            position, result, history = solve_ra_lshade(
                p_vital, p_nonvital, p_pv, strategy, seed
            )
            elapsed = time.perf_counter() - started
            fault_started = time.perf_counter()
            fault = solve_fault(
                result,
                p_vital,
                p_nonvital,
                p_pv,
                seed=seed,
            )
            fault_elapsed = time.perf_counter() - fault_started
            validation = normal_validation(result, strategy)
            run_rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "cost": f"{result['total_cost']:.9f}",
                    "cv_base": f"{result['cv_base']:.3e}",
                    "cv_res": f"{result['cv_res']:.3e}",
                    "feasible": "true",
                    "runtime_seconds": f"{elapsed:.3f}",
                    "fault_cv": f"{fault['cv_f']:.3e}",
                    "fault_runtime_seconds": f"{fault_elapsed:.3f}",
                    "reserve_energy_max_mwh": f"{np.nanmax(result['reserve_energy']):.9f}",
                    "reserve_power_max_mw": f"{validation['reserve_power_max']:.9f}",
                }
            )
            all_runs[strategy].append(
                {
                    "seed": seed,
                    "position": position,
                    "result": result,
                    "history": history,
                    "fault": fault,
                }
            )
    _write_csv(output_dir / "normal_runs.csv", run_rows)

    selected = {}
    faults = {}
    for strategy, runs in all_runs.items():
        selected_run = min(runs, key=lambda run: run["result"]["total_cost"])
        selected[strategy] = selected_run
        faults[strategy] = selected_run["fault"]
        write_normal_schedule(
            output_dir / f"normal_{strategy}.csv",
            hours,
            p_vital,
            p_nonvital,
            p_pv,
            selected_run["result"],
        )
        write_fault_schedule(
            output_dir / f"fault_{strategy}.csv",
            p_vital,
            p_nonvital,
            p_pv,
            selected_run["result"],
            faults[strategy],
        )
        _write_csv(output_dir / f"convergence_{strategy}.csv", selected_run["history"])
        _write_csv(
            output_dir / f"convergence_fault_{strategy}.csv",
            selected_run["fault"]["convergence_history"],
        )

    summary_rows = []
    for strategy, runs in all_runs.items():
        costs = np.array([run["result"]["total_cost"] for run in runs])
        selected_result = selected[strategy]["result"]
        fault = faults[strategy]
        summary_rows.append(
            {
                "strategy": strategy,
                "runs": len(runs),
                "feasible_rate": "1.000000",
                "cost_mean": f"{np.mean(costs):.9f}",
                "cost_std": f"{np.std(costs, ddof=1):.9f}",
                "best_seed": selected[strategy]["seed"],
                "best_cost": f"{selected_result['total_cost']:.9f}",
                "reserve_energy_max_mwh": f"{np.nanmax(selected_result['reserve_energy']):.9f}",
                "normal_charge_input_mwh": f"{np.sum(np.maximum(0.0, -selected_result['p_ess'])):.9f}",
                "normal_discharge_output_mwh": f"{np.sum(np.maximum(0.0, selected_result['p_ess'])):.9f}",
                "normal_end_soc": f"{selected_result['soc'][-1]:.9f}",
                "fault_cost": f"{fault['total_cost']:.9f}",
                "fault_pre_soc": f"{fault['pre_fault_energy'] / ESS_ENERGY_MAX:.9f}",
                "fault_ess_output_mwh": f"{np.sum(np.maximum(0.0, fault['p_ess'])):.9f}",
                "fault_end_soc": f"{fault['energy_ess'][-1] / ESS_ENERGY_MAX:.9f}",
                "fault_g2_max_mw": f"{np.max(fault['p_g2']):.9f}",
                "shed_energy_mwh": f"{fault['shed_energy']:.9f}",
                "load_retention": f"{fault['load_retention']:.9f}",
            }
        )
    _write_csv(output_dir / "summary.csv", summary_rows)

    no_summary, dynamic_summary = summary_rows
    lines = [
        "# CASE3 计算结果",
        "",
        f"- ESS 容量：`{ESS_ENERGY_MAX:g} MWh`，初始 SOC：`0.5`，最低 SOC：`0.2`",
        f"- G1 故障容量：`10 MW -> {FAULT_G1_MAX:.0f} MW` (`alpha_F={FAULT_ALPHA}`)",
        f"- 备用设计时长：`{RESERVE_HORIZON} h`",
        f"- 实际故障：`hour={FAULT_START_HOUR}..{FAULT_START_HOUR + FAULT_DURATION - 1}` (`{FAULT_DURATION} h`)",
        f"- P1/P2 RA-LSHADE：每种策略 `{len(PAIRED_SEEDS)}` 次配对运行",
        "",
        "| 策略 | 可行率 | 最优正常成本 | 正常充电输入 (MWh) | 正常放电输出 (MWh) | 故障前 SOC | 故障结束 SOC | G2 故障最大出力 (MW) | 失负荷 (MWh) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| No reserve | {no_summary['feasible_rate']} | {no_summary['best_cost']} | {no_summary['normal_charge_input_mwh']} | {no_summary['normal_discharge_output_mwh']} | {no_summary['fault_pre_soc']} | {no_summary['fault_end_soc']} | {no_summary['fault_g2_max_mw']} | {no_summary['shed_energy_mwh']} |",
        f"| Dynamic reserve | {dynamic_summary['feasible_rate']} | {dynamic_summary['best_cost']} | {dynamic_summary['normal_charge_input_mwh']} | {dynamic_summary['normal_discharge_output_mwh']} | {dynamic_summary['fault_pre_soc']} | {dynamic_summary['fault_end_soc']} | {dynamic_summary['fault_g2_max_mw']} | {dynamic_summary['shed_energy_mwh']} |",
        "",
        f"Dynamic reserve 将故障前 SOC 从 `{no_summary['fault_pre_soc']}` 提高到 `{dynamic_summary['fault_pre_soc']}`，并将失负荷从 `{no_summary['shed_energy_mwh']} MWh` 降至 `{dynamic_summary['shed_energy_mwh']} MWh`。",
    ]
    (base_dir / result_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Run the Case 3 resilience experiment")
    parser.add_argument(
        "--ess-capacity-mwh",
        type=float,
        default=15.0,
        help="ESS energy capacity in MWh; initial/minimum SOC remain 0.5/0.2",
    )
    parser.add_argument(
        "--fault-g1-max-mw",
        type=float,
        default=0.0,
        help="Maximum available G1 power after the fault in MW",
    )
    args = parser.parse_args()
    configure_ess_capacity(args.ess_capacity_mwh)
    configure_fault_g1_max(args.fault_g1_max_mw)
    base_dir = Path(__file__).resolve().parent
    if math.isclose(ESS_ENERGY_MAX, 75.0) and math.isclose(FAULT_G1_MAX, 6.0):
        output_directory_name = "results"
        result_filename = "case3_result.md"
    elif math.isclose(FAULT_G1_MAX, 6.0):
        capacity_tag = f"{ESS_ENERGY_MAX:g}".replace(".", "p")
        output_directory_name = f"results_ess{capacity_tag}"
        result_filename = f"case3_result_ess{capacity_tag}.md"
    else:
        capacity_tag = f"{ESS_ENERGY_MAX:g}".replace(".", "p")
        fault_tag = f"{FAULT_G1_MAX:g}".replace(".", "p")
        output_directory_name = f"results_ess{capacity_tag}_g1fault{fault_tag}"
        result_filename = f"case3_result_ess{capacity_tag}_g1fault{fault_tag}.md"
    summaries = run_experiment(
        base_dir,
        output_directory_name=output_directory_name,
        result_filename=result_filename,
    )
    for row in summaries:
        print(
            f"{row['strategy']}: cost={row['best_cost']}, "
            f"shed={row['shed_energy_mwh']} MWh, retention={row['load_retention']}"
        )


if __name__ == "__main__":
    main()
