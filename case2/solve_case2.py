from pathlib import Path
import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def evaluate(positions, p_vital, p_nonvital, p_pv):
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
    fitness = total_cost + PENALTY_LAMBDA * cv

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


def is_better(cost, cv, reference_cost, reference_cv):
    feasible = cv <= FEASIBILITY_TOLERANCE
    reference_feasible = reference_cv <= FEASIBILITY_TOLERANCE
    both_infeasible = ~feasible & ~reference_feasible
    return (
        (feasible & ~reference_feasible)
        | (feasible & reference_feasible & (cost < reference_cost))
        | (both_infeasible & (cv < reference_cv))
        | (both_infeasible & (cv == reference_cv) & (cost < reference_cost))
    )


def best_index(cost, cv):
    feasible = cv <= FEASIBILITY_TOLERANCE
    if np.any(feasible):
        candidates = np.flatnonzero(feasible)
        return int(candidates[np.argmin(cost[candidates])])
    return int(np.lexsort((cost, cv))[0])


def protected_denominator(value):
    return max(abs(value), NUMERICAL_EPSILON)


def solve(p_vital, p_nonvital, p_pv):
    rng = np.random.default_rng(SEED)
    speeds = project_speeds(rng.uniform(0.0, V_MAX, (POPULATION_SIZE, HOURS)))
    q_ess = rng.uniform(0.0, 1.0, (POPULATION_SIZE, HOURS))
    q_generator = rng.uniform(0.0, 1.0, (POPULATION_SIZE, HOURS))
    positions = np.hstack((speeds, q_ess, q_generator))

    velocity_limits = np.r_[
        np.full(HOURS, VELOCITY_CLAMP_V),
        np.full(HOURS, VELOCITY_CLAMP_QE),
        np.full(HOURS, VELOCITY_CLAMP_QG),
    ]
    velocities = np.zeros_like(positions)
    previous_positions = positions.copy()
    previous_velocities = velocities.copy()

    results = evaluate(positions, p_vital, p_nonvital, p_pv)
    personal_best = positions.copy()
    personal_best_cost = results["total_cost"].copy()
    personal_best_cv = results["cv"].copy()
    global_index = best_index(personal_best_cost, personal_best_cv)
    global_best = personal_best[global_index].copy()
    global_best_cost = float(personal_best_cost[global_index])
    global_best_cv = float(personal_best_cv[global_index])

    history = [
        {
            "iteration": 0,
            "best_cost": global_best_cost,
            "best_cv": global_best_cv,
            "best_fitness": global_best_cost + PENALTY_LAMBDA * global_best_cv,
            "feasible": global_best_cv <= FEASIBILITY_TOLERANCE,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        fitness = results["fitness"]
        f_min = float(np.min(fitness))
        f_mean = float(np.mean(fitness))
        f_max = float(np.max(fitness))

        omega = np.empty(POPULATION_SIZE)
        upper_group = fitness >= f_mean
        omega[upper_group] = OMEGA_MIN - (OMEGA_MIN - OMEGA_MAX) * (
            (fitness[upper_group] - f_mean)
            / protected_denominator(f_max - f_mean)
        )
        omega[~upper_group] = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * (
            (fitness[~upper_group] - f_min)
            / protected_denominator(f_mean - f_min)
        )

        c1 = np.full(POPULATION_SIZE, C_MIN)
        lower_group = fitness <= f_mean
        c1[lower_group] = C_MAX + (C_MAX - C_MIN) * (
            (fitness[lower_group] - f_min)
            / protected_denominator(f_mean - f_min)
        )
        c2 = 2.0 - c1

        population_center = positions.mean(axis=0)
        blended_positions = ALPHA * positions + (1.0 - ALPHA) * previous_positions
        r3 = rng.uniform(0.0, 1.0, positions.shape)
        new_velocities = (
            omega[:, None]
            * (ZETA * velocities + (1.0 - ZETA) * previous_velocities)
            + c1[:, None] * (personal_best - blended_positions)
            + c2[:, None] * (global_best[None, :] - blended_positions)
            + C3 * r3 * (population_center[None, :] - blended_positions)
        )
        new_velocities = np.clip(new_velocities, -velocity_limits, velocity_limits)

        new_positions = positions + new_velocities
        new_positions[:, :HOURS] = project_speeds(new_positions[:, :HOURS])
        new_positions[:, HOURS : 2 * HOURS] = np.clip(
            new_positions[:, HOURS : 2 * HOURS], 0.0, 1.0
        )
        new_positions[:, 2 * HOURS :] = np.clip(
            new_positions[:, 2 * HOURS :], 0.0, 1.0
        )

        previous_positions, positions = positions, new_positions
        previous_velocities, velocities = velocities, new_velocities
        results = evaluate(positions, p_vital, p_nonvital, p_pv)

        better = is_better(
            results["total_cost"],
            results["cv"],
            personal_best_cost,
            personal_best_cv,
        )
        personal_best[better] = positions[better]
        personal_best_cost[better] = results["total_cost"][better]
        personal_best_cv[better] = results["cv"][better]

        global_index = best_index(personal_best_cost, personal_best_cv)
        global_best = personal_best[global_index].copy()
        global_best_cost = float(personal_best_cost[global_index])
        global_best_cv = float(personal_best_cv[global_index])
        history.append(
            {
                "iteration": iteration,
                "best_cost": global_best_cost,
                "best_cv": global_best_cv,
                "best_fitness": global_best_cost
                + PENALTY_LAMBDA * global_best_cv,
                "feasible": global_best_cv <= FEASIBILITY_TOLERANCE,
            }
        )

        if iteration == 1 or iteration % 50 == 0:
            print(
                f"iteration={iteration:3d} "
                f"cost={global_best_cost:.6f} cv={global_best_cv:.3e}"
            )

    return global_best, evaluate(global_best, p_vital, p_nonvital, p_pv), history


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


def validation_failures(result, metrics):
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
    if metrics["cv_total"] > FEASIBILITY_TOLERANCE:
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
