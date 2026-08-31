from pathlib import Path
import csv
import re

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
NUMERICAL_EPSILON = 1.0e-12
FEASIBILITY_TOLERANCE = 1.0e-10

V_MAX = 11.0
DISTANCE = 240.0
VELOCITY_CLAMP_V = 2.2
VELOCITY_CLAMP_Q = 0.2


def load_service_loads(path):
    rows = []
    row_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_pattern.match(line)
        if match:
            hour, vital, nonvital = match.groups()
            rows.append((int(hour), float(vital) + float(nonvital)))

    if len(rows) != HOURS or [row[0] for row in rows] != list(range(HOURS)):
        raise ValueError("data.md must contain exactly the hourly rows 0 through 23")

    return np.array([row[0] for row in rows]), np.array([row[1] for row in rows])


def project_speeds(values):
    """Project each row onto 0 <= V <= 11 and sum(V) = 240."""
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1
    rows = values.reshape(1, -1) if one_dimensional else values
    if rows.shape[1] != HOURS:
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


def evaluate(positions, p_load):
    speeds = positions[:, :HOURS]
    allocation = positions[:, HOURS:]

    p_propulsion = 0.0022 * speeds**3
    p_demand = p_load[None, :] + p_propulsion
    lower_g1 = np.maximum(0.0, p_demand - 20.0)
    upper_g1 = np.minimum(10.0, p_demand)
    p_g1 = lower_g1 + allocation * (upper_g1 - lower_g1)
    p_g2 = p_demand - p_g1

    cost_g1 = 13.0 * p_g1**2 + 12.0 * p_g1 + 430.0
    cost_g2 = 5.2 * p_g2**2 + 52.0 * p_g2 + 340.0
    total_cost = (cost_g1 + cost_g2).sum(axis=1)

    ramp_g1 = np.abs(np.diff(p_g1, axis=1))
    ramp_g2 = np.abs(np.diff(p_g2, axis=1))
    cv_r = (
        np.maximum(0.0, (ramp_g1 - 2.0) / 2.0).sum(axis=1)
        + np.maximum(0.0, (ramp_g2 - 3.0) / 3.0).sum(axis=1)
    )

    emissions_g1 = 13.5 * p_g1**2 + 10.0 * p_g1 + 450.0
    emissions_g2 = 5.2 * p_g2**2 + 58.0 * p_g2 + 390.0
    eeoi = (emissions_g1 + emissions_g2) / (
        20.0 * np.maximum(speeds, 1.0e-6)
    )
    cv_e = np.maximum(0.0, (eeoi - 23.0) / 23.0).sum(axis=1)
    cv_d = np.maximum(0.0, (p_demand - 30.0) / 30.0).sum(axis=1)
    cv = cv_r + cv_e + cv_d
    fitness = total_cost + PENALTY_LAMBDA * cv

    return {
        "speeds": speeds,
        "p_propulsion": p_propulsion,
        "p_demand": p_demand,
        "p_g1": p_g1,
        "p_g2": p_g2,
        "cost_g1": cost_g1,
        "cost_g2": cost_g2,
        "eeoi": eeoi,
        "total_cost": total_cost,
        "cv_r": cv_r,
        "cv_e": cv_e,
        "cv_d": cv_d,
        "cv": cv,
        "fitness": fitness,
    }


def is_better(cost, cv, reference_cost, reference_cv):
    feasible = cv <= FEASIBILITY_TOLERANCE
    reference_feasible = reference_cv <= FEASIBILITY_TOLERANCE
    return (
        (feasible & ~reference_feasible)
        | (feasible & reference_feasible & (cost < reference_cost))
        | (~feasible & ~reference_feasible & (cv < reference_cv))
    )


def best_index(cost, cv):
    feasible = cv <= FEASIBILITY_TOLERANCE
    if np.any(feasible):
        candidates = np.flatnonzero(feasible)
        return candidates[np.argmin(cost[candidates])]
    return int(np.argmin(cv))


def protected_denominator(value):
    return max(abs(value), NUMERICAL_EPSILON)


def solve(p_load):
    rng = np.random.default_rng(SEED)

    speeds = project_speeds(rng.uniform(0.0, V_MAX, (POPULATION_SIZE, HOURS)))
    allocation = rng.uniform(0.0, 1.0, (POPULATION_SIZE, HOURS))
    positions = np.hstack((speeds, allocation))

    velocity_limits = np.r_[
        np.full(HOURS, VELOCITY_CLAMP_V),
        np.full(HOURS, VELOCITY_CLAMP_Q),
    ]
    velocities = rng.uniform(-velocity_limits, velocity_limits, positions.shape)
    previous_positions = positions.copy()
    previous_velocities = velocities.copy()

    results = evaluate(positions, p_load)
    personal_best = positions.copy()
    personal_best_cost = results["total_cost"].copy()
    personal_best_cv = results["cv"].copy()
    global_index = best_index(personal_best_cost, personal_best_cv)
    global_best = personal_best[global_index].copy()

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
        new_positions[:, HOURS:] = np.clip(new_positions[:, HOURS:], 0.0, 1.0)

        previous_positions, positions = positions, new_positions
        previous_velocities, velocities = velocities, new_velocities
        results = evaluate(positions, p_load)

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
        candidate = personal_best[global_index]
        candidate_cost = personal_best_cost[global_index]
        candidate_cv = personal_best_cv[global_index]
        current_global = evaluate(global_best[None, :], p_load)
        if is_better(
            np.array([candidate_cost]),
            np.array([candidate_cv]),
            current_global["total_cost"],
            current_global["cv"],
        )[0]:
            global_best = candidate.copy()

        if iteration == 1 or iteration % 50 == 0:
            best = evaluate(global_best[None, :], p_load)
            print(
                f"iteration={iteration:3d} "
                f"cost={best['total_cost'][0]:.6f} cv={best['cv'][0]:.3e}"
            )

    return global_best, evaluate(global_best[None, :], p_load)


def write_schedule(path, hours, p_load, result):
    fieldnames = [
        "hour",
        "p_load_mw",
        "v_kn",
        "p_propulsion_mw",
        "p_g1_mw",
        "p_g2_mw",
        "eeoi",
        "cost_g1",
        "cost_g2",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, hour in enumerate(hours):
            writer.writerow(
                {
                    "hour": int(hour),
                    "p_load_mw": f"{p_load[index]:.6f}",
                    "v_kn": f"{result['speeds'][0, index]:.9f}",
                    "p_propulsion_mw": f"{result['p_propulsion'][0, index]:.9f}",
                    "p_g1_mw": f"{result['p_g1'][0, index]:.9f}",
                    "p_g2_mw": f"{result['p_g2'][0, index]:.9f}",
                    "eeoi": f"{result['eeoi'][0, index]:.9f}",
                    "cost_g1": f"{result['cost_g1'][0, index]:.9f}",
                    "cost_g2": f"{result['cost_g2'][0, index]:.9f}",
                }
            )


def validation_metrics(result):
    speeds = result["speeds"][0]
    p_g1 = result["p_g1"][0]
    p_g2 = result["p_g2"][0]
    p_demand = result["p_demand"][0]
    return {
        "distance_nm": speeds.sum(),
        "power_balance_residual_mw": np.max(np.abs(p_g1 + p_g2 - p_demand)),
        "p_g1_min_mw": p_g1.min(),
        "p_g1_max_mw": p_g1.max(),
        "p_g2_min_mw": p_g2.min(),
        "p_g2_max_mw": p_g2.max(),
        "ramp_g1_max_mw_per_h": np.max(np.abs(np.diff(p_g1))),
        "ramp_g2_max_mw_per_h": np.max(np.abs(np.diff(p_g2))),
        "eeoi_max": result["eeoi"][0].max(),
        "cv_r": result["cv_r"][0],
        "cv_e": result["cv_e"][0],
        "cv_d": result["cv_d"][0],
        "cv_total": result["cv"][0],
    }


def write_summary(path, result, metrics):
    lines = [
        "# CASE1 调度优化结果",
        "",
        f"- 随机种子：`{SEED}`",
        f"- MPPSO 参数：种群 `{POPULATION_SIZE}`，迭代 `{MAX_ITERATIONS}`",
        f"- 本次 MPPSO 最佳总运行成本：`{result['total_cost'][0]:.6f}`",
        "- 时段标签：CSV 的 `hour=0..23` 沿用 `data.md`",
        f"- 总航程：`{metrics['distance_nm']:.9f} nm`",
        f"- 最大功率平衡残差：`{metrics['power_balance_residual_mw']:.3e} MW`",
        f"- G1 出力范围：`[{metrics['p_g1_min_mw']:.6f}, {metrics['p_g1_max_mw']:.6f}] MW`",
        f"- G2 出力范围：`[{metrics['p_g2_min_mw']:.6f}, {metrics['p_g2_max_mw']:.6f}] MW`",
        f"- G1 最大爬坡：`{metrics['ramp_g1_max_mw_per_h']:.6f} MW/h`",
        f"- G2 最大爬坡：`{metrics['ramp_g2_max_mw_per_h']:.6f} MW/h`",
        f"- 最大 EEOI：`{metrics['eeoi_max']:.6f}`",
        f"- 约束违反量：`CV_R={metrics['cv_r']:.3e}`，`CV_E={metrics['cv_e']:.3e}`，`CV_D={metrics['cv_d']:.3e}`",
        f"- 总约束违反量：`{metrics['cv_total']:.3e}`",
        "",
        "可行性结论："
        + ("满足全部模型约束。" if metrics["cv_total"] <= FEASIBILITY_TOLERANCE else "未找到可行解。"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    hours, p_load = load_service_loads(base_dir / "data.md")
    _, result = solve(p_load)
    metrics = validation_metrics(result)

    if metrics["cv_total"] > FEASIBILITY_TOLERANCE:
        raise RuntimeError(
            f"MPPSO did not find a feasible solution: CV={metrics['cv_total']:.6e}"
        )

    write_schedule(base_dir / "case1_schedule.csv", hours, p_load, result)
    write_summary(base_dir / "case1_result.md", result, metrics)
    print(f"best_total_cost={result['total_cost'][0]:.6f}")
    print(f"total_cv={metrics['cv_total']:.3e}")


if __name__ == "__main__":
    main()
