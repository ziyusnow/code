from pathlib import Path
import csv
import math
import re

import gurobipy as gp
from gurobipy import GRB


HOURS = 24
DISTANCE_NM = 240.0
TIME_LIMIT_SECONDS = 7200.0
TARGET_MIP_GAP = 1.0e-6
SOLVER_TOLERANCE = 1.0e-9
VALIDATION_TOLERANCE = 1.0e-7

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

    p_vital = [row[1] for row in service_rows]
    p_nonvital = [row[2] for row in service_rows]
    p_pv = [row[1] for row in pv_rows]
    if not all(math.isfinite(value) for value in p_vital + p_nonvital + p_pv):
        raise ValueError("Input data contains NaN or infinite values")
    if any(value < 0.0 for value in p_vital + p_nonvital):
        raise ValueError("Service loads cannot be negative")
    if any(value < 0.0 or value > 4.2 for value in p_pv):
        raise ValueError("PV power must remain within [0, 4.2] MW")

    return expected_hours, p_vital, p_nonvital, p_pv


def build_model(p_vital, p_nonvital, p_pv, log_path):
    model = gp.Model("case2_gurobi")
    model.Params.LogFile = str(log_path)
    model.Params.TimeLimit = TIME_LIMIT_SECONDS
    model.Params.MIPGap = TARGET_MIP_GAP
    model.Params.FeasibilityTol = SOLVER_TOLERANCE
    model.Params.IntFeasTol = SOLVER_TOLERANCE
    model.Params.OptimalityTol = SOLVER_TOLERANCE
    model.Params.NonConvex = 2
    model.Params.FuncNonlinear = 1

    periods = range(HOURS)
    speed = model.addVars(periods, lb=0.0, ub=11.0, name="speed")
    speed_cubed = model.addVars(periods, lb=0.0, ub=1331.0, name="speed_cubed")
    propulsion = model.addVars(periods, lb=0.0, ub=2.9282, name="propulsion")
    p_ess = model.addVars(periods, lb=-3.0, ub=3.0, name="p_ess")
    energy_change = model.addVars(
        periods,
        lb=-3.0 / ESS_EFFICIENCY_OUT,
        ub=3.0 * ESS_EFFICIENCY_IN,
        name="energy_change",
    )
    energy = model.addVars(
        periods, lb=ESS_ENERGY_MIN, ub=ESS_ENERGY_MAX, name="energy"
    )
    p_g1 = model.addVars(periods, lb=0.0, ub=10.0, name="p_g1")
    p_g2 = model.addVars(periods, lb=0.0, ub=20.0, name="p_g2")

    model.addConstr(gp.quicksum(speed[t] for t in periods) == DISTANCE_NM, name="distance")

    for t in periods:
        model.addGenConstrPow(
            speed[t],
            speed_cubed[t],
            3.0,
            name=f"speed_cube[{t}]",
            options="FuncNonlinear=1",
        )
        model.addConstr(
            propulsion[t] == 0.0022 * speed_cubed[t], name=f"propulsion_def[{t}]"
        )

        p_load = p_vital[t] + p_nonvital[t]
        model.addConstr(
            p_g1[t] + p_g2[t] + p_ess[t] + p_pv[t]
            == p_load + propulsion[t],
            name=f"power_balance[{t}]",
        )

        model.addGenConstrPWL(
            p_ess[t],
            energy_change[t],
            [-3.0, 0.0, 3.0],
            [3.0 * ESS_EFFICIENCY_IN, 0.0, -3.0 / ESS_EFFICIENCY_OUT],
            name=f"ess_efficiency[{t}]",
        )
        previous_energy = ESS_ENERGY_INITIAL if t == 0 else energy[t - 1]
        model.addConstr(
            energy[t] == previous_energy + energy_change[t],
            name=f"energy_balance[{t}]",
        )

        emissions = (
            13.5 * p_g1[t] * p_g1[t]
            + 10.0 * p_g1[t]
            + 450.0
            + 5.2 * p_g2[t] * p_g2[t]
            + 58.0 * p_g2[t]
            + 390.0
        )
        model.addQConstr(emissions <= 460.0 * speed[t], name=f"eeoi[{t}]")

    for t in range(1, HOURS):
        model.addConstr(p_g1[t] - p_g1[t - 1] <= 2.0, name=f"ramp_g1_up[{t}]")
        model.addConstr(p_g1[t - 1] - p_g1[t] <= 2.0, name=f"ramp_g1_down[{t}]")
        model.addConstr(p_g2[t] - p_g2[t - 1] <= 3.0, name=f"ramp_g2_up[{t}]")
        model.addConstr(p_g2[t - 1] - p_g2[t] <= 3.0, name=f"ramp_g2_down[{t}]")
        model.addConstr(p_ess[t] - p_ess[t - 1] <= 1.0, name=f"ramp_ess_up[{t}]")
        model.addConstr(p_ess[t - 1] - p_ess[t] <= 1.0, name=f"ramp_ess_down[{t}]")

    objective = gp.quicksum(
        13.0 * p_g1[t] * p_g1[t]
        + 12.0 * p_g1[t]
        + 430.0
        + 5.2 * p_g2[t] * p_g2[t]
        + 52.0 * p_g2[t]
        + 340.0
        + 4.3 * p_ess[t] * p_ess[t]
        + 1.0
        + 10.2 * p_pv[t]
        for t in periods
    )
    model.setObjective(objective, GRB.MINIMIZE)

    variables = {
        "speed": speed,
        "propulsion": propulsion,
        "p_ess": p_ess,
        "energy": energy,
        "p_g1": p_g1,
        "p_g2": p_g2,
    }
    return model, variables


def solution_rows(hours, p_vital, p_nonvital, p_pv, variables):
    rows = []
    for t, hour in enumerate(hours):
        speed = variables["speed"][t].X
        propulsion = variables["propulsion"][t].X
        p_ess = variables["p_ess"][t].X
        energy = variables["energy"][t].X
        p_g1 = variables["p_g1"][t].X
        p_g2 = variables["p_g2"][t].X
        emissions_g1 = 13.5 * p_g1**2 + 10.0 * p_g1 + 450.0
        emissions_g2 = 5.2 * p_g2**2 + 58.0 * p_g2 + 390.0
        cost_g1 = 13.0 * p_g1**2 + 12.0 * p_g1 + 430.0
        cost_g2 = 5.2 * p_g2**2 + 52.0 * p_g2 + 340.0
        cost_ess = 4.3 * p_ess**2 + 1.0
        cost_pv = 10.2 * p_pv[t]
        rows.append(
            {
                "hour": hour,
                "p_vital_mw": p_vital[t],
                "p_nonvital_mw": p_nonvital[t],
                "p_load_mw": p_vital[t] + p_nonvital[t],
                "p_pv_mw": p_pv[t],
                "v_kn": speed,
                "p_propulsion_mw": propulsion,
                "p_ess_mw": p_ess,
                "energy_ess_mwh": energy,
                "soc": energy / ESS_ENERGY_MAX,
                "p_g1_mw": p_g1,
                "p_g2_mw": p_g2,
                "eeoi": (emissions_g1 + emissions_g2) / (20.0 * speed),
                "co2_g1": emissions_g1,
                "co2_g2": emissions_g2,
                "cost_g1": cost_g1,
                "cost_g2": cost_g2,
                "cost_ess": cost_ess,
                "cost_pv": cost_pv,
            }
        )
    return rows


def write_schedule(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: str(value) if key == "hour" else f"{value:.12f}"
                    for key, value in row.items()
                }
            )


def read_schedule(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != HOURS:
        raise ValueError(f"Expected {HOURS} schedule rows, found {len(rows)}")
    parsed = []
    for row in rows:
        parsed.append(
            {
                key: int(value) if key == "hour" else float(value)
                for key, value in row.items()
            }
        )
    if [row["hour"] for row in parsed] != list(range(HOURS)):
        raise ValueError("Schedule hours must be 0 through 23")
    return parsed


def max_ramp(rows, field):
    return max(abs(rows[t][field] - rows[t - 1][field]) for t in range(1, HOURS))


def interval_violation(values, lower, upper):
    return max(max(lower - value, value - upper, 0.0) for value in values)


def validate_schedule(rows, solver_objective):
    speeds = [row["v_kn"] for row in rows]
    p_ess = [row["p_ess_mw"] for row in rows]
    p_g1 = [row["p_g1_mw"] for row in rows]
    p_g2 = [row["p_g2_mw"] for row in rows]
    energy = [row["energy_ess_mwh"] for row in rows]

    propulsion_residual = max(
        abs(row["p_propulsion_mw"] - 0.0022 * row["v_kn"] ** 3) for row in rows
    )
    power_balance_residual = max(
        abs(
            row["p_g1_mw"]
            + row["p_g2_mw"]
            + row["p_ess_mw"]
            + row["p_pv_mw"]
            - row["p_load_mw"]
            - row["p_propulsion_mw"]
        )
        for row in rows
    )

    recomputed_energy = []
    previous_energy = ESS_ENERGY_INITIAL
    for value in p_ess:
        change = (
            -value * ESS_EFFICIENCY_IN
            if value <= 0.0
            else -value / ESS_EFFICIENCY_OUT
        )
        previous_energy += change
        recomputed_energy.append(previous_energy)
    energy_residual = max(
        abs(actual - expected) for actual, expected in zip(energy, recomputed_energy)
    )

    emissions_g1 = [13.5 * value**2 + 10.0 * value + 450.0 for value in p_g1]
    emissions_g2 = [5.2 * value**2 + 58.0 * value + 390.0 for value in p_g2]
    eeoi = [
        (value_g1 + value_g2) / (20.0 * speed)
        for value_g1, value_g2, speed in zip(emissions_g1, emissions_g2, speeds)
    ]
    objective = sum(
        13.0 * row["p_g1_mw"] ** 2
        + 12.0 * row["p_g1_mw"]
        + 430.0
        + 5.2 * row["p_g2_mw"] ** 2
        + 52.0 * row["p_g2_mw"]
        + 340.0
        + 4.3 * row["p_ess_mw"] ** 2
        + 1.0
        + 10.2 * row["p_pv_mw"]
        for row in rows
    )

    metrics = {
        "objective_recomputed": objective,
        "objective_error": abs(objective - solver_objective),
        "total_emissions": sum(emissions_g1) + sum(emissions_g2),
        "distance_nm": sum(speeds),
        "distance_error_nm": abs(sum(speeds) - DISTANCE_NM),
        "propulsion_residual_mw": propulsion_residual,
        "power_balance_residual_mw": power_balance_residual,
        "energy_recursion_residual_mwh": energy_residual,
        "speed_min_kn": min(speeds),
        "speed_max_kn": max(speeds),
        "p_g1_min_mw": min(p_g1),
        "p_g1_max_mw": max(p_g1),
        "p_g2_min_mw": min(p_g2),
        "p_g2_max_mw": max(p_g2),
        "p_ess_min_mw": min(p_ess),
        "p_ess_max_mw": max(p_ess),
        "ramp_g1_max_mw_per_h": max_ramp(rows, "p_g1_mw"),
        "ramp_g2_max_mw_per_h": max_ramp(rows, "p_g2_mw"),
        "ramp_ess_max_mw_per_h": max_ramp(rows, "p_ess_mw"),
        "energy_min_mwh": min(energy),
        "energy_max_mwh": max(energy),
        "energy_final_mwh": energy[-1],
        "soc_min": min(value / ESS_ENERGY_MAX for value in energy),
        "soc_max": max(value / ESS_ENERGY_MAX for value in energy),
        "soc_final": energy[-1] / ESS_ENERGY_MAX,
        "eeoi_max": max(eeoi),
    }

    failures = []
    equality_checks = {
        "distance": metrics["distance_error_nm"],
        "propulsion relation": propulsion_residual,
        "power balance": power_balance_residual,
        "ESS energy recursion": energy_residual,
        "objective recomputation": metrics["objective_error"],
    }
    for name, violation in equality_checks.items():
        if violation > VALIDATION_TOLERANCE:
            failures.append(f"{name} residual {violation:.3e} exceeds tolerance")

    bound_checks = {
        "speed bounds": interval_violation(speeds, 0.0, 11.0),
        "G1 bounds": interval_violation(p_g1, 0.0, 10.0),
        "G2 bounds": interval_violation(p_g2, 0.0, 20.0),
        "ESS power bounds": interval_violation(p_ess, -3.0, 3.0),
        "ESS energy bounds": interval_violation(energy, 15.0, 75.0),
        "G1 ramp": max(0.0, metrics["ramp_g1_max_mw_per_h"] - 2.0),
        "G2 ramp": max(0.0, metrics["ramp_g2_max_mw_per_h"] - 3.0),
        "ESS ramp": max(0.0, metrics["ramp_ess_max_mw_per_h"] - 1.0),
        "EEOI": max(0.0, metrics["eeoi_max"] - 23.0),
    }
    for name, violation in bound_checks.items():
        if violation > VALIDATION_TOLERANCE:
            failures.append(f"{name} violation {violation:.3e} exceeds tolerance")

    if not all(math.isfinite(value) for value in metrics.values()):
        failures.append("validation metrics contain NaN or infinite values")
    return metrics, failures


def status_name(status):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def write_result(path, model, metrics):
    status = status_name(model.Status)
    proven_optimal = model.Status == GRB.OPTIMAL
    conclusion = (
        "在设定的相对间隙容差内，Gurobi 已证明该解为全局最优解。"
        if proven_optimal
        else "Gurobi 已返回可行解，但尚未在设定容差内证明其为全局最优解。"
    )
    lines = [
        "# CASE2 Gurobi 求解结果",
        "",
        "## 求解状态",
        "",
        f"- Gurobi 版本：`{'.'.join(map(str, gp.gurobi.version()))}`",
        f"- 状态：`{status}`",
        f"- 目标值：`{model.ObjVal:.12f}`",
        f"- 全局下界：`{model.ObjBound:.12f}`",
        f"- 相对间隙：`{model.MIPGap:.6e}`",
        f"- 运行时间：`{model.Runtime:.3f} s`",
        f"- 搜索节点数：`{model.NodeCount:.0f}`",
        f"- 目标相对间隙：`{TARGET_MIP_GAP:.1e}`",
        f"- 时间上限：`{TIME_LIMIT_SECONDS:.0f} s`",
        f"- 最优性结论：{conclusion}",
        "",
        "## 独立复算",
        "",
        f"- 复算总成本：`{metrics['objective_recomputed']:.12f}`",
        f"- 目标值复算误差：`{metrics['objective_error']:.3e}`",
        f"- 总排放：`{metrics['total_emissions']:.12f}`",
        f"- 总航程：`{metrics['distance_nm']:.12f} nm`",
        f"- 航程误差：`{metrics['distance_error_nm']:.3e} nm`",
        f"- 推进立方关系最大残差：`{metrics['propulsion_residual_mw']:.3e} MW`",
        f"- 最大功率平衡残差：`{metrics['power_balance_residual_mw']:.3e} MW`",
        f"- ESS 能量递推最大残差：`{metrics['energy_recursion_residual_mwh']:.3e} MWh`",
        f"- 航速范围：`[{metrics['speed_min_kn']:.9f}, {metrics['speed_max_kn']:.9f}] kn`",
        f"- G1 出力范围：`[{metrics['p_g1_min_mw']:.9f}, {metrics['p_g1_max_mw']:.9f}] MW`",
        f"- G2 出力范围：`[{metrics['p_g2_min_mw']:.9f}, {metrics['p_g2_max_mw']:.9f}] MW`",
        f"- ESS 功率范围：`[{metrics['p_ess_min_mw']:.9f}, {metrics['p_ess_max_mw']:.9f}] MW`",
        f"- G1 最大爬坡：`{metrics['ramp_g1_max_mw_per_h']:.9f} MW/h`",
        f"- G2 最大爬坡：`{metrics['ramp_g2_max_mw_per_h']:.9f} MW/h`",
        f"- ESS 最大爬坡：`{metrics['ramp_ess_max_mw_per_h']:.9f} MW/h`",
        f"- ESS 能量范围：`[{metrics['energy_min_mwh']:.9f}, {metrics['energy_max_mwh']:.9f}] MWh`",
        f"- 终端 ESS 能量：`{metrics['energy_final_mwh']:.9f} MWh`",
        f"- SOC 范围：`[{metrics['soc_min']:.9f}, {metrics['soc_max']:.9f}]`",
        f"- 终端 SOC：`{metrics['soc_final']:.9f}`",
        f"- 最大 EEOI：`{metrics['eeoi_max']:.9f}`",
        "",
        "输入仅来自 `data.md`；未读取或使用既有 Case 2 调度与求解结果。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data.md"
    log_path = base_dir / "case2_gurobi.log"
    schedule_path = base_dir / "case2_gurobi_schedule.csv"
    result_path = base_dir / "case2_gurobi_result.md"

    hours, p_vital, p_nonvital, p_pv = load_input_data(data_path)
    model, variables = build_model(p_vital, p_nonvital, p_pv, log_path)
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(
            f"Gurobi returned {status_name(model.Status)} without a feasible solution"
        )

    rows = solution_rows(hours, p_vital, p_nonvital, p_pv, variables)
    write_schedule(schedule_path, rows)
    exported_rows = read_schedule(schedule_path)
    metrics, failures = validate_schedule(exported_rows, model.ObjVal)
    if failures:
        raise RuntimeError("Independent validation failed: " + "; ".join(failures))
    write_result(result_path, model, metrics)

    print(f"status={status_name(model.Status)}")
    print(f"objective={model.ObjVal:.12f}")
    print(f"best_bound={model.ObjBound:.12f}")
    print(f"relative_gap={model.MIPGap:.6e}")
    print(f"runtime_seconds={model.Runtime:.3f}")
    print(f"validation=OK (tolerance={VALIDATION_TOLERANCE:.1e})")


if __name__ == "__main__":
    main()
