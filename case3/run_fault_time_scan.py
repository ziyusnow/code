from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from case3 import solve_case3 as model
except ModuleNotFoundError:
    import solve_case3 as model


FAULT_START_HOURS = (5, 8, 11, 14, 17, 20)
FAULT_DURATION = 4
ESS_CAPACITY_MWH = 15.0
G2_MAX_MW = 20.0
G1_FAULT_MAX_MW = 0.0
RETRY_SEEDS = (20260826, 20260827, 20260828, 20260829, 20260830)


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve_normal_plans(p_vital, p_nonvital, p_pv):
    plans = {}
    for strategy in ("no_reserve", "dynamic_reserve"):
        last_error = None
        for seed in RETRY_SEEDS:
            try:
                _, result, _ = model.solve_ra_lshade(
                    p_vital,
                    p_nonvital,
                    p_pv,
                    strategy,
                    seed,
                    np_max=model.NP_MAX,
                    np_min=model.NP_MIN,
                    iterations=model.MAX_ITERATIONS,
                )
                plans[strategy] = {"seed": seed, "result": result}
                break
            except RuntimeError as error:
                last_error = error
        else:
            raise RuntimeError(
                f"P1 RA-LSHADE did not find a feasible {strategy} solution"
            ) from last_error
    return plans


def solve_fault_with_retries(
    normal_result, p_vital, p_nonvital, p_pv, start_hour
):
    last_error = None
    for seed in RETRY_SEEDS:
        try:
            fault = model.solve_fault(
                normal_result,
                p_vital,
                p_nonvital,
                p_pv,
                fault_start_hour=start_hour,
                fault_duration=FAULT_DURATION,
                seed=seed,
                np_max=model.NP_MAX,
                np_min=model.NP_MIN,
                iterations=model.MAX_ITERATIONS,
            )
            return seed, fault
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        f"P2 RA-LSHADE did not find a feasible solution at hour {start_hour}"
    ) from last_error


def minimum_shortfall_diagnostic(
    normal_result, p_vital, p_nonvital, p_pv, start_hour
):
    hours = np.arange(start_hour, start_hour + FAULT_DURATION)
    energy = float(normal_result["energy_ess"][start_hour - 1])
    p_g2 = min(
        model.G2_MAX,
        float(normal_result["p_g2"][start_hour - 1]) + model.G2_RAMP,
    )
    g2_available = []
    for index in range(FAULT_DURATION):
        if index > 0:
            p_g2 = min(model.G2_MAX, p_g2 + model.G2_RAMP)
        g2_available.append(p_g2)
    critical_demand = (
        p_vital[hours]
        + normal_result["p_propulsion"][hours]
        - p_pv[hours]
    )
    critical_deficit = np.maximum(
        0.0,
        critical_demand - model.FAULT_G1_MAX - np.array(g2_available),
    )
    critical_ess_need = np.minimum(model.ESS_POWER_MAX, critical_deficit)

    nonvital_shed = []
    vital_shed = []
    ess_output = []
    g2_output = []
    remaining_energy = []
    for index, hour in enumerate(hours):
        p_g2 = g2_available[index]
        maximum_energy_output = max(
            0.0, (energy - model.ESS_ENERGY_MIN) * model.ESS_EFFICIENCY_OUT
        )
        critical_output = min(
            model.ESS_POWER_MAX,
            maximum_energy_output,
            critical_deficit[index],
        )
        future_critical_need = float(np.sum(critical_ess_need[index + 1 :]))
        surplus_energy_output = max(
            0.0,
            maximum_energy_output - critical_output - future_critical_need,
        )
        supply_after_critical = max(
            0.0,
            model.FAULT_G1_MAX
            + p_g2
            + critical_output
            - critical_demand[index],
        )
        nonvital_deficit = max(0.0, p_nonvital[hour] - supply_after_critical)
        nonvital_output = min(
            model.ESS_POWER_MAX - critical_output,
            surplus_energy_output,
            nonvital_deficit,
        )
        p_ess = critical_output + nonvital_output
        shed_nonvital = max(0.0, nonvital_deficit - nonvital_output)
        shed_vital = max(0.0, critical_deficit[index] - critical_output)
        energy -= p_ess / model.ESS_EFFICIENCY_OUT * model.DELTA_T
        nonvital_shed.append(shed_nonvital)
        vital_shed.append(shed_vital)
        ess_output.append(p_ess)
        g2_output.append(p_g2)
        remaining_energy.append(energy)
    return {
        "hours": hours,
        "p_g2": np.array(g2_output),
        "p_ess": np.array(ess_output),
        "energy_ess": np.array(remaining_energy),
        "p_shed": np.array(nonvital_shed),
        "p_vital_shortfall": np.array(vital_shed),
        "pre_fault_energy": float(normal_result["energy_ess"][start_hour - 1]),
        "shed_energy": float(np.sum(nonvital_shed) * model.DELTA_T),
        "vital_shortfall_energy": float(np.sum(vital_shed) * model.DELTA_T),
    }


def run_scan(base_dir):
    model.configure_ess_capacity(ESS_CAPACITY_MWH)
    model.configure_g2_max(G2_MAX_MW)
    model.configure_fault_g1_max(G1_FAULT_MAX_MW)
    _, p_vital, p_nonvital, p_pv = model.load_input_data(base_dir.parent / "data.md")
    plans = solve_normal_plans(p_vital, p_nonvital, p_pv)

    result_rows = []
    hourly_rows = []
    for strategy, plan in plans.items():
        p1_seed = plan["seed"]
        normal_result = plan["result"]
        normal_validation = model.normal_validation(normal_result, strategy)
        for start_hour in FAULT_START_HOURS:
            status = "RA_LSHADE_FEASIBLE"
            p2_seed = None
            fault_cv = np.nan
            try:
                p2_seed, fault = solve_fault_with_retries(
                    normal_result,
                    p_vital,
                    p_nonvital,
                    p_pv,
                    start_hour,
                )
                fault["p_vital_shortfall"] = np.zeros(FAULT_DURATION)
                fault["vital_shortfall_energy"] = 0.0
                fault_cv = fault["cv_f"]
                fault_balance_error = float(
                    np.max(np.abs(fault["balance_residual"]))
                )
                minimum_slack = fault["minimum_inequality_slack"]
            except RuntimeError as search_error:
                fault = minimum_shortfall_diagnostic(
                    normal_result, p_vital, p_nonvital, p_pv, start_hour
                )
                if fault["vital_shortfall_energy"] <= 1.0e-8:
                    raise search_error
                status = "INFEASIBLE_CRITICAL_LOAD"
                fault_balance_error = np.nan
                minimum_slack = np.nan
            nonvital_energy = float(
                np.sum(p_nonvital[fault["hours"]]) * model.DELTA_T
            )
            loss_rate = fault["shed_energy"] / nonvital_energy
            vital_energy = float(
                np.sum(p_vital[fault["hours"]]) * model.DELTA_T
            )
            vital_loss_rate = fault["vital_shortfall_energy"] / vital_energy
            result_rows.append(
                {
                    "strategy": strategy,
                    "status": status,
                    "p1_seed": p1_seed,
                    "p2_seed": "" if p2_seed is None else p2_seed,
                    "fault_start_hour": start_hour,
                    "fault_end_hour": start_hour + FAULT_DURATION - 1,
                    "fault_duration_h": FAULT_DURATION,
                    "normal_cost": normal_result["total_cost"],
                    "normal_cv_base": normal_result["cv_base"],
                    "normal_cv_res": normal_result["cv_res"],
                    "fault_cv": fault_cv,
                    "fault_pre_soc": fault["pre_fault_energy"] / model.ESS_ENERGY_MAX,
                    "fault_end_soc": fault["energy_ess"][-1] / model.ESS_ENERGY_MAX,
                    "fault_ess_output_mwh": float(
                        np.sum(np.maximum(0.0, fault["p_ess"])) * model.DELTA_T
                    ),
                    "fault_g2_max_mw": float(np.max(fault["p_g2"])),
                    "nonvital_energy_mwh": nonvital_energy,
                    "shed_energy_mwh": fault["shed_energy"],
                    "loss_rate": loss_rate,
                    "vital_shortfall_energy_mwh": fault[
                        "vital_shortfall_energy"
                    ],
                    "vital_loss_rate": vital_loss_rate,
                    "load_retention": 1.0 - loss_rate,
                    "normal_balance_error_mw": normal_validation["balance_error"],
                    "fault_balance_error_mw": fault_balance_error,
                    "minimum_inequality_slack": minimum_slack,
                }
            )
            for index, hour in enumerate(fault["hours"]):
                hourly_rows.append(
                    {
                        "strategy": strategy,
                        "fault_start_hour": start_hour,
                        "hour": int(hour),
                        "p_vital_mw": p_vital[hour],
                        "p_nonvital_mw": p_nonvital[hour],
                        "p_pv_mw": p_pv[hour],
                        "v_kn": normal_result["speeds"][hour],
                        "p_propulsion_mw": normal_result["p_propulsion"][hour],
                        "p_g1_mw": 0.0
                        if status != "RA_LSHADE_FEASIBLE"
                        else fault["p_g1"][index],
                        "p_g2_mw": fault["p_g2"][index],
                        "p_ess_mw": fault["p_ess"][index],
                        "energy_ess_mwh": fault["energy_ess"][index],
                        "p_shed_mw": fault["p_shed"][index],
                        "p_vital_shortfall_mw": fault["p_vital_shortfall"][index],
                        "balance_residual_mw": ""
                        if status != "RA_LSHADE_FEASIBLE"
                        else fault["balance_residual"][index],
                    }
                )

    output_dir = base_dir / "fault_time_scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fault_time_results.csv", result_rows)
    write_csv(output_dir / "fault_time_hourly.csv", hourly_rows)
    write_summary(output_dir / "summary.md", result_rows)
    return result_rows


def write_summary(path, rows):
    lookup = {
        (row["strategy"], row["fault_start_hour"]): row for row in rows
    }
    lines = [
        "# 代表性设备参数下的故障时刻扫描",
        "",
        "| 参数 | 取值 |",
        "|---|---:|",
        f"| ESS 容量 / 最大功率 | {ESS_CAPACITY_MWH:g} MWh / {model.ESS_POWER_MAX:g} MW |",
        f"| G2 最大出力 / 爬坡 | {G2_MAX_MW:g} MW / {model.G2_RAMP:g} MW/h |",
        f"| G1 故障后最大出力 | {G1_FAULT_MAX_MW:g} MW |",
        f"| 故障持续时间 | {FAULT_DURATION} h |",
        f"| 故障起点 | {', '.join(str(value) for value in FAULT_START_HOURS)} h |",
        f"| P1/P2 算法 | RA-LSHADE (`NP={model.NP_MAX}->{model.NP_MIN}, K={model.MAX_ITERATIONS}`) |",
        f"| 重试种子 | {', '.join(str(value) for value in RETRY_SEEDS)} |",
        "",
        "| 故障区间 | No reserve 状态 | No reserve 非重要失负荷 | No reserve 重要负荷缺额 | Dynamic 状态 | Dynamic 非重要失负荷 | Dynamic 重要负荷缺额 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for start_hour in FAULT_START_HOURS:
        no = lookup[("no_reserve", start_hour)]
        dynamic = lookup[("dynamic_reserve", start_hour)]
        lines.append(
            f"| {start_hour}–{start_hour + FAULT_DURATION - 1} h | "
            f"{no['status']} | "
            f"{no['shed_energy_mwh']:.6f} MWh ({100 * no['loss_rate']:.3f}%) | "
            f"{no['vital_shortfall_energy_mwh']:.6f} MWh | "
            f"{dynamic['status']} | "
            f"{dynamic['shed_energy_mwh']:.6f} MWh ({100 * dynamic['loss_rate']:.3f}%) | "
            f"{dynamic['vital_shortfall_energy_mwh']:.6f} MWh |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    rows = run_scan(base_dir)
    worst = max(rows, key=lambda row: row["loss_rate"])
    print(
        f"worst: {worst['strategy']} at hour={worst['fault_start_hour']}, "
        f"shed={worst['shed_energy_mwh']:.6f} MWh, "
        f"loss_rate={100 * worst['loss_rate']:.3f}%"
    )


if __name__ == "__main__":
    main()
